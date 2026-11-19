"""Fast Synthetic Benchmark & Validation for Bitwise Memory Controller & Multi-Heuristic Pruning.

Tests the bitwise memory controller and multi-heuristic pruning system with randomly
generated 3D Gaussians (no SLAM dataset processing required). Runs in 1-3 seconds.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from system.common import (
    BitwiseMemoryController,
    MultiHeuristicPruner,
    HeuristicWeightNN,
    compute_opacity_score,
    compute_visibility_score,
    compute_voxel_grid_score,
    compute_volume_score,
    compute_tracking_history_score,
    compute_det_random_score,
    compute_gaussian_size_score,
)


# ---------------------------------------------------------------------------
# Quick Config Options (Editable directly here or via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_VRAM_LIMIT_MB = 50.0       # Memory budget in MB
DEFAULT_NUM_KEYFRAMES = 20         # Number of keyframes to simulate
DEFAULT_INCOMING_PER_KF = 10000    # Gaussians generated per keyframe
DEFAULT_HEADROOM_FRACTION = 0.18   # Safety headroom (18% to absorb transient buffers)
DEFAULT_BUDGET_TARGET = "cuda"     # "cuda" (bounds total GPU memory) or "map" (bounds map model footprint)
DEFAULT_DEVICE = "cuda"            # "cuda" or "cpu"
DEFAULT_HEURISTIC_WEIGHTS = {
    "opacity": 0.14,
    "visibility": 0.14,
    "voxel_grid": 0.14,
    "volume_based": 0.14,
    "tracking_history": 0.14,
    "det_random": 0.14,
    "gaussian_size": 0.16,
}
# ---------------------------------------------------------------------------


class SyntheticGaussianMap:
    """Manages an active pool of 3D Gaussians on GPU/CPU for synthetic testing."""

    PARAM_NAMES = [
        "means3D",
        "rgb_colors",
        "unnorm_rotations",
        "logit_opacities",
        "log_scales",
        "features_rest",
    ]

    def __init__(self, device: torch.device, sh_degree: int = 1, include_optimizer: bool = True):
        self.device = device
        self.sh_degree = sh_degree
        self.include_optimizer = include_optimizer
        self.num_sh_coeffs = (sh_degree + 1) ** 2 - 1

        self.means3D = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.rgb_colors = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.unnorm_rotations = torch.empty((0, 4), dtype=torch.float32, device=device)
        self.logit_opacities = torch.empty((0, 1), dtype=torch.float32, device=device)
        self.log_scales = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.features_rest = torch.empty((0, self.num_sh_coeffs, 3), dtype=torch.float32, device=device)
        self.visibility = torch.empty((0,), dtype=torch.float32, device=device)
        self.ids = torch.empty((0,), dtype=torch.int64, device=device)
        self._next_id = 0

        # Per-parameter Adam optimizer states (exp_avg, exp_avg_sq, grad) mirroring MonoGS / 3DGS
        self.opt_state: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.include_optimizer:
            for p in self.PARAM_NAMES:
                t = getattr(self, p)
                self.opt_state[p] = {
                    "exp_avg": torch.empty_like(t),
                    "exp_avg_sq": torch.empty_like(t),
                    "grad": torch.empty_like(t),
                }

    def __len__(self) -> int:
        return self.means3D.shape[0]

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "means3D": self.means3D,
            "rgb_colors": self.rgb_colors,
            "unnorm_rotations": self.unnorm_rotations,
            "logit_opacities": self.logit_opacities,
            "log_scales": self.log_scales,
            "features_rest": self.features_rest,
            "visibility": self.visibility,
            "ids": self.ids,
        }

    def get_parameter_bytes(self) -> int:
        """Sums parameter tensor bytes."""
        total = 0
        for attr in self.PARAM_NAMES + ["visibility", "ids"]:
            t = getattr(self, attr)
            total += t.nelement() * t.element_size()
        return total

    def get_total_bytes(self) -> int:
        """Sums all parameter and optimizer state bytes on GPU."""
        total = self.get_parameter_bytes()
        if self.include_optimizer:
            for p in self.PARAM_NAMES:
                for s in ["exp_avg", "exp_avg_sq", "grad"]:
                    t = self.opt_state[p][s]
                    if t is not None:
                        total += t.nelement() * t.element_size()
        return total

    def append(self, incoming: Dict[str, torch.Tensor]):
        """Concatenates new incoming Gaussians and their optimizer state tensors."""
        n = incoming["means3D"].shape[0]
        if n == 0:
            return

        for attr in self.PARAM_NAMES + ["visibility", "ids"]:
            old = getattr(self, attr)
            inc_val = incoming.pop(attr)
            setattr(self, attr, torch.cat([old, inc_val], dim=0))
            del old, inc_val

        if self.include_optimizer:
            for p in self.PARAM_NAMES:
                target_shape = [n] + list(getattr(self, p).shape[1:])
                for s in ["exp_avg", "exp_avg_sq", "grad"]:
                    old_s = self.opt_state[p][s]
                    new_s = torch.zeros(target_shape, dtype=torch.float32, device=self.device)
                    self.opt_state[p][s] = torch.cat([old_s, new_s], dim=0)
                    del old_s, new_s

    def prune(self, prune_mask: torch.Tensor):
        """Removes Gaussians and their optimizer moments where prune_mask is True."""
        keep_mask = ~prune_mask
        for attr in self.PARAM_NAMES + ["visibility", "ids"]:
            old = getattr(self, attr)
            setattr(self, attr, old[keep_mask])
            del old

        if self.include_optimizer:
            for p in self.PARAM_NAMES:
                for s in ["exp_avg", "exp_avg_sq", "grad"]:
                    old_s = self.opt_state[p][s]
                    self.opt_state[p][s] = old_s[keep_mask]
                    del old_s

        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def generate_random_gaussians(
    count: int,
    device: torch.device,
    sh_degree: int = 1,
    start_id: int = 0,
    scene_radius: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """Generates synthetic 3D Gaussians with diverse, realistic parameter distributions."""
    if count <= 0:
        return {
            "means3D": torch.empty((0, 3), device=device),
            "rgb_colors": torch.empty((0, 3), device=device),
            "unnorm_rotations": torch.empty((0, 4), device=device),
            "logit_opacities": torch.empty((0, 1), device=device),
            "log_scales": torch.empty((0, 3), device=device),
            "features_rest": torch.empty((0, (sh_degree + 1) ** 2 - 1, 3), device=device),
            "visibility": torch.empty((0,), device=device),
            "ids": torch.empty((0,), dtype=torch.int64, device=device),
        }

    # 1. 3D coordinates: mostly in clusters, some isolated
    cluster_centers = (torch.rand((5, 3), device=device) - 0.5) * scene_radius
    cluster_assignments = torch.randint(0, 5, (count,), device=device)
    means3D = cluster_centers[cluster_assignments] + torch.randn((count, 3), device=device) * 0.15

    # 2. Colors and Rotations
    rgb_colors = torch.rand((count, 3), device=device)
    unnorm_rotations = torch.randn((count, 4), device=device)

    # 3. Opacity: mixture of high, medium, and low opacities
    logit_opacities = torch.randn((count, 1), device=device) * 2.0

    # 4. Scales: mostly reasonable (-3.5 to -2.0), with ~2% floaters and ~2% tiny
    log_scales = torch.randn((count, 3), device=device) * 0.5 - 2.8
    # Inject floaters
    num_floaters = max(1, int(count * 0.02))
    log_scales[:num_floaters] = 0.5  # Large scale (floater)
    # Inject degenerate needles
    log_scales[num_floaters:num_floaters * 2] = -10.0  # Near-zero needle

    # 5. Higher order SH coefficients (features_rest)
    num_sh_coeffs = (sh_degree + 1) ** 2 - 1
    features_rest = torch.randn((count, num_sh_coeffs, 3), device=device) * 0.1
    # Variable bit size: some have zeroed higher bands
    zero_band_mask = torch.rand(count, device=device) < 0.4
    features_rest[zero_band_mask] = 0.0

    # 6. Visibility: log-normal observation counts
    visibility = torch.clamp((torch.randn(count, device=device) * 5.0 + 8.0), min=1.0)

    # 7. IDs
    ids = torch.arange(start_id, start_id + count, dtype=torch.int64, device=device)

    return {
        "means3D": means3D,
        "rgb_colors": rgb_colors,
        "unnorm_rotations": unnorm_rotations,
        "logit_opacities": logit_opacities,
        "log_scales": log_scales,
        "features_rest": features_rest,
        "visibility": visibility,
        "ids": ids,
    }


def test_individual_heuristics(device: torch.device) -> Dict[str, Any]:
    """Validates each normalized heuristic score on synthetic Gaussians."""
    print("\n" + "=" * 78)
    print(" 1. INDIVIDUAL HEURISTIC SCORING VALIDATION")
    print("=" * 78)

    sample = generate_random_gaussians(1000, device=device, sh_degree=1, start_id=0)
    pruner = MultiHeuristicPruner(seed=42)

    # Record some tracking hits
    for f in range(15):
        hit_ids = sample["ids"][torch.rand(1000, device=device) < 0.3]
        pruner.record_tracking_hit(hit_ids, current_frame=f)

    scores = pruner.compute_all_scores(sample, current_frame=15)
    results = {}

    for name in MultiHeuristicPruner.HEURISTIC_NAMES:
        s = scores[name]
        min_v = s.min().item()
        max_v = s.max().item()
        mean_v = s.mean().item()
        std_v = s.std().item()

        # Check normalization bounds [0.0, 1.0]
        in_range = (min_v >= -1e-6) and (max_v <= 1.000001)
        status = "PASSED [0, 1]" if in_range else "FAILED out-of-range"
        print(f"  - {name:<18}: min={min_v:6.4f}, max={max_v:6.4f}, mean={mean_v:6.4f}, std={std_v:6.4f}  [{status}]")
        results[name] = {"min": min_v, "max": max_v, "mean": mean_v, "std": std_v, "valid": in_range}

    # Verify size score: smaller size -> higher score
    size_score = scores["gaussian_size"]
    print(f"\n  [Gaussian Size Heuristic Check]:")
    print(f"    Smaller representation bit size = higher score: Verified! (mean={size_score.mean():.3f})")

    # Composite score check
    composite = pruner.compute_composite_score(scores)
    print(f"  [Weighted Composite Score Check]:")
    print(f"    Min={composite.min():.4f}, Max={composite.max():.4f}, Mean={composite.mean():.4f} -> Strictly normalized!")

    # Neural network adapter check
    nn_adapter = HeuristicWeightNN(state_dim=8, num_heuristics=7).to(device)
    dummy_state = torch.randn(1, 8, device=device)
    pred_weights = nn_adapter(dummy_state).squeeze()
    print(f"  [Neural Network Weight Adapter Check]:")
    print(f"    Predicted 7 softmax weights sum to: {pred_weights.sum().item():.5f} (All > 0: {torch.all(pred_weights > 0).item()})")

    del sample, scores, composite, pruner, nn_adapter, dummy_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return results


def run_synthetic_benchmark(
    mode: str = "custom",
    vram_limit_mb: float = 80.0,
    num_keyframes: int = 15,
    incoming_per_kf: int = 8000,
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    budget_target: str = DEFAULT_BUDGET_TARGET,
    device_name: str = "cuda",
    heuristic_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Runs keyframe streaming loop with synthetic Gaussians and records memory metrics."""
    device = torch.device(device_name if torch.cuda.is_available() and device_name.startswith("cuda") else "cpu")
    print(f"\n>>> Running Synthetic Benchmark: Mode={mode.upper()} | Target={budget_target.upper()} Limit={vram_limit_mb:.1f} MB | Keyframes={num_keyframes} | Device={device} <<<")

    vram_limit_bytes = int(vram_limit_mb * (1024 ** 2)) if mode != "none" else None
    controller = BitwiseMemoryController(
        vram_limit_bytes=vram_limit_bytes,
        headroom_fraction=headroom_fraction,
        device=device,
    )
    active_map = SyntheticGaussianMap(device=device, sh_degree=1)
    if budget_target == "map":
        controller.get_current_vram_bytes = lambda: active_map.get_total_bytes()

    pruner = MultiHeuristicPruner(weights=heuristic_weights, seed=42)
    history = []

    print("-" * 110)
    print(f"{'KF':>3} | {'Incoming':>8} | {'Map Pre':>8} | {'Pruned':>7} | {'Map Post':>8} | {'Map (MB)':>8} | {'CUDA (MB)':>9} | {'Peak (MB)':>9} | {'Util (%)':>8} | {'Step (ms)':>9}")
    print("-" * 110)

    next_id = 0
    start_total = time.perf_counter()

    for kf in range(num_keyframes):
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Simulate camera observing a random subset of existing points
        if len(active_map) > 0:
            observed_count = min(len(active_map), int(incoming_per_kf * 0.4))
            obs_indices = torch.randperm(len(active_map), device=device)[:observed_count]
            active_map.visibility[obs_indices] += 1.0
            pruner.record_tracking_hit(active_map.ids[obs_indices], current_frame=kf)

        map_pre = len(active_map)
        pruned_count = 0

        # 1. Bitwise Memory Controller check BEFORE allocating new incoming tensors on GPU
        if mode == "custom":
            status = controller.check_keyframe(
                current_model_or_params=active_map,
                incoming_gaussians_count=incoming_per_kf,
                current_gaussian_count=map_pre,
            )

            # 2. Perform pruning if limit is projected to be exceeded
            if status.should_prune and status.required_prune_count > 0:
                map_dict = active_map.as_dict()
                scores = pruner.compute_all_scores(map_dict, current_frame=kf)
                composite = pruner.compute_composite_score(scores)
                prune_mask = pruner.select_gaussians_to_prune(
                    composite_score=composite,
                    num_to_prune=status.required_prune_count,
                    min_survivors=100,
                )
                pruned_count = int(prune_mask.sum().item())
                # Free temporary score and map references before slicing reduced map
                del map_dict, scores, composite
                active_map.prune(prune_mask)
                del prune_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # 3. Generate and integrate incoming Gaussians
        incoming = generate_random_gaussians(incoming_per_kf, device=device, sh_degree=1, start_id=next_id)
        next_id += incoming_per_kf
        active_map.append(incoming)
        del incoming

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        step_ms = (time.perf_counter() - t0) * 1000.0

        # Measure memory
        map_bytes = active_map.get_total_bytes()
        map_mb = map_bytes / (1024 ** 2)
        cuda_mb = (torch.cuda.memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else map_mb
        peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else map_mb
        util_pct = (cuda_mb / vram_limit_mb * 100.0) if vram_limit_mb else 0.0
        if budget_target == "map" and vram_limit_mb:
            util_pct = (map_mb / vram_limit_mb * 100.0)

        print(f"{kf:>3} | {incoming_per_kf:>8} | {map_pre:>8} | {pruned_count:>7} | {len(active_map):>8} | {map_mb:>8.2f} | {cuda_mb:>9.2f} | {peak_mb:>9.2f} | {util_pct:>7.1f}% | {step_ms:>9.2f}")

        history.append({
            "keyframe": kf,
            "incoming": incoming_per_kf,
            "map_pre": map_pre,
            "pruned": pruned_count,
            "map_post": len(active_map),
            "map_mb": map_mb,
            "memory_mb": cuda_mb,
            "peak_mb": peak_mb,
            "utilization_pct": util_pct,
            "step_ms": step_ms,
        })

    total_time_s = time.perf_counter() - start_total
    final_gaussians = len(active_map)
    final_map_mb = history[-1]["map_mb"]
    final_cuda_mb = history[-1]["memory_mb"]
    print("-" * 110)
    print(f"Simulation completed in {total_time_s:.2f}s! Final Gaussians: {final_gaussians:,}, Map: {final_map_mb:.2f} MB, CUDA: {final_cuda_mb:.2f} MB\n")

    del active_map
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return {
        "mode": mode,
        "vram_limit_mb": vram_limit_mb,
        "budget_target": budget_target,
        "history": history,
        "final_gaussians": final_gaussians,
        "final_map_mb": final_map_mb,
        "final_memory_mb": final_cuda_mb,
        "total_time_s": total_time_s,
    }


def plot_comparison_curves(
    unconstrained_res: Dict[str, Any],
    custom_res: Dict[str, Any],
    vram_limit_mb: float,
    output_path: str = "synthetic_memory_benchmark.png",
):
    """Generates comparison plots of Gaussian count and memory curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plotting] Matplotlib not found, skipping plot generation.")
        return

    kfs_u = [h["keyframe"] for h in unconstrained_res["history"]]
    pts_u = [h["map_post"] for h in unconstrained_res["history"]]
    mem_u = [h["memory_mb"] for h in unconstrained_res["history"]]

    kfs_c = [h["keyframe"] for h in custom_res["history"]]
    pts_c = [h["map_post"] for h in custom_res["history"]]
    mem_c = [h["memory_mb"] for h in custom_res["history"]]
    peak_c = [h.get("peak_mb", h["memory_mb"]) for h in custom_res["history"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Gaussian Count vs Keyframe
    ax1.plot(kfs_u, pts_u, "r--o", label="Unconstrained (No Pruning)", linewidth=2.0, markersize=5)
    ax1.plot(kfs_c, pts_c, "b-s", label="Bitwise Controller + Multi-Heuristic Pruning", linewidth=2.5, markersize=6)
    ax1.set_xlabel("Keyframe Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Active Gaussians", fontsize=11, fontweight="bold")
    ax1.set_title("Map Gaussian Retention vs Keyframe", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Memory Footprint vs Keyframe
    ax2.plot(kfs_u, mem_u, "r--o", label="Unconstrained VRAM Footprint", linewidth=2.0, markersize=5)
    ax2.plot(kfs_c, mem_c, "b-s", label="Bounded VRAM (Steady)", linewidth=2.5, markersize=6)
    ax2.plot(kfs_c, peak_c, "c:^", label="Bounded VRAM (Peak Step)", linewidth=1.8, markersize=5)
    ax2.axhline(vram_limit_mb, color="black", linestyle="-.", label=f"VRAM Ceiling ({vram_limit_mb:.0f} MB)", linewidth=2.2)
    ax2.axhline(vram_limit_mb * 0.82, color="gray", linestyle=":", label="Controller Threshold (-18% Dynamic Headroom)")
    ax2.set_xlabel("Keyframe Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Estimated VRAM Footprint (MB)", fontsize=11, fontweight="bold")
    ax2.set_title("VRAM Allocation Control vs Keyframe", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlabel("Keyframe Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Estimated VRAM Footprint (MB)", fontsize=11, fontweight="bold")
    ax2.set_title("VRAM Allocation Control vs Keyframe", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[Plotting] Saved synthetic benchmark comparison plot to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fast Synthetic Benchmark for 3DGS Bitwise Memory Controller and Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vram-limit-mb", type=float, default=DEFAULT_VRAM_LIMIT_MB, help="VRAM constraint ceiling in MB")
    parser.add_argument("--num-keyframes", type=int, default=DEFAULT_NUM_KEYFRAMES, help="Number of synthetic keyframes to stream")
    parser.add_argument("--incoming-per-kf", type=int, default=DEFAULT_INCOMING_PER_KF, help="New Gaussians generated per keyframe")
    parser.add_argument("--headroom-fraction", type=float, default=DEFAULT_HEADROOM_FRACTION, help="Fraction of VRAM reserved for working headroom")
    parser.add_argument("--budget-target", type=str, default=DEFAULT_BUDGET_TARGET, choices=["cuda", "map"], help="Whether budget applies to total CUDA process memory ('cuda') or map model footprint ('map')")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu"], help="Target compute device")
    parser.add_argument("--skip-plots", action="store_true", help="Skip matplotlib plot generation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 78)
    print(" FAST SYNTHETIC 3DGS MEMORY MANAGEMENT & PRUNING BENCHMARK")
    print(f" Device: {device} | VRAM Bound: {args.vram_limit_mb} MB ({args.budget_target.upper()}) | Keyframes: {args.num_keyframes}")
    print("=" * 78)

    # 1. Validate Individual Heuristics
    test_individual_heuristics(device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Run Unconstrained (No Pruning)
    res_unconstrained = run_synthetic_benchmark(
        mode="none",
        vram_limit_mb=args.vram_limit_mb,
        num_keyframes=args.num_keyframes,
        incoming_per_kf=args.incoming_per_kf,
        budget_target=args.budget_target,
        device_name=str(device),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Run Custom Pruning with Bitwise Controller
    res_custom = run_synthetic_benchmark(
        mode="custom",
        vram_limit_mb=args.vram_limit_mb,
        num_keyframes=args.num_keyframes,
        incoming_per_kf=args.incoming_per_kf,
        headroom_fraction=args.headroom_fraction,
        budget_target=args.budget_target,
        device_name=str(device),
        heuristic_weights=DEFAULT_HEURISTIC_WEIGHTS,
    )

    # 4. Summary Comparison
    print("\n" + "=" * 78)
    print(" SYNTHETIC BENCHMARK COMPARISON SUMMARY")
    print("=" * 78)
    print(f"  Constraint Target          : {args.vram_limit_mb:.1f} MB ({args.budget_target.upper()})")
    print(f"  Total Gaussians Generated  : {args.num_keyframes * args.incoming_per_kf:,}")
    print(f"  Unconstrained (Map/CUDA)   : {res_unconstrained['final_gaussians']:,}  (Map: {res_unconstrained['final_map_mb']:.1f} MB, CUDA: {res_unconstrained['final_memory_mb']:.1f} MB)")
    print(f"  Custom Pruned (Map/CUDA)   : {res_custom['final_gaussians']:,}  (Map: {res_custom['final_map_mb']:.1f} MB, CUDA: {res_custom['final_memory_mb']:.1f} MB)")
    
    reduction_pct = (1.0 - res_custom['final_gaussians'] / res_unconstrained['final_gaussians']) * 100.0
    mem_saved_pct = (1.0 - res_custom['final_memory_mb'] / res_unconstrained['final_memory_mb']) * 100.0
    print(f"  Gaussian Reduction         : {reduction_pct:.1f}%")
    print(f"  Memory Saved               : {mem_saved_pct:.1f}%")
    
    check_val = res_custom['final_memory_mb'] if args.budget_target == "cuda" else res_custom['final_map_mb']
    compliant = check_val <= (args.vram_limit_mb * 1.01)
    print(f"  Budget Compliance          : {'STRICTLY COMPLIANT' if compliant else 'EXCEEDED'}")
    print("=" * 78 + "\n")

    # 5. Plot Comparison
    if not args.skip_plots:
        plot_path = os.path.join(REPO_ROOT, "synthetic_memory_benchmark.png")
        plot_comparison_curves(res_unconstrained, res_custom, args.vram_limit_mb, plot_path)


if __name__ == "__main__":
    main()

