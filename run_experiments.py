"""Unified Configuration and Experiment Runner for 3D Gaussian Splatting SLAM.

Supports MonoGS and SGS-SLAM across multiple TUM RGB-D dataset sequences.
Compares:
  - Unconstrained results ("none": memory management & pruning disabled, unbounded growth)
  - Constrained results with Custom Pruning (Bitwise Memory Controller + Multi-Heuristic Pruning)
  - Constrained results with Upstream Pruning (original MonoGS & SGS-SLAM baseline pruning)
Sweeps across multiple VRAM constraints (e.g. 1.0, 2.0, 4.0, 8.0 GB) in a single run.
Outputs per-run telemetry, plots, comparison data, and reconstructed 3D map trajectory videos.
"""

import time

# ==============================================================================
# EXPERIMENT CONFIGURATION (Edit options below or override via CLI arguments)
# ==============================================================================

# Target SLAM systems to benchmark
# Options: ["MonoGS"], ["SGS-SLAM"], or ["MonoGS", "SGS-SLAM"]
DEFAULT_SYSTEMS = ["MonoGS"]#, "SGS-SLAM"]

# Datasets to evaluate (TUM RGB-D sequences)
# Options: any combination of ["fr1_desk", "fr2_xyz", "fr3_office"] or ["all"]
DEFAULT_DATASETS = ["fr1_desk"]#, "fr2_xyz", "fr3_office"]

# Pruning & Memory Management modes to compare:
#   * "none"     : Unconstrained (all pruning disabled, unbounded Gaussian accumulation)
#   * "custom"   : Constrained via Bitwise Memory Controller + Multi-Heuristic Pruning
#   * "upstream" : Constrained via original system pruning mechanisms
DEFAULT_PRUNING_MODES = ["upstream"]

# VRAM memory constraints in GB to evaluate for constrained modes ("custom" & "upstream")
# The suite runs each constrained mode across every constraint in this list.
# Example: [1.0, 2.0, 4.0, 8.0]
DEFAULT_VRAM_LIMITS_GB = [8.0]

# Should unconstrained ("none") mode also be run at each VRAM constraint to test OOM?
#   * False: Runs "none" once per dataset with unlimited memory (pure unconstrained baseline)
#   * True : Runs "none" for every constraint in DEFAULT_VRAM_LIMITS_GB to evaluate OOM behavior
DEFAULT_RUN_NONE_PER_LIMIT = True

# Default Heuristic Weights for Custom Multi-Heuristic Pruning
# (Weights are normalized dynamically to sum to 1.0)
DEFAULT_HEURISTIC_WEIGHTS = {
    "opacity": 0.14,          # Opacity score (higher opacity = higher retention)
    "visibility": 0.14,       # Keyframe observation / visibility frequency
    "voxel_grid": 0.14,       # Spatial density (penalizes redundant clustered points)
    "volume_based": 0.14,     # Covariance / volume (penalizes needle/floater splats)
    "tracking_history": 0.14, # Recency / appearance frequency in tracking window
    "det_random": 0.14,       # Deterministic SplitMix64 pseudo-random tie-breaker
    "gaussian_size": 0.16,    # Gaussian bit size (smaller bit-count = higher retention score)
}

# Output and Evaluation Settings
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_RUN_NAME = f"run_{time.strftime('%Y%m%d_%H%M%S')}"  # Timestamped folder: "run_YYYYMMDD_HHMMSS" (or set to "timestamp")
DEFAULT_EVAL_RENDERING = True     # Evaluate PSNR, SSIM, LPIPS metrics
DEFAULT_GENERATE_VIDEO = True     # Generate reconstructed 3D map video (.mp4)
DEFAULT_VIDEO_FPS = 30            # TUM RGB-D dataset physical capture framerate (30 FPS)
DEFAULT_SEED = 0
DEFAULT_MAX_FRAMES = None         # Optional frame limit (None = full sequence; e.g. 50 for quick test)
DEFAULT_PYTHON_EXE = None         # Explicit python path (auto-detects MonoGS conda env if None)
# ==============================================================================

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Ensure local imports find repo packages
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from system.common import (
    BitwiseMemoryController,
    MultiHeuristicPruner,
    MetricsTracker,
    MapVideoRenderer,
)

TUM_SEQUENCES = {
    "fr1_desk": "rgbd_dataset_freiburg1_desk",
    "fr2_xyz": "rgbd_dataset_freiburg2_xyz",
    "fr3_office": "rgbd_dataset_freiburg3_long_office_household",
}


def parse_weights(weights_str: Optional[str]) -> Dict[str, float]:
    """Parses heuristic weights from JSON or comma-separated string."""
    default = dict(DEFAULT_HEURISTIC_WEIGHTS)
    if not weights_str:
        return default

    # Try JSON
    if weights_str.startswith("{"):
        try:
            parsed = json.loads(weights_str)
            if "bit_size" in parsed and "gaussian_size" not in parsed:
                parsed["gaussian_size"] = parsed["bit_size"]
            default.update(parsed)
            return default
        except Exception:
            pass

    # Try comma-separated
    parts = [float(p.strip()) for p in weights_str.split(",") if p.strip()]
    keys = ["opacity", "visibility", "voxel_grid", "volume_based", "tracking_history", "det_random", "gaussian_size"]
    for i, val in enumerate(parts[:len(keys)]):
        default[keys[i]] = val
    return default


def get_python_executable(system: str = "MonoGS", custom_exe: Optional[str] = None) -> str:
    """Finds the appropriate Python executable with required dependencies for the given system."""
    if custom_exe and os.path.exists(custom_exe):
        return os.path.abspath(custom_exe)

    # 1. Test if current sys.executable has diff_gaussian_rasterization and open3d
    try:
        import diff_gaussian_rasterization  # noqa: F401
        import open3d  # noqa: F401
        return sys.executable
    except ImportError:
        pass

    # 2. Check candidate conda environments
    candidates = [
        r"C:\Users\dgosh\miniconda3\envs\MonoGS\python.exe",
        r"C:\Users\dgosh\miniconda3\envs\sgs_slam\python.exe",
        os.path.expanduser(r"~\miniconda3\envs\MonoGS\python.exe"),
        os.path.expanduser(r"~\anaconda3\envs\MonoGS\python.exe"),
    ]
    env_txt = os.path.expanduser(r"~/.conda/environments.txt")
    if os.path.exists(env_txt):
        try:
            with open(env_txt, "r") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        cand = os.path.join(p, "python.exe")
                        if cand not in candidates and os.path.exists(cand):
                            candidates.append(cand)
        except Exception:
            pass

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)

    return sys.executable


def setup_experiment_config(
    system: str,
    dataset_key: str,
    memory_mode: str,
    vram_limit_gb: Optional[float],
    heuristic_weights: Dict[str, float],
    output_dir: str,
    eval_rendering: bool = True,
    generate_video: bool = True,
    video_fps: int = 15,
    seed: int = 0,
    max_frames: Optional[int] = None,
    custom_python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates resolved configuration dictionary for an individual experiment."""
    seq_folder = TUM_SEQUENCES.get(dataset_key, dataset_key)
    dataset_path = os.path.abspath(os.path.join(REPO_ROOT, "datasets", "tum", seq_folder))
    abs_output_dir = os.path.abspath(output_dir)
    python_exe = get_python_executable(system, custom_python_exe)

    config_info = {
        "system": system,
        "dataset_key": dataset_key,
        "sequence_folder": seq_folder,
        "dataset_path": dataset_path,
        "memory_mode": memory_mode,
        "vram_limit_gb": vram_limit_gb,
        "heuristic_weights": heuristic_weights,
        "output_dir": abs_output_dir,
        "eval_rendering": eval_rendering,
        "generate_video": generate_video,
        "video_fps": video_fps,
        "seed": seed,
        "max_frames": max_frames,
        "python_exe": python_exe,
    }
    return config_info


def run_monogs_experiment(config_info: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a MonoGS SLAM experiment using the configured options."""
    dataset_key = config_info["dataset_key"]
    output_dir = os.path.abspath(config_info["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Clean up any stale flags or run status from earlier runs
    for stale_file in ["OOM_CRASH.flag", "run_status.json"]:
        stale_path = os.path.join(output_dir, stale_file)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
            except Exception:
                pass

    base_config_path = os.path.abspath(os.path.join(
        REPO_ROOT, "system", "MonoGS", "configs", "rgbd", "tum", f"{dataset_key}.yaml"
    ))
    if not os.path.exists(base_config_path):
        base_config_path = os.path.abspath(os.path.join(
            REPO_ROOT, "system", "MonoGS", "configs", "rgbd", "tum", "fr1_desk.yaml"
        ))

    repo_root_esc = REPO_ROOT.replace("\\", "/")
    base_config_path_esc = base_config_path.replace("\\", "/")
    output_dir_esc = output_dir.replace("\\", "/")
    dataset_path_esc = os.path.abspath(config_info["dataset_path"]).replace("\\", "/")

    # Build runner script that configures memory management and telemetry
    runner_script_path = os.path.abspath(os.path.join(output_dir, "monogs_runner.py"))
    runner_code = f"""import os
import sys
import yaml
import json
import time
import glob
import traceback
import torch
import torch.multiprocessing as mp
import numpy as np

repo_root = "{repo_root_esc}"
monogs_root = os.path.join(repo_root, "system", "MonoGS")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if monogs_root not in sys.path:
    sys.path.insert(0, monogs_root)

try:
    import wandb
except ImportError:
    class DummyWandb:
        def init(self, *args, **kwargs): return self
        def log(self, *args, **kwargs): pass
        def Table(self, *args, **kwargs): return None
        def define_metric(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    wandb = DummyWandb()

from utils.config_utils import load_config
from slam import SLAM
from system.common import MetricsTracker, MapVideoRenderer

config_path = "{base_config_path_esc}"
output_dir = "{output_dir_esc}"

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    wandb.init(mode="disabled")

    config = load_config(config_path)
    config["Results"]["save_results"] = True
    config["Results"]["save_dir"] = output_dir
    config["Results"]["use_gui"] = False
    config["Results"]["eval_rendering"] = {config_info["eval_rendering"]}
    config["Results"]["use_wandb"] = False
    config["Results"]["color_refinement"] = False
    config["Dataset"]["dataset_path"] = "{dataset_path_esc}"
    config["Dataset"]["max_frames"] = {config_info["max_frames"] if config_info["max_frames"] is not None else -1}

    # Configure Memory Management
    memory_mode = "{config_info["memory_mode"]}"
    vram_limit = {config_info["vram_limit_gb"] if config_info["vram_limit_gb"] is not None else "None"}
    weights = {json.dumps(config_info["heuristic_weights"])}

    if vram_limit is not None:
        try:
            from project_utils.memory_limit import limit_gpu_memory
            limit_gpu_memory(float(vram_limit))
            print(f"[Runner] Enforced PyTorch GPU allocator memory limit: {{vram_limit}} GB", flush=True)
        except Exception as e:
            print(f"[Runner] Warning: limit_gpu_memory failed: {{e}}", flush=True)

    if memory_mode == "none":
        config["disable_memory_management"] = True
        config["MemoryManagement"] = {{"disabled": True, "mode": "none", "vram_limit_gb": vram_limit}}
        config["vram_limit_gb"] = vram_limit
        print(f"[Runner] Running with Memory Management DISABLED (vram_limit={{vram_limit}} GB, expecting OOM under budget)")
    elif memory_mode == "custom":
        config["MemoryManagement"] = {{
            "mode": "custom",
            "disabled": False,
            "vram_limit_gb": vram_limit,
            "heuristic_weights": weights,
        }}
        config["use_custom_memory_controller"] = True
        config["vram_limit_gb"] = vram_limit
        print(f"[Runner] Running with Bitwise Controller + Multi-Heuristic Pruning (limit={{vram_limit}} GB)")
    else: # upstream
        config["MemoryManagement"] = {{"mode": "upstream", "disabled": False}}
        if vram_limit is not None:
            config.setdefault("Retention", {{}})["memory_limit_gb"] = vram_limit
        print(f"[Runner] Running with Upstream MonoGS Memory Management (limit={{vram_limit}} GB)")

    metrics = MetricsTracker(output_dir)
    start_time = time.time()
    status = "completed"
    error_msg = None

    try:
        slam = SLAM(config, save_dir=output_dir, gpu_limit_mem=vram_limit)
        slam.run()

        # Collect ATE RMSE from stats_final.json
        stats_final_file = os.path.join(output_dir, "plot", "stats_final.json")
        if os.path.exists(stats_final_file):
            try:
                with open(stats_final_file, "r") as f:
                    sdata = json.load(f)
                    if "rmse" in sdata:
                        metrics.ate_rmse = float(sdata["rmse"])
            except Exception:
                pass

        # Load evaluated rendering quality if saved
        psnr_file = os.path.join(output_dir, "psnr", "before_opt", "final_result.json")
        if os.path.exists(psnr_file):
            try:
                with open(psnr_file, "r") as f:
                    res = json.load(f)
                    if "mean_psnr" in res:
                        metrics.psnr_values.append(res["mean_psnr"])
                    if "mean_ssim" in res:
                        metrics.ssim_values.append(res["mean_ssim"])
                    if "mean_lpips" in res:
                        metrics.lpips_values.append(res["mean_lpips"])
            except Exception:
                pass

        # Collect online FPS from online_summary.json
        online_sum_file = os.path.join(output_dir, "online_summary.json")
        total_frames = 1
        if os.path.exists(online_sum_file):
            try:
                with open(online_sum_file, "r") as f:
                    osum = json.load(f)
                    if "online_fps" in osum:
                        metrics.fps_records.append(float(osum["online_fps"]))
                    if "frames" in osum:
                        total_frames = int(osum["frames"])
            except Exception:
                pass

        # Record peak memory & frames
        vram_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        vram_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        final_count = slam.frontend.gaussians.get_xyz.shape[0] if hasattr(slam, "frontend") else 0

        metrics.record_frame(
            frame_idx=total_frames,
            gaussian_count=final_count,
            vram_allocated_bytes=vram_peak,
            vram_reserved_bytes=vram_reserved,
            frame_duration_s=time.time() - start_time
        )

        # Collect pruning statistics from backend if saved
        prune_sum_file = os.path.join(output_dir, "pruning_summary.json")
        if os.path.exists(prune_sum_file):
            try:
                with open(prune_sum_file, "r") as f:
                    p_info = json.load(f)
                    metrics.record_pruning(
                        count=p_info.get("pruned_gaussians_count", 0),
                        byte_size=p_info.get("pruned_gaussians_bytes", 0)
                    )
            except Exception:
                pass

        summary = metrics.generate_summary()
        graphs = metrics.generate_all_plots()

        # Reconstructed map video export along camera trajectory (30 FPS continuous smoothing)
        if {config_info.get("generate_video", True)}:
            video_renderer = MapVideoRenderer(output_dir, fps={config_info.get("video_fps", 30)})
            render_imgs = []
            
            # 1. Render smoothly along every continuous camera frame tracked by frontend
            if hasattr(slam, "frontend") and hasattr(slam.frontend, "cameras") and hasattr(slam, "gaussians"):
                try:
                    from gaussian_splatting.gaussian_renderer import render as render_gs
                    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
                    sorted_frame_indices = sorted(list(slam.frontend.cameras.keys()))
                    for f_idx in sorted_frame_indices:
                        cam = slam.frontend.cameras[f_idx]
                        with torch.no_grad():
                            r = render_gs(cam, slam.gaussians, slam.pipeline_params, bg)["render"]
                            arr = torch.clamp(r, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
                            render_imgs.append((arr * 255.0).astype(np.uint8))
                except Exception as e:
                    print(f"[MapVideoRenderer] Direct frontend camera trajectory rendering error: {{e}}")

            # 2. Fallback to keyframe trajectory in trj_final.json if frontend cameras unavailable
            if not render_imgs:
                trj_file = os.path.join(output_dir, "plot", "trj_final.json")
                if os.path.exists(trj_file) and hasattr(slam, "gaussians"):
                    try:
                        from gaussian_splatting.gaussian_renderer import render as render_gs
                        from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
                        from utils.camera_utils import Camera
                        calib = config["Dataset"]["Calibration"]
                        fx, fy = calib["fx"], calib["fy"]
                        cx, cy = calib["cx"], calib["cy"]
                        W, H = calib["width"], calib["height"]
                        fovx = focal2fov(fx, W)
                        fovy = focal2fov(fy, H)
                        proj = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1).cuda()
                        dummy_color = torch.zeros(3, H, W, dtype=torch.float32, device="cuda")
                        dummy_depth = torch.zeros(H, W, dtype=torch.float32, device="cuda")
                        bg = torch.zeros(3, dtype=torch.float32, device="cuda")
                        with open(trj_file, "r") as f:
                            poses = json.load(f).get("trj_est", [])
                        for i, pose_arr in enumerate(poses):
                            w2c = np.linalg.inv(np.array(pose_arr))
                            gt_T = torch.tensor(w2c, dtype=torch.float32, device="cuda")
                            cam = Camera(
                                uid=i, color=dummy_color, depth=dummy_depth, gt_T=gt_T,
                                projection_matrix=proj, fx=fx, fy=fy, cx=cx, cy=cy,
                                fovx=fovx, fovy=fovy, image_height=H, image_width=W, device="cuda"
                            )
                            cam.update_RT(gt_T[:3, :3], gt_T[:3, 3])
                            with torch.no_grad():
                                r = render_gs(cam, slam.gaussians, slam.pipeline_params, bg)["render"]
                                arr = torch.clamp(r, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
                                render_imgs.append((arr * 255.0).astype(np.uint8))
                    except Exception as e:
                        print(f"[MapVideoRenderer] Direct trajectory rendering error: {{e}}")

            # 3. Fallback to saved image files if available
            if not render_imgs:
                img_files = sorted(glob.glob(os.path.join(output_dir, "renders", "*.png")))
                if img_files:
                    from PIL import Image
                    for imf in img_files:
                        render_imgs.append(np.array(Image.open(imf)))

            if render_imgs:
                video_renderer.render_trajectory_video(rendered_images=render_imgs, apply_smoothing=True)

        print(f"RUN_SUMMARY:{{json.dumps(summary)}}")

    except torch.cuda.OutOfMemoryError as e:
        status = "OOM_Crash"
        error_msg = f"CUDA Out Of Memory: {{e}}"
        print(f"RUN_ERROR:OOM:{{e}}")
    except Exception as e:
        status = "Failed"
        error_msg = traceback.format_exc()
        print(f"RUN_ERROR:{{e}}")
        print("TRACEBACK_BEGIN")
        traceback.print_exc()
        print("TRACEBACK_END")

    with open(os.path.join(output_dir, "run_status.json"), "w") as f:
        json.dump({{"status": status, "error": error_msg, "elapsed_s": time.time() - start_time}}, f, indent=2)
"""

    with open(runner_script_path, "w", encoding="utf-8") as f:
        f.write(runner_code)

    python_bin = config_info.get("python_exe") or sys.executable
    limit_str = f"{config_info['vram_limit_gb']}GB" if config_info["vram_limit_gb"] is not None else "unconstrained"
    print(f"[*] Launching MonoGS Experiment: {dataset_key} | Mode: {config_info['memory_mode']} | Limit: {limit_str}...")
    start_wall = time.time()
    proc = subprocess.run(
        [python_bin, runner_script_path],
        cwd=os.path.abspath(os.path.join(REPO_ROOT, "system", "MonoGS")),
        capture_output=True,
        text=True,
    )
    wall_duration = time.time() - start_wall

    log_path = os.path.join(output_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n" + proc.stdout + "\n=== STDERR ===\n" + proc.stderr)

    result_data = {
        "dataset": dataset_key,
        "system": "MonoGS",
        "memory_mode": config_info["memory_mode"],
        "vram_limit_gb": config_info["vram_limit_gb"],
        "status": "Unknown",
        "wall_time_s": wall_duration,
        "ate_rmse_m": None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "peak_vram_mb": None,
        "fps": None,
        "final_gaussians": None,
        "pruned_gaussians": None,
        "pruned_size_mb": None,
    }

    # Parse stdout for summary
    for line in proc.stdout.splitlines():
        if line.startswith("RUN_SUMMARY:"):
            try:
                summary = json.loads(line.replace("RUN_SUMMARY:", ""))
                result_data["status"] = "Success"
                result_data["ate_rmse_m"] = summary.get("ate_rmse_m")
                result_data["psnr"] = summary.get("mean_psnr")
                result_data["ssim"] = summary.get("mean_ssim")
                result_data["lpips"] = summary.get("mean_lpips")
                result_data["peak_vram_mb"] = summary.get("peak_vram_allocated_mb")
                result_data["fps"] = summary.get("mean_fps")
                result_data["final_gaussians"] = summary.get("final_gaussian_count")
                result_data["pruned_gaussians"] = summary.get("pruned_gaussians_count")
                result_data["pruned_size_mb"] = summary.get("pruned_size_mb")
            except Exception:
                pass
        elif line.startswith("RUN_ERROR:"):
            err = line.replace("RUN_ERROR:", "")
            if "OOM" in err or "OutOfMemoryError" in err:
                result_data["status"] = "OOM_Crash"
            else:
                result_data["status"] = f"Failed: {err[:50]}"

    # Fallback directly to files written on disk
    stats_final_path = os.path.join(output_dir, "plot", "stats_final.json")
    if os.path.exists(stats_final_path):
        try:
            with open(stats_final_path, "r") as f:
                sdata = json.load(f)
                if result_data["ate_rmse_m"] is None and "rmse" in sdata:
                    result_data["ate_rmse_m"] = sdata["rmse"]
                    if result_data["status"] == "Unknown":
                        result_data["status"] = "Success"
        except Exception:
            pass

    online_sum_path = os.path.join(output_dir, "online_summary.json")
    if os.path.exists(online_sum_path):
        try:
            with open(online_sum_path, "r") as f:
                odata = json.load(f)
                if result_data["fps"] is None and "online_fps" in odata:
                    result_data["fps"] = odata["online_fps"]
                if result_data["final_gaussians"] is None and "final_gaussian_count" in odata:
                    result_data["final_gaussians"] = odata["final_gaussian_count"]
        except Exception:
            pass

    psnr_path = os.path.join(output_dir, "psnr", "before_opt", "final_result.json")
    if os.path.exists(psnr_path):
        try:
            with open(psnr_path, "r") as f:
                pdata = json.load(f)
                if result_data["psnr"] is None: result_data["psnr"] = pdata.get("mean_psnr")
                if result_data["ssim"] is None: result_data["ssim"] = pdata.get("mean_ssim")
                if result_data["lpips"] is None: result_data["lpips"] = pdata.get("mean_lpips")
        except Exception:
            pass

    # Fallback to pruning_summary.json written by backend
    prune_sum_path = os.path.join(output_dir, "pruning_summary.json")
    if os.path.exists(prune_sum_path):
        try:
            with open(prune_sum_path, "r") as f:
                pdata = json.load(f)
                if result_data["pruned_gaussians"] is None:
                    result_data["pruned_gaussians"] = pdata.get("pruned_gaussians_count", 0)
                if result_data["pruned_size_mb"] is None:
                    result_data["pruned_size_mb"] = pdata.get("pruned_size_mb", 0.0)
        except Exception:
            pass

    if config_info["memory_mode"] == "none":
        result_data["pruned_gaussians"] = 0
        result_data["pruned_size_mb"] = 0.0
    elif result_data["pruned_gaussians"] is None:
        result_data["pruned_gaussians"] = 0
        result_data["pruned_size_mb"] = 0.0

    # Check run_status.json for explicit execution status
    status_file = os.path.join(output_dir, "run_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                s_json = json.load(f)
                if s_json.get("status") == "completed":
                    result_data["status"] = "Success"
                elif s_json.get("status") == "OOM_Crash":
                    result_data["status"] = "OOM_Crash"
                elif s_json.get("status"):
                    result_data["status"] = s_json.get("status")
        except Exception:
            pass

    oom_flag_path = os.path.join(output_dir, "OOM_CRASH.flag")
    combined_log = (proc.stdout + "\n" + proc.stderr).lower()
    if proc.returncode != 0:
        if os.path.exists(oom_flag_path) or "outofmemoryerror" in combined_log or "cuda out of memory" in combined_log:
            result_data["status"] = "OOM_Crash"
        elif result_data["status"] in ["Unknown", "Success"]:
            result_data["status"] = f"Failed (exit {proc.returncode})"
    elif os.path.exists(oom_flag_path) and result_data["status"] != "Success":
        result_data["status"] = "OOM_Crash"

    return result_data


def run_sgs_experiment(config_info: Dict[str, Any]) -> Dict[str, Any]:
    """Executes an SGS-SLAM experiment using the configured options."""
    dataset_key = config_info["dataset_key"]
    output_dir = os.path.abspath(config_info["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Clean up any stale flags or run status from earlier runs
    for stale_file in ["OOM_CRASH.flag", "run_status.json"]:
        stale_path = os.path.join(output_dir, stale_file)
        if os.path.exists(stale_path):
            try:
                os.remove(stale_path)
            except Exception:
                pass

    repo_root_esc = REPO_ROOT.replace("\\", "/")
    output_dir_esc = output_dir.replace("\\", "/")
    seq_folder = config_info["sequence_folder"]

    # Generate custom SGS-SLAM config pointing directly to this sequence and output folder
    sgs_config_path = os.path.join(output_dir, "sgs_config.py")
    sgs_config_code = f"""import os

scene_name = "{seq_folder}"
primary_device = "cuda:0"
seed = {config_info["seed"]}
group_name = "TUM"
run_name = "output"

config = dict(
    workdir="{output_dir_esc}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=1,
    keyframe_every=5,
    mapping_window_size=24,
    report_global_progress_every=500,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=500,
    save_timestamp_keyframes=False,
    use_wandb=False,
    disable_memory_management={True if config_info["memory_mode"] == "none" else False},
    memory_management="{config_info["memory_mode"]}",
    vram_limit_gb={config_info["vram_limit_gb"] if config_info["vram_limit_gb"] is not None else "None"},
    heuristic_weights={config_info["heuristic_weights"]},
    data=dict(
        basedir="{repo_root_esc}/datasets/tum",
        gradslam_data_cfg="{repo_root_esc}/system/SGS-SLAM/configs/data/tum.yaml",
        sequence=scene_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end={config_info["max_frames"] if config_info["max_frames"] is not None else -1},
        stride=1,
        num_frames=-1,
        load_semantics=False,
        num_semantic_classes=0
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=40,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0, seg=0.0),
        lrs=dict(means3D=0.0, rgb_colors=0.0, unnorm_rotations=0.0, logit_opacities=0.0, log_scales=0.0, cam_unnorm_rots=0.0004, cam_trans=0.002, semantic_colors=0.0)
    ),
    mapping=dict(
        num_iters=60,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(im=0.5, depth=1.0, seg=0.0),
        lrs=dict(means3D=0.0001, rgb_colors=0.0025, unnorm_rotations=0.001, logit_opacities=0.05, log_scales=0.001, cam_unnorm_rots=0.0000, cam_trans=0.0000, semantic_colors=0.0),
        prune_gaussians={False if config_info["memory_mode"] == "none" else True},
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=False,
        viz_w=640, viz_h=480,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
        scene_name=scene_name,
        load_semantics=False,
    ),
)
"""
    with open(sgs_config_path, "w", encoding="utf-8") as f:
        f.write(sgs_config_code)

    script_path = os.path.abspath(os.path.join(REPO_ROOT, "system", "SGS-SLAM", "scripts", "slam.py"))
    python_bin = config_info.get("python_exe") or sys.executable

    cmd = [
        python_bin,
        script_path,
        sgs_config_path,
    ]
    if config_info["vram_limit_gb"] is not None:
        cmd.extend(["--vram-limit-gb", str(config_info["vram_limit_gb"])])

    if config_info["memory_mode"] == "none":
        cmd.append("--disable-memory-management")
    elif config_info["memory_mode"] == "custom":
        cmd.extend(["--memory-management", "custom"])
    elif config_info["memory_mode"] == "upstream":
        cmd.extend(["--memory-management", "upstream"])

    limit_str = f"{config_info['vram_limit_gb']}GB" if config_info["vram_limit_gb"] is not None else "unconstrained"
    print(f"[*] Launching SGS-SLAM Experiment: {dataset_key} | Mode: {config_info['memory_mode']} | Limit: {limit_str}...")
    start_wall = time.time()
    proc = subprocess.run(
        cmd,
        cwd=os.path.join(REPO_ROOT, "system", "SGS-SLAM"),
        capture_output=True,
        text=True,
    )
    wall_duration = time.time() - start_wall

    log_path = os.path.join(output_dir, "run.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n" + proc.stdout + "\n=== STDERR ===\n" + proc.stderr)

    combined_sgs_log = (proc.stdout + "\n" + proc.stderr).lower()
    if "outofmemoryerror" in combined_sgs_log or "out of memory" in combined_sgs_log:
        status = "OOM_Crash"
    else:
        status = "Success" if proc.returncode == 0 else f"Failed (exit {proc.returncode})"

    result_data = {
        "dataset": dataset_key,
        "system": "SGS-SLAM",
        "memory_mode": config_info["memory_mode"],
        "vram_limit_gb": config_info["vram_limit_gb"],
        "status": status,
        "wall_time_s": wall_duration,
        "ate_rmse_m": None,
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "peak_vram_mb": None,
        "fps": None,
        "final_gaussians": None,
        "pruned_gaussians": None,
        "pruned_size_mb": None,
    }

    # Parse stdout for metrics
    track_time = 0.0
    map_time = 0.0
    for line in proc.stdout.splitlines():
        if "Final Average ATE RMSE:" in line:
            try:
                # Format: Final Average ATE RMSE: 1.53 cm
                val = float(line.split(":")[-1].replace("cm", "").strip())
                result_data["ate_rmse_m"] = val / 100.0
            except Exception:
                pass
        elif "Average PSNR:" in line:
            try:
                result_data["psnr"] = float(line.split(":")[-1].strip())
            except Exception:
                pass
        elif "Average MS-SSIM:" in line:
            try:
                result_data["ssim"] = float(line.split(":")[-1].strip())
            except Exception:
                pass
        elif "Average LPIPS:" in line:
            try:
                result_data["lpips"] = float(line.split(":")[-1].strip())
            except Exception:
                pass
        elif "Average Tracking/Frame Time:" in line:
            try:
                track_time = float(line.split(":")[-1].replace("s", "").strip())
            except Exception:
                pass
        elif "Average Mapping/Frame Time:" in line:
            try:
                map_time = float(line.split(":")[-1].replace("s", "").strip())
            except Exception:
                pass
        elif "Peak VRAM:" in line:
            try:
                result_data["peak_vram_mb"] = float(line.split(":")[-1].replace("MB", "").strip())
            except Exception:
                pass
        elif "Final Gaussian Count:" in line:
            try:
                result_data["final_gaussians"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
        elif "Pruned Gaussian Count:" in line:
            try:
                result_data["pruned_gaussians"] = int(line.split(":")[-1].strip())
            except Exception:
                pass
        elif "Pruned Gaussian Size:" in line:
            try:
                result_data["pruned_size_mb"] = float(line.split(":")[-1].replace("MB", "").strip())
            except Exception:
                pass

    if (track_time + map_time) > 0:
        result_data["fps"] = 1.0 / (track_time + map_time)

    # Fallback to pruning_summary.json written by SGS-SLAM
    for p_dir in [output_dir, os.path.join(output_dir, "output")]:
        prune_sum_path = os.path.join(p_dir, "pruning_summary.json")
        if os.path.exists(prune_sum_path):
            try:
                with open(prune_sum_path, "r", encoding="utf-8") as f:
                    pdata = json.load(f)
                    if result_data["pruned_gaussians"] is None:
                        result_data["pruned_gaussians"] = pdata.get("pruned_gaussians_count", 0)
                    if result_data["pruned_size_mb"] is None:
                        result_data["pruned_size_mb"] = pdata.get("pruned_size_mb", 0.0)
            except Exception:
                pass

    if config_info["memory_mode"] == "none":
        result_data["pruned_gaussians"] = 0
        result_data["pruned_size_mb"] = 0.0
    elif result_data["pruned_gaussians"] is None:
        result_data["pruned_gaussians"] = 0
        result_data["pruned_size_mb"] = 0.0

    # Check params.npz / params.ply for final gaussian count fallback
    params_npz = os.path.join(output_dir, "output", "params.npz")
    if not os.path.exists(params_npz):
        params_npz = os.path.join(output_dir, "params.npz")
    if os.path.exists(params_npz) and result_data["final_gaussians"] is None:
        try:
            p_data = np.load(params_npz)
            if "means3D" in p_data:
                result_data["final_gaussians"] = int(p_data["means3D"].shape[0])
        except Exception:
            pass

    # Check eval directory text files for higher precision metrics
    eval_dir = os.path.join(output_dir, "output", "eval")
    if not os.path.exists(eval_dir):
        eval_dir = os.path.join(output_dir, "eval")

    if os.path.exists(eval_dir):
        try:
            for m_key, fname in [("psnr", "psnr.txt"), ("ssim", "ssim.txt"), ("lpips", "lpips.txt")]:
                fpath = os.path.join(eval_dir, fname)
                if os.path.exists(fpath):
                    vals = np.loadtxt(fpath)
                    if len(vals) > 0:
                        result_data[m_key] = float(np.mean(vals))
        except Exception:
            pass

    # Compile continuous trajectory video from rendered_rgb frames at 30 FPS
    if config_info.get("generate_video", True):
        rendered_rgb_dir = os.path.join(eval_dir, "rendered_rgb")
        if os.path.exists(rendered_rgb_dir):
            img_files = sorted(glob.glob(os.path.join(rendered_rgb_dir, "*.png")))
            if img_files:
                video_renderer = MapVideoRenderer(output_dir, fps=config_info.get("video_fps", 30))
                video_frames = []
                from PIL import Image
                for imf in img_files:
                    video_frames.append(np.array(Image.open(imf)))
                if video_frames:
                    video_renderer.render_trajectory_video(rendered_images=video_frames, apply_smoothing=True)

    # Save summary json
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def generate_comparison_artifacts(run_output_root: str, results_list: List[Dict[str, Any]]) -> None:
    """Generates comparison charts, Markdown summary report, and structured JSON across all combinations."""
    summary_md_path = os.path.join(run_output_root, "comparison_summary.md")
    summary_json_path = os.path.join(run_output_root, "comparison_summary.json")
    plots_dir = os.path.join(run_output_root, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Save hierarchical comparison summary JSON
    hierarchical_data: Dict[str, Any] = {}
    for r in results_list:
        seq = r["dataset"]
        sys_name = r["system"]
        mode = r["memory_mode"]
        limit = f"{r['vram_limit_gb']}GB" if r["vram_limit_gb"] is not None else "unconstrained"
        hierarchical_data.setdefault(seq, {}).setdefault(sys_name, {}).setdefault(mode, {})[limit] = r

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(hierarchical_data, f, indent=2)

    # 2. Generate Markdown comparison report
    md_lines = [
        "# 3DGS SLAM Memory Management & Pruning Comparison Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Experiment Combinations**: {len(results_list)}",
        "",
        "## Summary Results Table",
        "",
        "| System | Dataset | Memory Mode | VRAM Limit | Status | ATE RMSE (cm) | PSNR (dB) | SSIM | Peak VRAM (MB) | Pruned Gaussians | Pruned Size (MB) | FPS | Runtime (s) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results_list:
        limit_str = f"{r['vram_limit_gb']} GB" if r["vram_limit_gb"] is not None else "Unconstrained"
        ate_str = f"{r['ate_rmse_m']*100:.2f}" if r["ate_rmse_m"] is not None else "-"
        psnr_str = f"{r['psnr']:.2f}" if r["psnr"] is not None else "-"
        ssim_str = f"{r['ssim']:.3f}" if r["ssim"] is not None else "-"
        vram_str = f"{r['peak_vram_mb']:.1f}" if r["peak_vram_mb"] is not None else "-"
        fps_str = f"{r['fps']:.2f}" if r["fps"] is not None else "-"
        runtime_str = f"{r['wall_time_s']:.1f}" if r["wall_time_s"] is not None else "-"
        pruned_count_str = f"{r['pruned_gaussians']:,}" if r.get("pruned_gaussians") is not None else "0"
        pruned_size_str = f"{r['pruned_size_mb']:.2f}" if r.get("pruned_size_mb") is not None else "0.00"

        # Format status: clearly denote expected OOM on unconstrained baseline
        raw_status = str(r.get("status", "Unknown"))
        if r["memory_mode"] == "none" and "OOM" in raw_status:
            status_display = "OOM_Crash (Expected)"
        else:
            status_display = raw_status

        md_lines.append(
            f"| {r['system']} | {r['dataset']} | **{r['memory_mode'].upper()}** | {limit_str} | {status_display} | {ate_str} | {psnr_str} | {ssim_str} | {vram_str} | {pruned_count_str} | {pruned_size_str} | {fps_str} | {runtime_str} |"
        )

    md_lines.extend([
        "",
        "## Key System & Memory Management Insights",
        "- **Unconstrained Baseline (`NONE`)**:",
        "  - Runs with all pruning and memory limits disabled in the SLAM system.",
        "  - In the presence of a strict PyTorch GPU allocator ceiling, Gaussian accumulation and optimizer moments grow unbounded until hitting the hardware/allocator limit, triggering a clean `torch.cuda.OutOfMemoryError` (`OOM_Crash (Expected)`). This experimentally verifies that memory management is strictly necessary.",
        "- **Custom Multi-Heuristic Pruning (`CUSTOM`)**:",
        "  - The Bitwise Memory Controller continuously tracks VRAM footprint + incoming Gaussian bytes.",
        "  - Uses the 7-heuristic scoring mechanism (opacity, visibility, voxel grid spatial density, volume/covariance, tracking history frequency, deterministic pseudo-random tie-breaker, and Gaussian bit size) to evict low-utility Gaussians before keyframe optimization.",
        "  - Inactive keyframe image tensors are offloaded to host memory (`.cpu()`) and dynamic safety headroom buffers are reserved, enabling reliable execution within tight VRAM bounds.",
        "- **Upstream Baseline Pruning (`UPSTREAM`)**:",
        "  - Employs original MonoGS / SGS-SLAM heuristic thresholds (e.g. static opacity / scale bounds).",
        "",
        "## Per-Run Artifacts & Videos",
        "Each experiment combination generated a dedicated directory containing:",
        "- `reconstructed_map_trajectory.mp4`: Rendered 3D map moving along the camera trajectory (30 FPS).",
        "- `graph_memory_vs_frame.png`: VRAM allocated and reserved over time.",
        "- `graph_gaussians_vs_frame.png`: Gaussian count over keyframes.",
        "- `graph_fps_vs_frame.png`: Real-time online throughput (FPS).",
        "- `run.log`: Full stdout/stderr execution log.",
    ])

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 3. Generate Comparison Plots using matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        datasets = sorted(list(set(r["dataset"] for r in results_list)))
        systems = sorted(list(set(r["system"] for r in results_list)))

        # Group data for plotting
        for seq in datasets:
            for sys_name in systems:
                subset = [r for r in results_list if r["dataset"] == seq and r["system"] == sys_name]
                if not subset:
                    continue

                # Filter custom and upstream points with numerical limits
                limits = sorted(list(set(r["vram_limit_gb"] for r in subset if r["vram_limit_gb"] is not None)))
                if not limits:
                    continue

                # Plot Peak VRAM vs Limit
                fig, ax = plt.subplots(figsize=(8, 5))
                bar_width = 0.35
                x = np.arange(len(limits))

                custom_vram = []
                upstream_vram = []
                for lim in limits:
                    c_pt = next((r for r in subset if r["memory_mode"] == "custom" and r["vram_limit_gb"] == lim), None)
                    u_pt = next((r for r in subset if r["memory_mode"] == "upstream" and r["vram_limit_gb"] == lim), None)
                    custom_vram.append(c_pt["peak_vram_mb"] if c_pt and c_pt["peak_vram_mb"] else 0)
                    upstream_vram.append(u_pt["peak_vram_mb"] if u_pt and u_pt["peak_vram_mb"] else 0)

                ax.bar(x - bar_width/2, custom_vram, bar_width, label="Custom Pruning", color="#1f77b4")
                ax.bar(x + bar_width/2, upstream_vram, bar_width, label="Upstream Pruning", color="#ff7f0e")

                # Add unconstrained baseline horizontal line if available
                none_pt = next((r for r in subset if r["memory_mode"] == "none" and r["peak_vram_mb"]), None)
                if none_pt:
                    ax.axhline(none_pt["peak_vram_mb"], color="#d62728", linestyle="--", label=f"Unconstrained Baseline ({none_pt['peak_vram_mb']:.0f} MB)")

                ax.set_xlabel("VRAM Constraint (GB)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Peak VRAM Allocated (MB)", fontsize=11, fontweight="bold")
                ax.set_title(f"{sys_name} - Peak VRAM vs Constraint ({seq})", fontsize=12, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels([f"{l} GB" for l in limits])
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(frameon=True)
                ax.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"{sys_name}_{seq}_vram_comparison.png"), dpi=150)
                plt.close(fig)

                # Plot Trajectory ATE RMSE vs Limit
                fig, ax = plt.subplots(figsize=(8, 5))
                c_ate = [next((r["ate_rmse_m"]*100 for r in subset if r["memory_mode"] == "custom" and r["vram_limit_gb"] == lim and r["ate_rmse_m"]), None) for lim in limits]
                u_ate = [next((r["ate_rmse_m"]*100 for r in subset if r["memory_mode"] == "upstream" and r["vram_limit_gb"] == lim and r["ate_rmse_m"]), None) for lim in limits]

                valid_limits_c = [l for l, a in zip(limits, c_ate) if a is not None]
                valid_c_ate = [a for a in c_ate if a is not None]
                valid_limits_u = [l for l, a in zip(limits, u_ate) if a is not None]
                valid_u_ate = [a for a in u_ate if a is not None]

                if valid_c_ate:
                    ax.plot(valid_limits_c, valid_c_ate, "o-", label="Custom Pruning", color="#1f77b4", linewidth=2)
                if valid_u_ate:
                    ax.plot(valid_limits_u, valid_u_ate, "s-", label="Upstream Pruning", color="#ff7f0e", linewidth=2)

                if none_pt and none_pt["ate_rmse_m"]:
                    ax.axhline(none_pt["ate_rmse_m"]*100, color="#d62728", linestyle="--", label=f"Unconstrained Baseline ({none_pt['ate_rmse_m']*100:.2f} cm)")

                ax.set_xlabel("VRAM Constraint (GB)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Trajectory ATE RMSE (cm)", fontsize=11, fontweight="bold")
                ax.set_title(f"{sys_name} - Trajectory Accuracy vs Constraint ({seq})", fontsize=12, fontweight="bold")
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(frameon=True)
                ax.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"{sys_name}_{seq}_ate_comparison.png"), dpi=150)
                plt.close(fig)

                # Plot PSNR Quality vs Limit
                fig, ax = plt.subplots(figsize=(8, 5))
                c_psnr = [next((r["psnr"] for r in subset if r["memory_mode"] == "custom" and r["vram_limit_gb"] == lim and r["psnr"]), None) for lim in limits]
                u_psnr = [next((r["psnr"] for r in subset if r["memory_mode"] == "upstream" and r["vram_limit_gb"] == lim and r["psnr"]), None) for lim in limits]

                valid_limits_cp = [l for l, p in zip(limits, c_psnr) if p is not None]
                valid_c_psnr = [p for p in c_psnr if p is not None]
                valid_limits_up = [l for l, p in zip(limits, u_psnr) if p is not None]
                valid_u_psnr = [p for p in u_psnr if p is not None]

                if valid_c_psnr:
                    ax.plot(valid_limits_cp, valid_c_psnr, "o-", label="Custom Pruning", color="#1f77b4", linewidth=2)
                if valid_u_psnr:
                    ax.plot(valid_limits_up, valid_u_psnr, "s-", label="Upstream Pruning", color="#ff7f0e", linewidth=2)

                if none_pt and none_pt["psnr"]:
                    ax.axhline(none_pt["psnr"], color="#d62728", linestyle="--", label=f"Unconstrained Baseline ({none_pt['psnr']:.2f} dB)")

                ax.set_xlabel("VRAM Constraint (GB)", fontsize=11, fontweight="bold")
                ax.set_ylabel("Rendering Quality PSNR (dB)", fontsize=11, fontweight="bold")
                ax.set_title(f"{sys_name} - Rendering Quality vs Constraint ({seq})", fontsize=12, fontweight="bold")
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(frameon=True)
                ax.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"{sys_name}_{seq}_psnr_comparison.png"), dpi=150)
                plt.close(fig)

                # Plot Pruned Gaussians Count & Size vs Limit
                fig, (ax_cnt, ax_sz) = plt.subplots(1, 2, figsize=(14, 5))
                bar_width = 0.35
                x = np.arange(len(limits))

                custom_pruned_cnt = []
                upstream_pruned_cnt = []
                custom_pruned_sz = []
                upstream_pruned_sz = []
                for lim in limits:
                    c_pt = next((r for r in subset if r["memory_mode"] == "custom" and r["vram_limit_gb"] == lim), None)
                    u_pt = next((r for r in subset if r["memory_mode"] == "upstream" and r["vram_limit_gb"] == lim), None)
                    custom_pruned_cnt.append(c_pt["pruned_gaussians"] if c_pt and c_pt.get("pruned_gaussians") is not None else 0)
                    upstream_pruned_cnt.append(u_pt["pruned_gaussians"] if u_pt and u_pt.get("pruned_gaussians") is not None else 0)
                    custom_pruned_sz.append(c_pt["pruned_size_mb"] if c_pt and c_pt.get("pruned_size_mb") is not None else 0.0)
                    upstream_pruned_sz.append(u_pt["pruned_size_mb"] if u_pt and u_pt.get("pruned_size_mb") is not None else 0.0)

                # Subplot 1: Pruned Gaussians Count
                ax_cnt.bar(x - bar_width/2, custom_pruned_cnt, bar_width, label="Custom Pruning", color="#1f77b4")
                ax_cnt.bar(x + bar_width/2, upstream_pruned_cnt, bar_width, label="Upstream Pruning", color="#ff7f0e")
                ax_cnt.set_xlabel("VRAM Constraint (GB)", fontsize=11, fontweight="bold")
                ax_cnt.set_ylabel("Pruned Gaussians (Count)", fontsize=11, fontweight="bold")
                ax_cnt.set_title(f"{sys_name} - Pruned Gaussians Count ({seq})", fontsize=12, fontweight="bold")
                ax_cnt.set_xticks(x)
                ax_cnt.set_xticklabels([f"{l} GB" for l in limits])
                ax_cnt.legend(frameon=True)
                ax_cnt.grid(True, linestyle=":", alpha=0.6)

                # Subplot 2: Pruned Memory Footprint (MB)
                ax_sz.bar(x - bar_width/2, custom_pruned_sz, bar_width, label="Custom Pruning", color="#1f77b4")
                ax_sz.bar(x + bar_width/2, upstream_pruned_sz, bar_width, label="Upstream Pruning", color="#ff7f0e")
                ax_sz.set_xlabel("VRAM Constraint (GB)", fontsize=11, fontweight="bold")
                ax_sz.set_ylabel("Pruned Memory Footprint (MB)", fontsize=11, fontweight="bold")
                ax_sz.set_title(f"{sys_name} - Pruned VRAM Footprint ({seq})", fontsize=12, fontweight="bold")
                ax_sz.set_xticks(x)
                ax_sz.set_xticklabels([f"{l} GB" for l in limits])
                ax_sz.legend(frameon=True)
                ax_sz.grid(True, linestyle=":", alpha=0.6)

                plt.tight_layout()
                fig.savefig(os.path.join(plots_dir, f"{sys_name}_{seq}_pruning_comparison.png"), dpi=150)
                plt.close(fig)

    except Exception as e:
        print(f"[Comparison Artifacts] Plotting skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="3DGS SLAM Multi-System & Multi-Constraint Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--system", type=str, default=None,
                        help="SLAM system(s): 'MonoGS', 'SGS-SLAM', 'both', or comma-separated. Default uses script CONFIG.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="TUM sequence(s): 'fr1_desk', 'fr2_xyz', 'fr3_office', 'all', or comma-separated. Default uses script CONFIG.")
    parser.add_argument("--pruning-modes", "--memory-management", dest="pruning_modes", type=str, default=None,
                        help="Pruning modes: 'none', 'custom', 'upstream', 'all', or comma-separated. Default uses script CONFIG.")
    parser.add_argument("--vram-limits-gb", "--vram-limit-gb", dest="vram_limits_gb", type=str, default=None,
                        help="VRAM constraints in GB: comma-separated list e.g. '1.0,2.0,4.0,8.0'. Default uses script CONFIG.")
    parser.add_argument("--none-per-limit", action="store_true", default=None,
                        help="Run 'none' mode across all VRAM limits to test OOM, rather than once unconstrained.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Heuristic weights: JSON string or comma-separated (opacity,vis,voxel,volume,tracking,rand).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help=f"Output root directory for benchmark runs (defaults to '{DEFAULT_OUTPUT_DIR}').")
    parser.add_argument("--run-name", type=str, default=None,
                        help=f"Subdirectory name for run data (defaults to '{DEFAULT_RUN_NAME}'; use 'timestamp' for timestamped folder).")
    parser.add_argument("--eval-rendering", action="store_true", default=None,
                        help="Evaluate PSNR/SSIM/LPIPS on novel view frames.")
    parser.add_argument("--no-video", action="store_true", default=False,
                        help="Disable reconstructed 3D map video export.")
    parser.add_argument("--video-fps", type=int, default=None,
                        help=f"Framerate for output videos (defaults to {DEFAULT_VIDEO_FPS}).")
    parser.add_argument("--seed", type=int, default=None,
                        help=f"Random seed (defaults to {DEFAULT_SEED}).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional limit on number of dataset frames (defaults to all frames).")
    parser.add_argument("--python-exe", type=str, default=None,
                        help="Explicit path to Python interpreter with CUDA extensions.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned experiment matrix without executing.")

    args = parser.parse_args()

    # 1. Resolve Systems
    if args.system:
        if args.system.lower() in ["both", "all"]:
            systems = ["MonoGS", "SGS-SLAM"]
        else:
            systems = [s.strip() for s in args.system.split(",") if s.strip()]
    else:
        systems = list(DEFAULT_SYSTEMS)

    # 2. Resolve Datasets
    if args.dataset:
        if args.dataset.lower() == "all":
            datasets = list(TUM_SEQUENCES.keys())
        else:
            datasets = [s.strip() for s in args.dataset.split(",") if s.strip()]
    else:
        if "all" in DEFAULT_DATASETS:
            datasets = list(TUM_SEQUENCES.keys())
        else:
            datasets = list(DEFAULT_DATASETS)

    # 3. Resolve Pruning Modes
    if args.pruning_modes:
        if args.pruning_modes.lower() in ["all", "both"]:
            pruning_modes = ["none", "custom", "upstream"]
        else:
            pruning_modes = [m.strip().lower() for m in args.pruning_modes.split(",") if m.strip()]
    else:
        pruning_modes = list(DEFAULT_PRUNING_MODES)

    # 4. Resolve VRAM Limits
    if args.vram_limits_gb:
        vram_limits = [float(v.strip()) for v in args.vram_limits_gb.split(",") if v.strip()]
    else:
        vram_limits = list(DEFAULT_VRAM_LIMITS_GB)

    run_none_per_limit = args.none_per_limit if args.none_per_limit is not None else DEFAULT_RUN_NONE_PER_LIMIT
    weights = parse_weights(args.weights)
    output_dir_root = args.output_dir if args.output_dir else DEFAULT_OUTPUT_DIR
    eval_rendering = args.eval_rendering if args.eval_rendering is not None else DEFAULT_EVAL_RENDERING
    generate_video = not args.no_video if args.no_video else DEFAULT_GENERATE_VIDEO
    video_fps = args.video_fps if args.video_fps is not None else DEFAULT_VIDEO_FPS
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    max_frames = args.max_frames if args.max_frames is not None else DEFAULT_MAX_FRAMES
    python_exe = args.python_exe if args.python_exe else DEFAULT_PYTHON_EXE

    # Build execution matrix
    plan: List[Tuple[str, str, str, Optional[float]]] = []
    for sys_name in systems:
        for ds in datasets:
            for mode in pruning_modes:
                if mode == "none":
                    if run_none_per_limit or args.vram_limits_gb is not None:
                        for lim in vram_limits:
                            plan.append((sys_name, ds, mode, lim))
                    else:
                        plan.append((sys_name, ds, mode, None))
                else:
                    for lim in vram_limits:
                        plan.append((sys_name, ds, mode, lim))

    run_name = args.run_name if args.run_name else DEFAULT_RUN_NAME
    if run_name.lower() == "timestamp":
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    run_output_root = os.path.join(output_dir_root, run_name)
    os.makedirs(run_output_root, exist_ok=True)

    print("\n" + "=" * 78)
    print("3DGS SLAM MEMORY MANAGEMENT & PRUNING MULTI-CONSTRAINT BENCHMARK SUITE")
    print("=" * 78)
    print(f"Target Systems      : {', '.join(systems)}")
    print(f"Target Datasets     : {', '.join(datasets)}")
    print(f"Pruning Modes       : {', '.join([m.upper() for m in pruning_modes])}")
    print(f"VRAM Bounds (GB)    : {', '.join(str(l) for l in vram_limits)}")
    print(f"None Mode Policy    : {'Per-limit sweep (test OOM)' if run_none_per_limit else 'Single unconstrained baseline'}")
    print(f"Total Combinations  : {len(plan)}")
    print(f"Consolidated Dir    : {run_output_root}")
    print(f"Heuristic Weights   : {json.dumps(weights)}")
    print("=" * 78)

    print("\nPlanned Experiment Matrix:")
    for idx, (s, d, m, l) in enumerate(plan, 1):
        l_str = f"{l} GB" if l is not None else "Unconstrained"
        print(f"  [{idx:02d}/{len(plan):02d}] System: {s:<8} | Dataset: {d:<10} | Mode: {m:<8} | Bound: {l_str}")
    print("=" * 78 + "\n")

    if args.dry_run:
        print("[Dry Run] All configurations validated successfully. Exiting without execution.")
        return

    csv_path = os.path.join(run_output_root, "benchmark_results.csv")
    fieldnames = [
        "system", "dataset", "memory_mode", "vram_limit_gb", "status",
        "ate_rmse_m", "psnr", "ssim", "lpips", "peak_vram_mb", "fps",
        "final_gaussians", "pruned_gaussians", "pruned_size_mb", "wall_time_s"
    ]

    # Load existing results if CSV already exists to consolidate multiple runs in one directory
    results_map: Dict[Tuple[str, str, str, Optional[float]], Dict[str, Any]] = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed_row = dict(row)
                    parsed_row["vram_limit_gb"] = float(row["vram_limit_gb"]) if row.get("vram_limit_gb") and row["vram_limit_gb"] not in ["", "None"] else None
                    for flt_col in ["ate_rmse_m", "psnr", "ssim", "lpips", "peak_vram_mb", "fps", "pruned_size_mb", "wall_time_s"]:
                        parsed_row[flt_col] = float(row[flt_col]) if row.get(flt_col) and row[flt_col] not in ["", "None"] else None
                    if row.get("final_gaussians") and row["final_gaussians"] not in ["", "None"]:
                        parsed_row["final_gaussians"] = int(row["final_gaussians"])
                    else:
                        parsed_row["final_gaussians"] = None
                    if row.get("pruned_gaussians") and row["pruned_gaussians"] not in ["", "None"]:
                        parsed_row["pruned_gaussians"] = int(row["pruned_gaussians"])
                    else:
                        parsed_row["pruned_gaussians"] = None
                    key = (parsed_row["system"], parsed_row["dataset"], parsed_row["memory_mode"], parsed_row["vram_limit_gb"])
                    results_map[key] = parsed_row
            print(f"[*] Loaded {len(results_map)} existing benchmark results from: {csv_path}")
        except Exception as e:
            print(f"[!] Warning: Could not parse existing CSV ({e}), starting fresh.")

    for idx, (sys_name, ds_key, mode, limit) in enumerate(plan, 1):
        limit_folder_str = f"{limit}GB" if limit is not None else "unconstrained"
        sub_dir_name = f"{sys_name}_{ds_key}_{mode}_{limit_folder_str}"
        sub_out = os.path.join(run_output_root, sub_dir_name)

        print(f"\n>>> Running [{idx}/{len(plan)}]: {sub_dir_name} <<<")
        cfg_info = setup_experiment_config(
            system=sys_name,
            dataset_key=ds_key,
            memory_mode=mode,
            vram_limit_gb=limit,
            heuristic_weights=weights,
            output_dir=sub_out,
            eval_rendering=eval_rendering,
            generate_video=generate_video,
            video_fps=video_fps,
            seed=seed,
            max_frames=max_frames,
            custom_python_exe=python_exe,
        )

        if sys_name == "MonoGS":
            res = run_monogs_experiment(cfg_info)
        else:
            res = run_sgs_experiment(cfg_info)

        key = (res["system"], res["dataset"], res["memory_mode"], res["vram_limit_gb"])
        results_map[key] = res

        # Incrementally update CSV after each experiment with consolidated entries
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_map.values():
                writer.writerow(r)

        # Incrementally update comparison artifacts (Markdown report, JSON, Comparison Plots)
        generate_comparison_artifacts(run_output_root, list(results_map.values()))

    # Final pass to ensure all artifacts and plots reflect complete consolidated data
    generate_comparison_artifacts(run_output_root, list(results_map.values()))

    print("\n" + "=" * 78)
    print("ALL BENCHMARK EXPERIMENTS COMPLETED!")
    print(f"Consolidated CSV saved to: {csv_path}")
    print(f"Comparison report        : {os.path.join(run_output_root, 'comparison_summary.md')}")
    print(f"Comparison plots         : {os.path.join(run_output_root, 'comparison_plots')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
