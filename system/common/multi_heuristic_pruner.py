"""Multi-Heuristic Map Pruning System for 3D Gaussian Splatting SLAM.

Implements normalized scoring functions [0, 1] for each pruning heuristic:
  1. Opacity Pruning: higher opacity = higher score
  2. Visibility Pruning: higher observation / view count = higher score
  3. Voxel Grid Pruning: points in sparser / representative voxels = higher score
  4. Volume-based Pruning: well-proportioned covariance / volume = higher score (penalizing floaters & degenerate needles)
  5. Tracking History Pruning: frequency seen in last n frames = higher score
  6. Deterministic Random Pruning: uniform pseudo-random score using stable deterministic hash

Calculates weighted composite scores and removes the lowest-scoring Gaussians until within budget.
Includes a neural network weight adapter (HeuristicWeightNN) for learned weight prediction.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual Normalized Heuristic Scoring Functions [0, 1]
# ---------------------------------------------------------------------------

def compute_opacity_score(
    opacities: torch.Tensor,
    is_logits: bool = True
) -> torch.Tensor:
    """
    Opacity heuristic score: higher opacity = higher score.
    Returns Tensor[N] in [0.0, 1.0].
    """
    if opacities.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=opacities.device)
    
    op = torch.sigmoid(opacities).squeeze() if is_logits else opacities.squeeze()
    score = torch.clamp(op.float(), 0.0, 1.0)
    return score


def compute_visibility_score(
    visibility_counts: Union[torch.Tensor, np.ndarray, List[int]],
    max_observations: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Visibility heuristic score: higher observation / covisibility count = higher score.
    Returns Tensor[N] in [0.0, 1.0].
    """
    if isinstance(visibility_counts, np.ndarray):
        counts = torch.from_numpy(visibility_counts)
    elif isinstance(visibility_counts, (list, tuple)):
        counts = torch.tensor(visibility_counts)
    else:
        counts = visibility_counts

    if device is not None:
        counts = counts.to(device)
    counts = counts.squeeze().float()

    if counts.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=counts.device)

    if max_observations is None or max_observations <= 0:
        max_obs = float(counts.max().item()) if counts.numel() > 0 else 1.0
        max_obs = max(max_obs, 1.0)
    else:
        max_obs = float(max_observations)

    score = torch.clamp(counts / max_obs, 0.0, 1.0)
    return score


def compute_voxel_grid_score(
    xyz: torch.Tensor,
    voxel_size: float = 0.05,
) -> torch.Tensor:
    """
    Voxel grid spatial distribution score:
    Gaussians in sparse or representative voxel locations get higher score (1.0).
    Gaussians clustered densely in redundant voxels receive lower scores (~ 1 / sqrt(N_voxel)).
    Returns Tensor[N] in [0.0, 1.0].
    """
    if xyz.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=xyz.device)

    # Discretize coordinates into voxel indices using memory-efficient in-place offsets
    min_xyz = xyz.min(dim=0).values
    voxel_coords = torch.floor((xyz - min_xyz) / voxel_size).to(torch.int32)
    max_coords = voxel_coords.max(dim=0).values.to(torch.int64)
    stride_y = max_coords[0] + 1
    stride_z = stride_y * (max_coords[1] + 1)

    linear_indices = (
        voxel_coords[:, 0].to(torch.int64)
        + voxel_coords[:, 1].to(torch.int64) * stride_y
        + voxel_coords[:, 2].to(torch.int64) * stride_z
    )
    del voxel_coords

    # Count points per voxel
    _, inverse_indices, counts = torch.unique(
        linear_indices, return_inverse=True, return_counts=True
    )
    del linear_indices

    # Inverted density: isolated points get 1.0, dense voxel points get 1 / sqrt(count)
    point_counts = counts[inverse_indices].float()
    del counts, inverse_indices
    score = torch.rsqrt(point_counts)
    return torch.clamp(score, 0.0, 1.0)


def compute_volume_score(
    scales: torch.Tensor,
    scene_radius: float = 5.0,
    is_log_scale: bool = True,
) -> torch.Tensor:
    """
    Volume / covariance heuristic score:
    Well-scaled, well-proportioned Gaussians get higher score (~1.0).
    Extremely large "floaters" (> 0.1 * scene_radius) or degenerate near-zero splats get lower score.
    Returns Tensor[N] in [0.0, 1.0].
    """
    if scales.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=scales.device)

    actual_scales = torch.exp(scales) if is_log_scale else scales
    max_scale = actual_scales.max(dim=1).values
    # Volume is proportional to product of the 3 axes
    volume = actual_scales[:, 0] * actual_scales[:, 1] * actual_scales[:, 2]

    # Floater penalty: scale exceeding threshold
    floater_threshold = 0.1 * scene_radius
    floater_penalty = torch.clamp((max_scale - floater_threshold) / (floater_threshold + 1e-6), 0.0, 1.0)

    # Sub-sampled median volume avoids full GPU RadixSort scratch allocations
    sub_sample = volume[::20] if volume.numel() > 100 else volume
    median_vol = float(sub_sample.median().item()) if sub_sample.numel() > 0 else 1e-4
    median_vol = max(median_vol, 1e-8)
    tiny_penalty = torch.clamp(1.0 - (volume / (median_vol * 0.01 + 1e-12)), 0.0, 1.0)

    # Healthy volume score is high when both floater and tiny penalties are low
    score = 1.0 - torch.clamp(floater_penalty * 0.7 + tiny_penalty * 0.3, 0.0, 1.0)
    return torch.clamp(score, 0.0, 1.0)


def compute_tracking_history_score(
    hits_in_last_n: Union[torch.Tensor, np.ndarray, List[int]],
    window_n_frames: int = 20,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Tracking history score: how many frames was the Gaussian seen in the last n frames.
    Higher count in recent window = higher score.
    Returns Tensor[N] in [0.0, 1.0].
    """
    if isinstance(hits_in_last_n, np.ndarray):
        hits = torch.from_numpy(hits_in_last_n)
    elif isinstance(hits_in_last_n, (list, tuple)):
        hits = torch.tensor(hits_in_last_n)
    else:
        hits = hits_in_last_n

    if device is not None:
        hits = hits.to(device)
    hits = hits.squeeze().float()

    if hits.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=hits.device)

    n_window = max(float(window_n_frames), 1.0)
    score = torch.clamp(hits / n_window, 0.0, 1.0)
    return score


def compute_det_random_score(
    ids_or_count: Union[torch.Tensor, np.ndarray, int],
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Deterministic pseudo-random score using SplitMix64 hash.
    Uniformly distributed in [0.0, 1.0) and completely reproducible.
    Returns Tensor[N] in [0.0, 1.0].
    """
    if isinstance(ids_or_count, int):
        ids = torch.arange(ids_or_count, dtype=torch.int64, device=device)
    elif isinstance(ids_or_count, np.ndarray):
        ids = torch.from_numpy(ids_or_count).to(device=device, dtype=torch.int64)
    else:
        ids = ids_or_count.to(device=device, dtype=torch.int64)

    if ids.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=ids.device)

    # 64-bit SplitMix-style hash (using signed int64 constants for PyTorch cross-platform compatibility)
    s = int(seed) & 0xFFFFFFFF
    z = (ids ^ s) * -4658895280553007687
    z = (z ^ (z >> 30)) * -7723592293110705685
    z = z ^ (z >> 31)
    
    # Map lowest 32 bits to [0, 1)
    u32 = (z & 0xFFFFFFFF).float()
    score = u32 / 4294967296.0
    return torch.clamp(score, 0.0, 1.0)


def compute_gaussian_size_score(
    model_or_params: Any,
    explicit_bit_sizes: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Gaussian size (in bits) heuristic score:
    Measures the memory footprint / representation bit size of each Gaussian.
    Returns Tensor[N] normalized in [0.0, 1.0]:
      - Smaller (by # bits) Gaussians get higher scores (~ 1.0).
      - Larger (by # bits) Gaussians get lower scores (~ 0.0).

    Bit size accounts for:
      - Fixed geometry & opacity parameters: 11 floats x 32 bits = 352 bits.
      - Color / Spherical Harmonics coefficients:
        * Degree 0 (DC only, 3 floats) = 96 bits.
        * Degree 1 (4x3 = 12 floats) = 384 bits.
        * Degree 2 (9x3 = 27 floats) = 864 bits.
        * Degree 3 (16x3 = 48 floats) = 1536 bits.
      - Active coefficient bits / parameter variance: Gaussians with higher-order
        frequencies or larger entropy require more bits to store and optimize.
    """
    if explicit_bit_sizes is not None:
        if isinstance(explicit_bit_sizes, np.ndarray):
            bits = torch.from_numpy(explicit_bit_sizes).to(device=device, dtype=torch.float32)
        elif isinstance(explicit_bit_sizes, (list, tuple)):
            bits = torch.tensor(explicit_bit_sizes, device=device, dtype=torch.float32)
        else:
            bits = explicit_bit_sizes.to(device=device, dtype=torch.float32)
        if bits.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)
        b_min = bits.min()
        b_max = bits.max()
        if (b_max - b_min) > 1e-4:
            score = 1.0 - (bits - b_min) / (b_max - b_min)
        else:
            score = 1.0 - torch.clamp(bits / 2000.0, 0.0, 1.0)
        return torch.clamp(score, 0.0, 1.0)

    # 1. MonoGS GaussianModel
    if hasattr(model_or_params, "get_xyz"):
        xyz = model_or_params.get_xyz
        N = xyz.shape[0] if xyz is not None else 0
        dev = xyz.device if xyz is not None else (device or torch.device("cpu"))
        if N == 0:
            return torch.zeros(0, dtype=torch.float32, device=dev)

        # Base geometric parameters: xyz(3), rot(4), scale(3), opacity(1) = 11 floats @ 32 bits = 352 bits
        base_bits = 352.0
        
        # Color & SH features: _features_dc is (N, 1, 3), _features_rest is (N, K, 3)
        sh_dc = getattr(model_or_params, "_features_dc", None)
        dc_bits = (sh_dc.shape[1] * sh_dc.shape[2] * 32.0) if (sh_dc is not None and sh_dc.ndim >= 3) else 96.0

        sh_rest = getattr(model_or_params, "_features_rest", None)
        if sh_rest is not None and isinstance(sh_rest, torch.Tensor) and sh_rest.numel() > 0:
            # Active SH rest bits: each non-negligible coefficient takes 3 * 32 = 96 bits
            rest_mag = sh_rest.abs().sum(dim=[1, 2])
            max_rest_mag = rest_mag.max().item() if rest_mag.numel() > 0 else 1.0
            max_rest_mag = max(max_rest_mag, 1e-6)
            sh_rest_bits = (rest_mag / max_rest_mag) * (sh_rest.shape[1] * sh_rest.shape[2] * 32.0)
        else:
            sh_rest_bits = torch.zeros(N, device=dev, dtype=torch.float32)

        # Scale variance / geometric complexity: higher eccentricity requires higher bit precision
        scales = model_or_params.get_scaling
        if scales is not None and scales.numel() > 0:
            scale_ratio = (scales.max(dim=-1).values / (scales.min(dim=-1).values + 1e-6)).clamp(1.0, 10.0)
            precision_bits = (scale_ratio - 1.0) * 16.0
        else:
            precision_bits = torch.zeros(N, device=dev, dtype=torch.float32)

        total_bits = base_bits + dc_bits + sh_rest_bits + precision_bits

    # 2. SGS-SLAM params dictionary
    elif isinstance(model_or_params, dict) and "means3D" in model_or_params:
        xyz = model_or_params["means3D"]
        N = xyz.shape[0] if xyz is not None else 0
        dev = xyz.device if xyz is not None else (device or torch.device("cpu"))
        if N == 0:
            return torch.zeros(0, dtype=torch.float32, device=dev)

        # Base: means3D(3x32=96), unnorm_rotations(4x32=128), log_scales(3x32=96), logit_opacities(1x32=32) = 352 bits
        base_bits = 352.0
        # Color: rgb_colors(3x32=96)
        color_bits = 96.0

        # Semantics or extra features if present
        extra_bits = 0.0
        if "semantic_colors" in model_or_params and model_or_params["semantic_colors"] is not None:
            sc = model_or_params["semantic_colors"]
            if isinstance(sc, torch.Tensor) and sc.ndim >= 2:
                extra_bits = sc.shape[1] * 32.0

        log_scales = model_or_params.get("log_scales", None)
        if log_scales is not None and log_scales.numel() > 0:
            scales = torch.exp(log_scales)
            scale_ratio = (scales.max(dim=-1).values / (scales.min(dim=-1).values + 1e-6)).clamp(1.0, 10.0)
            precision_bits = (scale_ratio - 1.0) * 16.0
        else:
            precision_bits = torch.zeros(N, device=dev, dtype=torch.float32)

        total_bits = (base_bits + color_bits + extra_bits) + precision_bits

    else:
        return torch.ones(0, dtype=torch.float32, device=device or torch.device("cpu"))

    # Invert: smaller bits -> higher score, larger bits -> lower score
    b_min = total_bits.min()
    b_max = total_bits.max()
    if (b_max - b_min) > 1e-4:
        score = 1.0 - (total_bits - b_min) / (b_max - b_min)
    else:
        score = 1.0 - torch.clamp(total_bits / 2000.0, 0.0, 1.0)

    return torch.clamp(score, 0.0, 1.0)

# Alias
compute_bit_size_score = compute_gaussian_size_score


# ---------------------------------------------------------------------------
# Neural Network Adapter for Learned Weight Prediction
# ---------------------------------------------------------------------------

class HeuristicWeightNN(nn.Module):
    """
    Small neural network that predicts dynamic heuristic weights based on scene and map state.
    Allows optimizing pruning weights end-to-end or via reinforcement learning.
    """

    def __init__(self, state_dim: int = 8, num_heuristics: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heuristics),
        )
        # Default initialization biased towards uniform weights
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Outputs normalized weights [w_opacity, w_vis, w_voxel, w_vol, w_track, w_rand, w_size] summing to 1.0."""
        logits = self.net(state_vector)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Multi-Heuristic Pruning Coordinator
# ---------------------------------------------------------------------------

class MultiHeuristicPruner:
    """Coordinates normalized multi-heuristic scoring, weighted combination, and map pruning."""

    HEURISTIC_NAMES = [
        "opacity",
        "visibility",
        "voxel_grid",
        "volume_based",
        "tracking_history",
        "det_random",
        "gaussian_size",
    ]

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        voxel_size: float = 0.05,
        tracking_window_frames: int = 20,
        seed: int = 42,
        weight_nn: Optional[HeuristicWeightNN] = None,
    ):
        """
        Args:
            weights: Dictionary of weights (defaults to equal weight for each heuristic).
            voxel_size: Voxel size in meters for voxel grid pruning.
            tracking_window_frames: Number of recent frames for tracking history.
            seed: Seed for deterministic random pruning.
            weight_nn: Optional Neural Network model for dynamic weight prediction.
        """
        # Equal default weights summing to ~1.0
        default_weights = {name: round(1.0 / len(self.HEURISTIC_NAMES), 4) for name in self.HEURISTIC_NAMES}
        if weights:
            # Map alias 'bit_size' -> 'gaussian_size'
            normalized_in = dict(weights)
            if "bit_size" in normalized_in and "gaussian_size" not in normalized_in:
                normalized_in["gaussian_size"] = normalized_in.pop("bit_size")
            default_weights.update(normalized_in)

        # Normalize weights so sum equals 1.0
        total = sum(default_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in default_weights.items()}
        else:
            self.weights = {k: 1.0 / len(self.HEURISTIC_NAMES) for k in self.HEURISTIC_NAMES}

        self.voxel_size = float(voxel_size)
        self.tracking_window_frames = int(tracking_window_frames)
        self.seed = int(seed)
        self.weight_nn = weight_nn

        # Tracking history buffer: maps gaussian ID -> list/deque of recent frame hits
        self.tracking_history_buffer: Dict[int, List[int]] = {}

    def update_weights(self, new_weights: Dict[str, float]):
        """Update heuristic weights dynamically."""
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in new_weights.items()}

    def record_tracking_hit(self, gaussian_ids: Union[torch.Tensor, np.ndarray, List[int]], current_frame: int):
        """Records tracking visibility hits for the current frame."""
        if isinstance(gaussian_ids, torch.Tensor):
            ids = gaussian_ids.detach().cpu().numpy().tolist()
        elif isinstance(gaussian_ids, np.ndarray):
            ids = gaussian_ids.tolist()
        else:
            ids = list(gaussian_ids)

        for gid in ids:
            buf = self.tracking_history_buffer.setdefault(gid, [])
            buf.append(current_frame)
            # Retain only frames within window
            cutoff = current_frame - self.tracking_window_frames
            while buf and buf[0] < cutoff:
                buf.pop(0)

    def get_tracking_hits_vector(
        self,
        current_ids: Union[torch.Tensor, np.ndarray, List[int]],
        current_frame: int,
        device: torch.device
    ) -> torch.Tensor:
        """Returns tensor of hit counts in the last n frames for each active Gaussian."""
        if isinstance(current_ids, torch.Tensor):
            ids = current_ids.detach().cpu().numpy()
        elif isinstance(current_ids, list):
            ids = np.array(current_ids)
        else:
            ids = current_ids

        cutoff = current_frame - self.tracking_window_frames
        hits = np.zeros(len(ids), dtype=np.float32)
        for i, gid in enumerate(ids):
            buf = self.tracking_history_buffer.get(int(gid), [])
            # Count entries >= cutoff
            count = sum(1 for f in buf if f >= cutoff)
            hits[i] = float(count)

        return torch.from_numpy(hits).to(device)

    def compute_all_scores(
        self,
        model_or_data: Any,
        current_frame: int = 0,
        scene_radius: float = 5.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes normalized score tensors [0, 1] for all 6 heuristics.
        Supports both MonoGS GaussianModel and SGS-SLAM dictionary format.
        """
        scores = {}

        # 1. MonoGS GaussianModel
        if hasattr(model_or_data, "get_xyz"):
            xyz = model_or_data.get_xyz
            N = xyz.shape[0]
            device = xyz.device
            if N == 0:
                for name in self.HEURISTIC_NAMES:
                    scores[name] = torch.zeros(0, dtype=torch.float32, device=device)
                return scores

            opacities = model_or_data.get_opacity
            scales = model_or_data.get_scaling
            n_obs = getattr(model_or_data, "n_obs", torch.zeros(N, device=device))

            # Opacity
            scores["opacity"] = compute_opacity_score(opacities, is_logits=False)

            # Visibility
            scores["visibility"] = compute_visibility_score(n_obs, device=device)

            # Voxel Grid
            scores["voxel_grid"] = compute_voxel_grid_score(xyz, voxel_size=self.voxel_size)

            # Volume-based
            scores["volume_based"] = compute_volume_score(scales, scene_radius=scene_radius, is_log_scale=False)

            # Tracking History
            ids = getattr(model_or_data, "unique_kfIDs", torch.arange(N, device=device))
            hits = self.get_tracking_hits_vector(ids, current_frame, device)
            scores["tracking_history"] = compute_tracking_history_score(hits, self.tracking_window_frames, device)

            # Deterministic Random
            scores["det_random"] = compute_det_random_score(ids, seed=self.seed, device=device)

            # Gaussian Size (Bit Size)
            scores["gaussian_size"] = compute_gaussian_size_score(model_or_data, device=device)
            scores["bit_size"] = scores["gaussian_size"]

        # 2. SGS-SLAM dictionary format
        elif isinstance(model_or_data, dict) and "means3D" in model_or_data:
            xyz = model_or_data["means3D"]
            N = xyz.shape[0]
            device = xyz.device
            if N == 0:
                for name in self.HEURISTIC_NAMES:
                    scores[name] = torch.zeros(0, dtype=torch.float32, device=device)
                scores["bit_size"] = torch.zeros(0, dtype=torch.float32, device=device)
                return scores

            logit_opacities = model_or_data.get("logit_opacities", torch.zeros((N, 1), device=device))
            log_scales = model_or_data.get("log_scales", torch.zeros((N, 3), device=device))
            vis = model_or_data.get("visibility", torch.ones(N, device=device))

            scores["opacity"] = compute_opacity_score(logit_opacities, is_logits=True)
            scores["visibility"] = compute_visibility_score(vis, device=device)
            scores["voxel_grid"] = compute_voxel_grid_score(xyz, voxel_size=self.voxel_size)
            scores["volume_based"] = compute_volume_score(log_scales, scene_radius=scene_radius, is_log_scale=True)

            ids = model_or_data.get("ids", torch.arange(N, device=device))
            hits = self.get_tracking_hits_vector(ids, current_frame, device)
            scores["tracking_history"] = compute_tracking_history_score(hits, self.tracking_window_frames, device)
            scores["det_random"] = compute_det_random_score(ids, seed=self.seed, device=device)

            # Gaussian Size (Bit Size)
            scores["gaussian_size"] = compute_gaussian_size_score(model_or_data, device=device)
            scores["bit_size"] = scores["gaussian_size"]

        else:
            raise ValueError(f"Unsupported model format: {type(model_or_data)}")

        return scores

    def compute_composite_score(
        self,
        scores_dict: Dict[str, torch.Tensor],
        state_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes weighted sum of heuristic scores.
        If weight_nn is present and state_vector provided, uses NN-predicted weights.
        """
        first_tensor = next(iter(scores_dict.values()))
        N = first_tensor.shape[0]
        device = first_tensor.device

        if N == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)

        # Dynamic weights from NN if available
        if self.weight_nn is not None and state_vector is not None:
            nn_weights = self.weight_nn(state_vector.to(device)).squeeze()
            weights_map = {name: float(nn_weights[i].item()) for i, name in enumerate(self.HEURISTIC_NAMES)}
        else:
            weights_map = self.weights

        composite = torch.zeros(N, dtype=torch.float32, device=device)
        processed = set()
        for name, score_t in scores_dict.items():
            canonical = "gaussian_size" if name in ["bit_size", "gaussian_size"] else name
            if canonical in processed:
                continue
            w = weights_map.get(canonical, weights_map.get(name, 0.0))
            composite += w * score_t.squeeze()
            processed.add(canonical)

        return composite

    def select_gaussians_to_prune(
        self,
        composite_score: torch.Tensor,
        num_to_prune: int,
        min_survivors: int = 1,
    ) -> torch.Tensor:
        """
        Sorts Gaussians by composite score ascending and selects lowest scoring points.
        Returns boolean prune_mask where True indicates point to be removed.
        """
        N = composite_score.shape[0]
        prune_mask = torch.zeros(N, dtype=torch.bool, device=composite_score.device)

        if num_to_prune <= 0 or N <= min_survivors:
            return prune_mask

        actual_prune = min(num_to_prune, N - min_survivors)
        # Select lowest scoring points using topk (avoids sorting entire array)
        _, prune_indices = torch.topk(composite_score, actual_prune, largest=False)
        prune_mask[prune_indices] = True
        return prune_mask
