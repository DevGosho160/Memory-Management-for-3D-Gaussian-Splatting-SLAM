"""Common memory management, pruning, and evaluation utilities for 3DGS SLAM."""

from .bitwise_controller import BitwiseMemoryController, MemoryControllerStatus
from .multi_heuristic_pruner import (
    MultiHeuristicPruner,
    HeuristicWeightNN,
    compute_opacity_score,
    compute_visibility_score,
    compute_voxel_grid_score,
    compute_volume_score,
    compute_tracking_history_score,
    compute_det_random_score,
    compute_gaussian_size_score,
    compute_bit_size_score,
)
from .metrics_tracker import MetricsTracker
from .video_renderer import MapVideoRenderer

__all__ = [
    "BitwiseMemoryController",
    "MemoryControllerStatus",
    "MultiHeuristicPruner",
    "HeuristicWeightNN",
    "compute_opacity_score",
    "compute_visibility_score",
    "compute_voxel_grid_score",
    "compute_volume_score",
    "compute_tracking_history_score",
    "compute_det_random_score",
    "compute_gaussian_size_score",
    "compute_bit_size_score",
    "MetricsTracker",
    "MapVideoRenderer",
]
