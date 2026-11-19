"""Bitwise Memory Controller for 3D Gaussian Splatting SLAM.

Measures VRAM usage (or RAM fallback) + incoming Gaussians in bytes on each keyframe,
compares against user-defined VRAM max limit, and signals when and how much to prune.
"""

import math
import os
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class MemoryControllerStatus:
    should_prune: bool
    current_vram_bytes: int
    incoming_gaussians_count: int
    incoming_bytes: int
    predicted_vram_bytes: int
    vram_max_limit_bytes: int
    excess_bytes: int
    required_prune_count: int
    bytes_per_gaussian: int
    utilization_fraction: float


class BitwiseMemoryController:
    """Bitwise / byte-accurate memory controller that monitors VRAM and bounds map footprint."""

    def __init__(
        self,
        vram_limit_bytes: Optional[int] = None,
        vram_limit_gb: Optional[float] = None,
        headroom_fraction: float = 0.05,
        default_bytes_per_gaussian: int = 240,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            vram_limit_bytes: Max limit in bytes.
            vram_limit_gb: Max limit in GB (takes precedence if both provided).
            headroom_fraction: Fraction of limit reserved for working transient allocations.
            default_bytes_per_gaussian: Initial fallback estimate of bytes per Gaussian.
            device: Target torch device.
        """
        if vram_limit_gb is not None:
            self.vram_limit_bytes = int(vram_limit_gb * (1024 ** 3))
        elif vram_limit_bytes is not None:
            self.vram_limit_bytes = int(vram_limit_bytes)
        else:
            self.vram_limit_bytes = None

        self.headroom_fraction = float(headroom_fraction)
        self.default_bytes_per_gaussian = int(default_bytes_per_gaussian)
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
        # Telemetry history
        self.peak_vram_bytes = 0
        self.peak_ram_bytes = 0
        self.history = []

    def get_current_vram_bytes(self) -> int:
        """Returns current allocated VRAM in bytes (or process RSS RAM if CPU/CUDA unavailable)."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            # Use allocated memory as primary measure, tracking maximum
            current = allocated
            self.peak_vram_bytes = max(self.peak_vram_bytes, reserved, allocated)
            return current
        else:
            # Fallback to system RAM (RSS)
            rss = self.get_current_ram_bytes()
            self.peak_ram_bytes = max(self.peak_ram_bytes, rss)
            return rss

    def get_current_ram_bytes(self) -> int:
        """Returns current process RSS in bytes."""
        if psutil is not None:
            return psutil.Process(os.getpid()).memory_info().rss
        return 0

    def estimate_bytes_per_gaussian(self, model_or_params: Any) -> int:
        """Dynamically computes the exact byte size of a Gaussian including optimizer moments."""
        # Direct support for containers reporting total bytes
        if hasattr(model_or_params, "get_total_bytes") and hasattr(model_or_params, "__len__"):
            n = len(model_or_params)
            if n > 0:
                return max(model_or_params.get_total_bytes() // n, self.default_bytes_per_gaussian)

        total_param_bytes = 0
        num_gaussians = 0

        # Check MonoGS GaussianModel
        if hasattr(model_or_params, "get_xyz"):
            xyz = model_or_params.get_xyz
            num_gaussians = xyz.shape[0] if xyz is not None else 0
            if num_gaussians > 0:
                # Sum parameter tensor sizes
                for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
                    t = getattr(model_or_params, attr, None)
                    if t is not None and isinstance(t, torch.Tensor):
                        total_param_bytes += t.nelement() * t.element_size()
                
                # In Adam optimization, each trainable tensor stores exp_avg and exp_avg_sq (2x)
                # plus gradients (1x), totaling 4x parameter storage in active optimizer
                bytes_per_point = total_param_bytes // num_gaussians
                active_bytes_per_gaussian = bytes_per_point * 4
                return max(active_bytes_per_gaussian, self.default_bytes_per_gaussian)

        # Check SGS-SLAM params dictionary
        elif isinstance(model_or_params, dict) and "means3D" in model_or_params:
            means = model_or_params["means3D"]
            num_gaussians = means.shape[0] if means is not None else 0
            if num_gaussians > 0:
                for k, v in model_or_params.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == num_gaussians:
                        total_param_bytes += v.nelement() * v.element_size()
                bytes_per_point = total_param_bytes // num_gaussians
                active_bytes_per_gaussian = bytes_per_point * 4
                return max(active_bytes_per_gaussian, self.default_bytes_per_gaussian)

        return self.default_bytes_per_gaussian

    def check_keyframe(
        self,
        current_model_or_params: Any,
        incoming_gaussians_count: int = 0,
        current_gaussian_count: Optional[int] = None,
    ) -> MemoryControllerStatus:
        """
        On a new keyframe:
        1. Measures current VRAM usage (in bytes)
        2. Estimates incoming Gaussians (in bytes)
        3. Compares (current VRAM + incoming bytes) to VRAM max limit
        4. Signals whether to perform pruning and how many points need removal.
        """
        if current_gaussian_count is None:
            if hasattr(current_model_or_params, "__len__"):
                current_gaussian_count = len(current_model_or_params)
            elif hasattr(current_model_or_params, "get_xyz") and current_model_or_params.get_xyz is not None:
                current_gaussian_count = current_model_or_params.get_xyz.shape[0]
            elif isinstance(current_model_or_params, dict) and "means3D" in current_model_or_params:
                current_gaussian_count = current_model_or_params["means3D"].shape[0]
            else:
                current_gaussian_count = 0

        current_vram = self.get_current_vram_bytes()
        bytes_per_gaussian = self.estimate_bytes_per_gaussian(current_model_or_params)
        incoming_bytes = incoming_gaussians_count * bytes_per_gaussian
        predicted_vram = current_vram + incoming_bytes

        # If no limit is configured, never prune
        if self.vram_limit_bytes is None or self.vram_limit_bytes <= 0:
            status = MemoryControllerStatus(
                should_prune=False,
                current_vram_bytes=current_vram,
                incoming_gaussians_count=incoming_gaussians_count,
                incoming_bytes=incoming_bytes,
                predicted_vram_bytes=predicted_vram,
                vram_max_limit_bytes=0,
                excess_bytes=0,
                required_prune_count=0,
                bytes_per_gaussian=bytes_per_gaussian,
                utilization_fraction=0.0,
            )
            self.history.append(status)
            return status

        # Dynamic safety headroom: must account for both static safety margin
        # AND incoming keyframe batch insertion plus PyTorch transient buffers
        dynamic_incoming_headroom = int(incoming_bytes * 1.35)
        static_headroom = int(self.vram_limit_bytes * self.headroom_fraction)
        effective_headroom_bytes = max(static_headroom, dynamic_incoming_headroom)
        effective_limit = max(0, self.vram_limit_bytes - effective_headroom_bytes)
        excess_bytes = max(0, predicted_vram - effective_limit)
        should_prune = excess_bytes > 0

        required_prune_count = 0
        if should_prune and bytes_per_gaussian > 0:
            required_prune_count = int(math.ceil(excess_bytes / bytes_per_gaussian))
            # Bound prune count by current count to prevent empty map
            if current_gaussian_count is not None and current_gaussian_count > 1:
                required_prune_count = min(required_prune_count, current_gaussian_count - 1)

        utilization = float(current_vram) / float(self.vram_limit_bytes)

        status = MemoryControllerStatus(
            should_prune=should_prune,
            current_vram_bytes=current_vram,
            incoming_gaussians_count=incoming_gaussians_count,
            incoming_bytes=incoming_bytes,
            predicted_vram_bytes=predicted_vram,
            vram_max_limit_bytes=self.vram_limit_bytes,
            excess_bytes=excess_bytes,
            required_prune_count=required_prune_count,
            bytes_per_gaussian=bytes_per_gaussian,
            utilization_fraction=utilization,
        )
        self.history.append(status)
        return status

