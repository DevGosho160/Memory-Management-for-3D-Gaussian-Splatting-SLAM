"""Comprehensive Metrics Tracker and Graph Generator for 3DGS SLAM.

Tracks:
  - Peak VRAM (or RAM) usage
  - Gaussian count over time
  - Frame-by-frame and overall throughput (FPS)
  - ATE (Absolute Trajectory Error) against ground truth
  - PSNR, SSIM, LPIPS rendering metrics
  - Detailed graphs over the course of the dataset
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MetricsTracker:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.start_time = time.time()
        self.frame_indices = []
        self.timestamps = []
        self.vram_allocated_mb = []
        self.vram_reserved_mb = []
        self.ram_rss_mb = []
        self.gaussian_counts = []
        self.fps_per_frame = []
        self.frame_durations_s = []

        # Evaluation metrics
        self.ate_rmse = None
        self.eval_frame_indices = []
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []

        # Pruning telemetry
        self.pruned_gaussians_count = 0
        self.pruned_gaussians_bytes = 0

        # Poses
        self.estimated_poses = []
        self.gt_poses = []

    def record_pruning(self, count: int, byte_size: int = 0):
        """Records a pruning event (number of points and memory bytes removed)."""
        self.pruned_gaussians_count += int(count)
        self.pruned_gaussians_bytes += int(byte_size)

    def record_frame(
        self,
        frame_idx: int,
        gaussian_count: int,
        vram_allocated_bytes: int = 0,
        vram_reserved_bytes: int = 0,
        ram_rss_bytes: int = 0,
        frame_duration_s: float = 0.0,
        estimated_pose: Optional[np.ndarray] = None,
        gt_pose: Optional[np.ndarray] = None,
    ):
        """Records telemetry for an online SLAM frame."""
        now = time.time()
        self.frame_indices.append(frame_idx)
        self.timestamps.append(now - self.start_time)
        self.vram_allocated_mb.append(vram_allocated_bytes / (1024 ** 2))
        self.vram_reserved_mb.append(vram_reserved_bytes / (1024 ** 2))
        self.ram_rss_mb.append(ram_rss_bytes / (1024 ** 2))
        self.gaussian_counts.append(gaussian_count)
        self.frame_durations_s.append(frame_duration_s)
        fps = (1.0 / frame_duration_s) if frame_duration_s > 0 else 0.0
        self.fps_per_frame.append(fps)

        if estimated_pose is not None:
            self.estimated_poses.append(np.asarray(estimated_pose))
        if gt_pose is not None:
            self.gt_poses.append(np.asarray(gt_pose))

    def record_quality_eval(
        self,
        frame_idx: int,
        psnr: float,
        ssim: float,
        lpips: Optional[float] = None
    ):
        """Records image reconstruction quality metrics."""
        self.eval_frame_indices.append(frame_idx)
        self.psnr_values.append(float(psnr))
        self.ssim_values.append(float(ssim))
        if lpips is not None:
            self.lpips_values.append(float(lpips))

    def compute_trajectory_ate(
        self,
        estimated_poses: Optional[List[np.ndarray]] = None,
        gt_poses: Optional[List[np.ndarray]] = None
    ) -> float:
        """Computes Absolute Trajectory Error (ATE RMSE) with Umeyama SE(3)/Sim(3) alignment."""
        est_list = estimated_poses if estimated_poses is not None else self.estimated_poses
        gt_list = gt_poses if gt_poses is not None else self.gt_poses

        if not est_list or not gt_list or len(est_list) != len(gt_list) or len(est_list) < 3:
            self.ate_rmse = None
            return 0.0

        est_xyz = np.array([p[:3, 3] for p in est_list])
        gt_xyz = np.array([p[:3, 3] for p in gt_list])

        # Centering
        mu_est = est_xyz.mean(axis=0)
        mu_gt = gt_xyz.mean(axis=0)
        est_centered = est_xyz - mu_est
        gt_centered = gt_xyz - mu_gt

        # Covariance matrix H
        H = est_centered.T @ gt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflection correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        var_est = np.sum(est_centered ** 2)
        scale = np.trace(np.diag(S)) / (var_est + 1e-12)
        # Use rigid SE(3) scale=1.0 for RGB-D
        t = mu_gt - R @ mu_est

        aligned_est = (R @ est_xyz.T).T + t
        errors = np.linalg.norm(aligned_est - gt_xyz, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        self.ate_rmse = rmse
        return rmse

    def generate_summary(self) -> Dict[str, Any]:
        """Calculates final aggregated metrics for the run."""
        total_time = time.time() - self.start_time
        num_frames = len(self.frame_indices)
        mean_fps = (num_frames / total_time) if total_time > 0 else 0.0

        peak_vram_mb = max(self.vram_allocated_mb) if self.vram_allocated_mb else 0.0
        peak_vram_reserved_mb = max(self.vram_reserved_mb) if self.vram_reserved_mb else 0.0
        peak_ram_mb = max(self.ram_rss_mb) if self.ram_rss_mb else 0.0

        mean_psnr = float(np.mean(self.psnr_values)) if self.psnr_values else None
        mean_ssim = float(np.mean(self.ssim_values)) if self.ssim_values else None
        mean_lpips = float(np.mean(self.lpips_values)) if self.lpips_values else None

        if self.ate_rmse is None and self.estimated_poses and self.gt_poses:
            self.compute_trajectory_ate()

        summary = {
            "num_frames": num_frames,
            "total_time_s": total_time,
            "mean_fps": mean_fps,
            "peak_vram_allocated_mb": peak_vram_mb,
            "peak_vram_reserved_mb": peak_vram_reserved_mb,
            "peak_ram_mb": peak_ram_mb,
            "ate_rmse_m": self.ate_rmse,
            "mean_psnr": mean_psnr,
            "mean_ssim": mean_ssim,
            "mean_lpips": mean_lpips,
            "final_gaussian_count": self.gaussian_counts[-1] if self.gaussian_counts else 0,
            "max_gaussian_count": max(self.gaussian_counts) if self.gaussian_counts else 0,
            "pruned_gaussians_count": self.pruned_gaussians_count,
            "pruned_gaussians_bytes": self.pruned_gaussians_bytes,
            "pruned_size_mb": round(self.pruned_gaussians_bytes / (1024 ** 2), 4),
        }

        # Save JSON
        json_path = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def generate_all_plots(self) -> List[str]:
        """Generates visual plots over the course of the dataset."""
        generated_files = []
        if not self.frame_indices:
            return generated_files

        # 1. VRAM & RAM Memory Plot over Frames
        plt.figure(figsize=(10, 5))
        if any(v > 0 for v in self.vram_allocated_mb):
            plt.plot(self.frame_indices, self.vram_allocated_mb, label="VRAM Allocated (MB)", color="tab:blue", lw=2)
            plt.plot(self.frame_indices, self.vram_reserved_mb, label="VRAM Reserved (MB)", color="tab:cyan", lw=1.5, linestyle="--")
        if any(r > 0 for r in self.ram_rss_mb):
            plt.plot(self.frame_indices, self.ram_rss_mb, label="Process RAM RSS (MB)", color="tab:green", lw=2)
        plt.title("Memory Footprint over Dataset")
        plt.xlabel("Frame Index")
        plt.ylabel("Memory (MB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p1 = os.path.join(self.output_dir, "graph_memory_vs_frame.png")
        plt.savefig(p1, bbox_inches="tight", dpi=150)
        plt.close()
        generated_files.append(p1)

        # 2. Gaussian Count over Frames
        plt.figure(figsize=(10, 5))
        plt.plot(self.frame_indices, self.gaussian_counts, label="Gaussian Count", color="tab:orange", lw=2)
        plt.title("Gaussian Count over Dataset")
        plt.xlabel("Frame Index")
        plt.ylabel("Active Gaussians")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p2 = os.path.join(self.output_dir, "graph_gaussians_vs_frame.png")
        plt.savefig(p2, bbox_inches="tight", dpi=150)
        plt.close()
        generated_files.append(p2)

        # 3. Throughput (FPS) over Frames
        plt.figure(figsize=(10, 5))
        plt.plot(self.frame_indices, self.fps_per_frame, label="Frame FPS", color="tab:purple", lw=1.5)
        mean_fps = np.mean(self.fps_per_frame) if self.fps_per_frame else 0
        plt.axhline(mean_fps, color="black", linestyle="--", label=f"Mean FPS ({mean_fps:.2f})")
        plt.title("Throughput (FPS) over Dataset")
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p3 = os.path.join(self.output_dir, "graph_fps_vs_frame.png")
        plt.savefig(p3, bbox_inches="tight", dpi=150)
        plt.close()
        generated_files.append(p3)

        # 4. Rendering Quality (PSNR / SSIM / LPIPS)
        if self.eval_frame_indices and self.psnr_values:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("Frame Index")
            ax1.set_ylabel("PSNR (dB)", color="tab:red")
            l1 = ax1.plot(self.eval_frame_indices, self.psnr_values, color="tab:red", label="PSNR (dB)", lw=2)
            ax1.tick_params(axis="y", labelcolor="tab:red")

            ax2 = ax1.twinx()
            ax2.set_ylabel("SSIM", color="tab:blue")
            l2 = ax2.plot(self.eval_frame_indices, self.ssim_values, color="tab:blue", label="SSIM", lw=2, linestyle="-.")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

            lines = l1 + l2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="lower right")
            plt.title("Reconstruction Quality (PSNR & SSIM) over Dataset")
            plt.grid(True, alpha=0.3)
            p4 = os.path.join(self.output_dir, "graph_quality_vs_frame.png")
            plt.savefig(p4, bbox_inches="tight", dpi=150)
            plt.close()
            generated_files.append(p4)

        # 5. 3D Camera Trajectory Comparison Plot
        if self.estimated_poses and self.gt_poses and len(self.estimated_poses) == len(self.gt_poses):
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")
            est_xyz = np.array([p[:3, 3] for p in self.estimated_poses])
            gt_xyz = np.array([p[:3, 3] for p in self.gt_poses])

            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="Ground Truth", color="tab:green", lw=2)
            ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label="Estimated", color="tab:blue", lw=2, linestyle="--")
            ax.set_title(f"Trajectory (ATE RMSE: {self.ate_rmse:.4f} m)" if self.ate_rmse is not None else "Trajectory")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()
            p5 = os.path.join(self.output_dir, "graph_trajectory_3d.png")
            plt.savefig(p5, bbox_inches="tight", dpi=150)
            plt.close()
            generated_files.append(p5)

        return generated_files

