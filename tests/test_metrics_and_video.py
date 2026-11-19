"""Unit tests for MetricsTracker and MapVideoRenderer."""

import os
import shutil
import sys
import tempfile
import unittest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from system.common import MetricsTracker, MapVideoRenderer


class TestMetricsAndVideo(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_metrics_tracker_and_plots(self):
        tracker = MetricsTracker(self.test_dir)

        # Record dummy frames
        for i in range(20):
            # Generate simulated estimated and GT poses
            gt = np.eye(4)
            gt[:3, 3] = [0.1 * i, 0.05 * np.sin(i), 0.0]
            est = gt.copy()
            est[:3, 3] += np.random.normal(0, 0.005, 3)

            tracker.record_frame(
                frame_idx=i,
                gaussian_count=1000 + i * 50,
                vram_allocated_bytes=int((200 + i * 10) * 1024**2),
                vram_reserved_bytes=int((300 + i * 10) * 1024**2),
                ram_rss_bytes=int((500 + i * 5) * 1024**2),
                frame_duration_s=0.05,
                estimated_pose=est,
                gt_pose=gt,
            )

        # Record quality
        for i in range(0, 20, 5):
            tracker.record_quality_eval(i, psnr=32.5 + i * 0.1, ssim=0.92 + i * 0.002, lpips=0.08)

        # Compute ATE
        ate = tracker.compute_trajectory_ate()
        self.assertIsNotNone(ate)
        self.assertLess(ate, 0.05)  # Expected small error

        # Summary
        summary = tracker.generate_summary()
        self.assertEqual(summary["num_frames"], 20)
        self.assertIn("ate_rmse_m", summary)
        self.assertIn("peak_vram_allocated_mb", summary)
        self.assertIn("mean_psnr", summary)

        # Plots
        plots = tracker.generate_all_plots()
        self.assertGreaterEqual(len(plots), 3)
        for p in plots:
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 0)

    def test_map_video_renderer(self):
        renderer = MapVideoRenderer(self.test_dir, fps=10)
        # Create 10 dummy frames
        dummy_frames = []
        for i in range(10):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :, 0] = int(i * 25)
            frame[:, :, 1] = int(255 - i * 25)
            dummy_frames.append(frame)

        video_path = renderer.render_trajectory_video(rendered_images=dummy_frames)
        self.assertIsNotNone(video_path)
        self.assertTrue(os.path.exists(video_path))
        self.assertGreater(os.path.getsize(video_path), 0)


if __name__ == "__main__":
    unittest.main()

