"""Unit tests for the Bitwise Memory Controller."""

import os
import sys
import unittest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from system.common import BitwiseMemoryController


class DummyMonoGSModel:
    """Mock GaussianModel with parameters for testing."""
    def __init__(self, num_points=1000):
        self._xyz = torch.nn.Parameter(torch.randn(num_points, 3, dtype=torch.float32))
        self._features_dc = torch.nn.Parameter(torch.randn(num_points, 1, 3, dtype=torch.float32))
        self._features_rest = torch.nn.Parameter(torch.zeros(num_points, 0, 3, dtype=torch.float32))
        self._scaling = torch.nn.Parameter(torch.randn(num_points, 3, dtype=torch.float32))
        self._rotation = torch.nn.Parameter(torch.randn(num_points, 4, dtype=torch.float32))
        self._opacity = torch.nn.Parameter(torch.randn(num_points, 1, dtype=torch.float32))

    @property
    def get_xyz(self):
        return self._xyz


class TestBitwiseMemoryController(unittest.TestCase):
    def test_estimate_bytes_per_gaussian(self):
        controller = BitwiseMemoryController(vram_limit_bytes=1000000)
        model = DummyMonoGSModel(num_points=500)
        bpg = controller.estimate_bytes_per_gaussian(model)
        self.assertGreater(bpg, 100)
        self.assertIsInstance(bpg, int)

    def test_estimate_bytes_sgs_params(self):
        controller = BitwiseMemoryController(vram_limit_bytes=1000000)
        params = {
            "means3D": torch.randn(200, 3),
            "rgb_colors": torch.randn(200, 3),
            "unnorm_rotations": torch.randn(200, 4),
            "logit_opacities": torch.randn(200, 1),
            "log_scales": torch.randn(200, 3),
        }
        bpg = controller.estimate_bytes_per_gaussian(params)
        self.assertGreater(bpg, 100)

    def test_check_keyframe_under_budget(self):
        # 100 GB limit -> never should prune
        controller = BitwiseMemoryController(vram_limit_gb=100.0)
        model = DummyMonoGSModel(num_points=100)
        status = controller.check_keyframe(model, incoming_gaussians_count=50, current_gaussian_count=100)
        self.assertFalse(status.should_prune)
        self.assertEqual(status.required_prune_count, 0)
        self.assertEqual(status.incoming_gaussians_count, 50)

    def test_check_keyframe_over_budget(self):
        # Very tiny limit (e.g. 100 bytes) -> must trigger pruning
        controller = BitwiseMemoryController(vram_limit_bytes=100, headroom_fraction=0.0)
        model = DummyMonoGSModel(num_points=500)
        status = controller.check_keyframe(model, incoming_gaussians_count=100, current_gaussian_count=500)
        self.assertTrue(status.should_prune)
        self.assertGreater(status.required_prune_count, 0)
        self.assertLessEqual(status.required_prune_count, 499)

    def test_no_limit_configured(self):
        controller = BitwiseMemoryController(vram_limit_bytes=None)
        model = DummyMonoGSModel(num_points=100)
        status = controller.check_keyframe(model, incoming_gaussians_count=200)
        self.assertFalse(status.should_prune)
        self.assertEqual(status.required_prune_count, 0)


if __name__ == "__main__":
    unittest.main()

