"""Unit tests for the Multi-Heuristic Pruning System."""

import os
import sys
import unittest
import torch
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from system.common import (
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


class TestMultiHeuristicPruner(unittest.TestCase):
    def test_opacity_score(self):
        # High opacity vs low opacity
        logits = torch.tensor([[-5.0], [0.0], [5.0]], dtype=torch.float32)
        scores = compute_opacity_score(logits, is_logits=True)
        self.assertEqual(scores.shape[0], 3)
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        # Higher opacity must yield higher score
        self.assertLess(scores[0].item(), scores[1].item())
        self.assertLess(scores[1].item(), scores[2].item())

    def test_visibility_score(self):
        counts = torch.tensor([1, 10, 50, 100], dtype=torch.float32)
        scores = compute_visibility_score(counts)
        self.assertEqual(scores.shape[0], 4)
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        self.assertLess(scores[0].item(), scores[1].item())
        self.assertLess(scores[2].item(), scores[3].item())
        self.assertAlmostEqual(scores[3].item(), 1.0, places=5)

    def test_voxel_grid_score(self):
        # 1 isolated point in voxel A, 5 dense points in voxel B
        xyz = torch.tensor([
            [0.0, 0.0, 0.0],       # Voxel A (isolated)
            [1.0, 1.0, 1.0],       # Voxel B
            [1.01, 1.01, 1.01],   # Voxel B
            [1.02, 1.02, 1.02],   # Voxel B
            [1.03, 1.03, 1.03],   # Voxel B
            [1.04, 1.04, 1.04],   # Voxel B
        ], dtype=torch.float32)

        scores = compute_voxel_grid_score(xyz, voxel_size=0.1)
        self.assertEqual(scores.shape[0], 6)
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        # Isolated point in voxel A should score higher than dense points in voxel B
        self.assertAlmostEqual(scores[0].item(), 1.0, places=4)
        self.assertLess(scores[1].item(), scores[0].item())

    def test_volume_score(self):
        # Normal vs giant floater vs tiny degenerate
        scales = torch.tensor([
            [0.01, 0.01, 0.01],    # Normal well-scaled
            [2.0, 2.0, 2.0],       # Giant floater
            [1e-9, 1e-9, 1e-9],    # Tiny degenerate
        ], dtype=torch.float32)

        scores = compute_volume_score(scales, scene_radius=5.0, is_log_scale=False)
        self.assertEqual(scores.shape[0], 3)
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        # Normal well-scaled Gaussian scores highest
        self.assertGreater(scores[0].item(), scores[1].item())
        self.assertGreater(scores[0].item(), scores[2].item())

    def test_tracking_history_score(self):
        hits = [0, 5, 10, 20]
        scores = compute_tracking_history_score(hits, window_n_frames=20)
        self.assertEqual(scores.shape[0], 4)
        self.assertAlmostEqual(scores[0].item(), 0.0, places=4)
        self.assertAlmostEqual(scores[1].item(), 0.25, places=4)
        self.assertAlmostEqual(scores[2].item(), 0.50, places=4)
        self.assertAlmostEqual(scores[3].item(), 1.0, places=4)

    def test_det_random_score_reproducibility(self):
        ids = torch.arange(100, dtype=torch.int64)
        s1 = compute_det_random_score(ids, seed=123)
        s2 = compute_det_random_score(ids, seed=123)
        s3 = compute_det_random_score(ids, seed=999)

        self.assertTrue(torch.all(s1 >= 0.0) and torch.all(s1 <= 1.0))
        # Exact reproduction with same seed
        self.assertTrue(torch.all(s1 == s2))
        # Different seed produces different pseudo-random values
        self.assertFalse(torch.all(s1 == s3))

    def test_gaussian_size_score(self):
        # 3 Gaussians with explicit bit sizes: 100 bits (small), 500 bits (medium), 1500 bits (large)
        bits = torch.tensor([100.0, 500.0, 1500.0], dtype=torch.float32)
        scores = compute_gaussian_size_score(None, explicit_bit_sizes=bits)
        self.assertEqual(scores.shape[0], 3)
        self.assertTrue(torch.all(scores >= 0.0) and torch.all(scores <= 1.0))
        # Smaller gaussians (by # bits) must receive HIGHER scores
        self.assertGreater(scores[0].item(), scores[1].item())
        self.assertGreater(scores[1].item(), scores[2].item())
        self.assertAlmostEqual(scores[0].item(), 1.0, places=4)
        self.assertAlmostEqual(scores[2].item(), 0.0, places=4)

        # Also verify alias
        alias_scores = compute_bit_size_score(None, explicit_bit_sizes=bits)
        self.assertTrue(torch.all(scores == alias_scores))

    def test_multi_heuristic_pruner_composite_and_selection(self):
        pruner = MultiHeuristicPruner(
            weights={
                "opacity": 0.14,
                "visibility": 0.14,
                "voxel_grid": 0.14,
                "volume_based": 0.14,
                "tracking_history": 0.14,
                "det_random": 0.14,
                "gaussian_size": 0.16,
            },
            seed=42,
        )

        mock_data = {
            "means3D": torch.randn(100, 3),
            "logit_opacities": torch.randn(100, 1),
            "log_scales": torch.randn(100, 3) * 0.1 - 2.0,
            "visibility": torch.randint(1, 20, (100,)),
            "ids": torch.arange(100),
        }

        scores = pruner.compute_all_scores(mock_data, current_frame=5)
        self.assertIn("gaussian_size", scores)
        self.assertIn("bit_size", scores)
        for k, v in scores.items():
            self.assertEqual(v.shape[0], 100)
            self.assertTrue(torch.all(v >= 0.0) and torch.all(v <= 1.0))

        composite = pruner.compute_composite_score(scores)
        self.assertEqual(composite.shape[0], 100)
        self.assertTrue(torch.all(composite >= 0.0) and torch.all(composite <= 1.0))

        # Select 30 lowest scoring points to prune
        prune_mask = pruner.select_gaussians_to_prune(composite, num_to_prune=30, min_survivors=1)
        self.assertEqual(prune_mask.sum().item(), 30)

        # Confirm the selected points actually have the lowest composite scores
        pruned_scores = composite[prune_mask]
        kept_scores = composite[~prune_mask]
        self.assertLessEqual(pruned_scores.max().item(), kept_scores.min().item() + 1e-6)

    def test_heuristic_weight_nn(self):
        nn_adapter = HeuristicWeightNN(state_dim=8, num_heuristics=6)
        state = torch.randn(1, 8)
        weights = nn_adapter(state).squeeze()
        self.assertEqual(weights.shape[0], 6)
        # Sum of softmax weights must equal 1.0
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        self.assertTrue(torch.all(weights >= 0.0) and torch.all(weights <= 1.0))


if __name__ == "__main__":
    unittest.main()

