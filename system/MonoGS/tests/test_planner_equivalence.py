"""Freeze the pre-optimization V1 planner's observable decisions."""
import numpy as np
import pytest

from project_utils.budget_controller import BudgetInfeasible, RowBudgetController
from project_utils.gaussian_metadata import GaussianMetadata
import torch


def old_plan(maximum, fraction, count, growth, priority, protected, costs):
    low = int(np.floor(maximum * fraction))
    if protected.sum() > maximum:
        raise BudgetInfeasible("Protected support cannot fit row ceiling")
    pressure = count + growth > maximum
    floor = max(int(protected.sum()), int(count > 0))
    effective_low = max(low, floor)
    capacity = effective_low - floor if pressure else maximum - count
    admitted = 0
    for cost in costs:
        if admitted + cost <= capacity:
            admitted += cost
        else:
            break
    target = min(count, effective_low - admitted) if pressure else count
    if target < floor:
        raise BudgetInfeasible("Protected support leaves no admission headroom")
    keep = protected.copy()
    for index in priority:
        if keep.sum() >= target:
            break
        keep[index] = True
    return keep, growth, admitted, growth - admitted, target, int(protected.sum())


@pytest.mark.parametrize("maximum", [1, 2, 4, 40, 400])
def test_exact_old_planner_decisions(maximum):
    rng = np.random.default_rng(20261001 + maximum)
    for _ in range(100):
        count = int(rng.integers(0, maximum + 1))
        priority = rng.permutation(count).astype(np.int64)
        protected = rng.random(count) < rng.uniform(0, 1)
        costs = rng.choice([1, 2], size=int(rng.integers(0, 20))).tolist()
        growth = sum(costs)
        fraction = float(rng.choice([.5, .95, 1.]))
        old = old_plan(maximum, fraction, count, growth, priority, protected, costs)
        controller = RowBudgetController(maximum, fraction)
        new = controller.plan(count, growth, priority, protected, atomic_costs=costs)
        assert np.array_equal(new.keep, old[0])
        assert (new.attempted_growth, new.admitted_growth, new.rejected_growth,
                new.target_existing, new.protected_count) == old[1:]
        assert (controller.attempted_growth, controller.admitted_growth,
                controller.rejected_growth) == (growth, old[2], old[3])


def test_floor_bound_and_atomic_split():
    priority = np.array([3, 2, 1, 0], dtype=np.int64)
    protected = np.array([True, True, False, False])
    old = old_plan(4, .5, 4, 4, priority, protected, [2, 2])
    new = RowBudgetController(4, .5).plan(4, 4, priority, protected,
                                          atomic_costs=[2, 2])
    assert np.array_equal(new.keep, old[0])
    assert new.admitted_growth == old[2]


def test_vector_feedback_matches_scalar_updates():
    rng = np.random.default_rng(16)
    for n in (0, 1, 39, 400):
        new = GaussianMetadata()
        new.append(n, 0)
        if n:
            keep = torch.as_tensor(rng.random(n) > .2)
            new.filter(keep)
        old = GaussianMetadata()
        old.__dict__.update({name: value.clone() if isinstance(value, torch.Tensor) else value
                             for name, value in new.__dict__.items()})
        ids = np.arange(n + 10, dtype=np.int64)
        hits = rng.random(n + 10).astype(np.float32)
        seen = rng.integers(-1, 30, size=n + 10, dtype=np.int32)
        positions = {int(value): i for i, value in enumerate(old.gaussian_id.tolist())}
        beta = 2 ** (-1 / 20)
        for j, value in enumerate(ids.tolist()):
            i = positions.get(value)
            if i is not None:
                old.tracking_ema[i] = beta ** 4 * old.tracking_ema[i] + torch.tensor(hits[j])
                old.last_seen_frame[i] = max(old.last_seen_frame[i], torch.tensor(seen[j]))
                if seen[j] >= 0:
                    old.observed_since_creation[i] = True
        new.apply_feedback(ids=ids, version=new.map_version, sequence=0, frames=4,
                           weighted_hits=hits, last_seen=seen, beta=beta)
        assert torch.equal(new.tracking_ema, old.tracking_ema)
        assert torch.equal(new.last_seen_frame, old.last_seen_frame)
        assert torch.equal(new.observed_since_creation, old.observed_since_creation)
