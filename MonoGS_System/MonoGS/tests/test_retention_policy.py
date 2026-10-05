import numpy as np
import pytest

from project_utils.budget_controller import BudgetInfeasible, RowBudgetController
from project_utils.retention_policy import protected_rows, rank_rows


def test_deterministic_rankings_and_no_history_ablation():
    ids = np.array([8, 3, 5], dtype=np.int64)
    common = dict(ids=ids, opacity=[.5, .5, .5], last_seen=[4, 4, -1],
                  lineage_birth=[0, 0, 0], tracking_ema=[.1, .8, 0.],
                  window_fraction=[0., 0., 0.], frame=5,
                  probation_active=[False] * 3, seed=0)
    assert rank_rows("opacity", **common).tolist() == [1, 2, 0]
    assert rank_rows("lru", **common).tolist() == [1, 0, 2]
    assert rank_rows("tracking_support", **common)[0] == 1
    a = rank_rows("random", **common)
    np.random.seed(999)
    np.random.random(100)
    assert np.array_equal(a, rank_rows("random", **common))
    assert not np.array_equal(rank_rows("tracking_support", **common),
                              rank_rows("tracking_support", **{
                                  **common, "tracking_ema": [1., 0., 0.]}))
    nonfinite = {**common, "opacity": [float("nan"), .2, .3]}
    assert rank_rows("opacity", **nonfinite)[-1] == 0


def test_support_union_and_strict_gross_growth():
    ids = np.arange(6)
    p = protected_rows(ids, [.9, .8, .7, .6, .5, .4],
                       [0, 0, 1, 1, 2, 2],
                       [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.],
                        [1., 0., 0.], [2., 0., 0.], [2., 0., 0.]],
                       [[1, 0, 1, 0, 0, 0]], [0.] * 6,
                       view_min=1, origin_min=1, cell_min=1)
    assert p.tolist() == [True, False, True, False, True, False]
    c = RowBudgetController(10)
    plan = c.plan(9, 2, np.arange(9), np.zeros(9, bool), atomic_costs=[2])
    assert plan.keep.sum() + plan.admitted_growth <= 10
    assert c.violations == 0
    # The split's net growth is one, but its two children are staged at once.
    assert plan.admitted_growth == 2
    assert plan.keep.sum() <= 8


def test_protection_infeasibility_and_no_future_deletion_credit():
    c = RowBudgetController(4)
    plan = c.plan(4, 2, np.arange(4), np.ones(4, bool), atomic_costs=[2])
    assert plan.admitted_growth == 0
    assert plan.rejected_growth == 2
    with pytest.raises(RuntimeError, match="ceiling violated"):
        RowBudgetController(3).plan(4, 0, np.arange(4), np.ones(4, bool))
