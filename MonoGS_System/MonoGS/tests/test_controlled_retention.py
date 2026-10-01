import numpy as np

from project_utils.controlled_retention import fixed_schedule, spatially_balanced_order


def test_fixed_schedule_repeats_and_uses_only_previous_keyframes():
    ids = [0, 4, 9, 15, 22]
    left = fixed_schedule(ids, window_size=2, steps=7, seed=3)
    right = fixed_schedule(ids, window_size=2, steps=7, seed=3)
    assert left == right
    for frame, entry in left.items():
        assert entry["window"][0] == frame
        assert len(entry["older"]) == 7
        assert all(set(sample).isdisjoint(entry["window"]) for sample in entry["older"])
        assert all(set(sample).issubset(ids[:ids.index(frame)]) for sample in entry["older"])


def test_spatial_population_counts_protected_rows():
    ids = np.arange(7)
    xyz = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [1, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    protected = np.array([True, True, False, False, False, False, False])
    order = spatially_balanced_order(ids, xyz, protected, seed=0, cell_m=.5)
    assert sorted(order.tolist()) == list(range(7))
    assert set(order[:2]) == {0, 1}
    # The least populated cells must be selected before another row in cell 0.
    unprotected = order[2:]
    assert set(unprotected[:2]).issubset({4, 5, 6})


def test_spatial_priority_is_deterministic_under_exhausted_cells():
    ids = np.arange(20)
    xyz = np.array([[0, 0, 0]] * 15 + [[1, 0, 0]] * 3 + [[2, 0, 0]] * 2)
    protected = np.zeros(20, bool)
    one = spatially_balanced_order(ids, xyz, protected, 17)
    two = spatially_balanced_order(ids, xyz, protected, 17)
    assert np.array_equal(one, two)
    assert len(set(one)) == 20
