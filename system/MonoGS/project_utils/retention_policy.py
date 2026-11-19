"""Policy-independent support floors and deterministic retention rankings."""

import numpy as np


def _splitmix64(values):
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = ((values ^ (values >> np.uint64(30))) *
                  np.uint64(0xBF58476D1CE4E5B9))
        values = ((values ^ (values >> np.uint64(27))) *
                  np.uint64(0x94D049BB133111EB))
        return values ^ (values >> np.uint64(31))


def rank_rows(policy, ids, opacity, last_seen, lineage_birth, tracking_ema,
              window_fraction, frame, probation_active, seed=0,
              use_tracking_history=True, half_life=20):
    """Return best-first row indices; stable ID breaks all non-random ties."""
    ids = np.asarray(ids, dtype=np.int64)
    opacity = np.asarray(opacity, dtype=np.float64)
    last_seen = np.asarray(last_seen, dtype=np.int64)
    lineage_birth = np.asarray(lineage_birth, dtype=np.int64)
    tracking_ema = np.asarray(tracking_ema, dtype=np.float64)
    window_fraction = np.asarray(window_fraction, dtype=np.float64)
    probation_active = np.asarray(probation_active, dtype=bool)
    if policy == "opacity":
        score = np.where(np.isfinite(opacity), opacity, -np.inf)
        return np.lexsort((ids, -score)).astype(np.int64)
    if policy == "random":
        priority = _splitmix64(ids.astype(np.uint64) ^ np.uint64(seed))
        return np.lexsort((ids, np.bitwise_not(priority))).astype(np.int64)
    if policy == "lru":
        return np.lexsort((ids, -lineage_birth, -last_seen)).astype(np.int64)
    if policy != "tracking_support":
        raise ValueError(f"Unknown retention policy: {policy}")
    recency = np.zeros(len(ids), dtype=np.float64)
    seen = last_seen >= 0
    recency[seen] = np.exp2(-np.maximum(0, frame - last_seen[seen]) / half_life)
    recency[probation_active & ~seen] = 1.0
    score = (0.50 * tracking_ema if use_tracking_history else 0.0)
    score = score + 0.25 * window_fraction + 0.15 * recency + 0.10 * opacity
    score = np.where(np.isfinite(score), score, -np.inf)
    return np.lexsort((ids, -score)).astype(np.int64)


def protected_rows(ids, opacity, origins, xyz, window_masks, tracking_ema,
                   *, cell_m=0.5, cell_min=2, origin_min=64, view_min=128):
    """Common union of actual-view, origin and currently supported cell floors."""
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=bool)
    opacity = np.asarray(opacity, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.int64)
    xyz = np.asarray(xyz, dtype=np.float64)
    tracking_ema = np.asarray(tracking_ema, dtype=np.float64)
    views = [np.asarray(mask, dtype=bool) for mask in window_masks]
    if any(len(mask) != n for mask in views):
        raise ValueError("Window visibility is not row aligned")
    order = np.lexsort((ids, -np.where(np.isfinite(opacity), opacity, -np.inf)))
    selected = np.zeros(n, dtype=bool)
    for mask in views:
        selected[order[mask[order]][:view_min]] = True
    for origin in np.unique(origins):
        selected[order[origins[order] == origin][:origin_min]] = True
    supported = np.any(np.stack(views), axis=0) if views else np.zeros(n, dtype=bool)
    supported |= tracking_ema >= 0.2
    cells = np.floor(xyz / cell_m).astype(np.int64)
    groups = {}
    for index in order:
        if supported[index]:
            key = tuple(cells[index])
            if groups.get(key, 0) < cell_min:
                selected[index] = True
                groups[key] = groups.get(key, 0) + 1
    return selected
