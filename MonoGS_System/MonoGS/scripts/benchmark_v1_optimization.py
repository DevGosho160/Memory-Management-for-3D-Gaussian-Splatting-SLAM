"""Paired CPU timing of frozen old and optimized V1 planner/feedback paths."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_utils.budget_controller import RowBudgetController
from project_utils.gaussian_metadata import GaussianMetadata


def old_plan(maximum, count, growth, priority, protected):
    priority = np.asarray(priority, dtype=np.int64)
    protected = np.asarray(protected, dtype=bool)
    if protected.size != count or priority.size != count:
        raise ValueError("Ranking/protection length mismatch")
    if len(np.unique(priority)) != count or np.any((priority < 0) | (priority >= count)):
        raise ValueError("Ranking must be a permutation")
    costs = [1] * growth
    if any(cost < 1 for cost in costs) or sum(costs) != growth:
        raise ValueError("Invalid atomic growth costs")
    low = int(np.floor(maximum * .95))
    if protected.sum() > maximum:
        raise RuntimeError("Protected support cannot fit row ceiling")
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
    keep = protected.copy()
    for index in priority:
        if keep.sum() >= target:
            break
        keep[index] = True
    return keep, admitted


def old_feedback(meta, ids, hits, seen, beta):
    positions = {int(value): idx for idx, value in enumerate(meta.gaussian_id.tolist())}
    for j, value in enumerate(ids.tolist()):
        i = positions.get(value)
        if i is None:
            continue
        meta.tracking_ema[i] = beta ** 4 * meta.tracking_ema[i] + hits[j]
        meta.last_seen_frame[i] = max(meta.last_seen_frame[i], seen[j])
        if seen[j] >= 0:
            meta.observed_since_creation[i] = True


def measure(function, repeats=5):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1000)
    return {"samples_ms": times, "median_ms": statistics.median(times)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n, maximum = 32476, 40000
    ids = np.arange(n, dtype=np.int64)
    protected = np.zeros(n, dtype=bool)
    protected[:1000] = True
    zeros = torch.zeros(n)
    never = torch.full((n,), -1, dtype=torch.int32)
    beta = 2 ** (-1 / 20)
    def new_feedback():
        meta = GaussianMetadata()
        meta.append(n, 0)
        meta.apply_feedback(ids=ids, version=meta.map_version, sequence=0,
                            frames=4, weighted_hits=zeros, last_seen=never, beta=beta)
    def scalar_feedback():
        meta = GaussianMetadata()
        meta.append(n, 0)
        old_feedback(meta, ids, zeros, never, beta)
    result = {"scope": "Paired CPU synthetic metadata/masks at historical V1 final row count; no SLAM run",
              "n": n, "maximum": maximum}
    for growth, label in ((0, "no_pressure"), (12729, "insert_pressure")):
        result[label] = {
            "old": measure(lambda: old_plan(maximum, n, growth, ids, protected)),
            "new": measure(lambda: RowBudgetController(maximum).plan(n, growth, ids, protected))}
    result["feedback"] = {"old": measure(scalar_feedback),
                          "new": measure(new_feedback)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: {version: values["median_ms"] for version, values in entry.items()}
                      for key, entry in result.items() if isinstance(entry, dict)}, indent=2))


if __name__ == "__main__":
    main()
