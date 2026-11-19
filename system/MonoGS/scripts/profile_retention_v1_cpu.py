"""CPU-only diagnostic microbenchmarks; synthetic metadata is explicitly labeled."""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_utils.budget_controller import RowBudgetController
from project_utils.gaussian_metadata import GaussianMetadata
from project_utils.retention_policy import protected_rows, rank_rows


def main():
    ply = PlyData.read(ROOT / "results/replica_office0_slice100/2026-10-01-03-01-19/point_cloud/final/point_cloud.ply")["vertex"].data
    xyz = np.column_stack([ply[k] for k in ("x", "y", "z")])
    n, opacity = len(xyz), 1 / (1 + np.exp(-ply["opacity"]))
    ids = np.arange(n)
    protected = np.zeros(n, bool)
    protected[:1000] = True
    result = {"scope": "Unchanged V1 CPU functions; actual final xyz/opacity; SYNTHETIC unavailable metadata/masks/priorities. Not historical per-stage timings.", "n": n}

    def measure(name, function):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            function()
            times.append((time.perf_counter() - start) * 1000)
        result[name + "_ms"] = times

    measure("planner_no_pressure", lambda: RowBudgetController(40000).plan(n, 0, ids, protected))
    measure("planner_insert_pressure", lambda: RowBudgetController(40000).plan(n, 12729, ids, protected))
    meta = GaussianMetadata()
    meta.append(n, 0)
    sequence = [0]

    def feedback():
        meta.apply_feedback(ids=ids, version=meta.map_version, sequence=sequence[0], frames=4,
                            weighted_hits=np.zeros(n, np.float32), last_seen=np.full(n, -1, np.int32), beta=2**(-1/20))
        sequence[0] += 1

    measure("metadata_feedback", feedback)
    measure("support_floors", lambda: protected_rows(ids, opacity, np.zeros(n, int), xyz, [np.ones(n, bool)], np.zeros(n)))
    measure("v1_ranking", lambda: rank_rows("tracking_support", ids, opacity, np.full(n, -1), np.zeros(n), np.zeros(n), np.ones(n), 99, np.zeros(n, bool)))
    profile = cProfile.Profile()
    profile.enable()
    RowBudgetController(40000).plan(n, 0, ids, protected)
    profile.disable()
    pstats.Stats(profile).sort_stats("cumtime").print_stats(8)
    output = ROOT.parents[1] / "docs/artifacts/retention_v1_diagnosis/cpu_microbenchmarks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
