"""Validate seed-0 control gates and summarize the four diagnostic arms."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("random", "v1", "no_t", "spatial_random")


def summary(path):
    runs = {arm: json.loads((path / f"{arm}.json").read_text()) for arm in ARMS}
    reference = runs["random"]
    event_fields = ("frame", "post_retention", "after", "admitted", "batch_hash", "window")
    for arm, run in runs.items():
        assert run["initial_hash"] == reference["initial_hash"], "Initial states differ"
        assert run["schedule_hash"] == reference["schedule_hash"], "Mapping schedules differ"
        assert run["batch_hashes"] == reference["batch_hashes"], "Cached batches differ"
        assert run["optimizer_steps"] == reference["optimizer_steps"] == 8400
        assert run["row_violations"] == 0 and run["max_rows"] <= 40000
        assert run["occupancy"] == reference["occupancy"]
        assert len(run["events"]) == 56
        for current, expected in zip(run["events"], reference["events"]):
            for field in event_fields:
                assert current[field] == expected[field], (arm, field, current["frame"])
        assert len(run["probes"]) == 26
        assert [row["frame"] for row in run["probes"]] == list(range(345, 371))
        assert all(row["iterations"] == 100 for row in run["probes"])
        for name, expected_ids in (("persistent_return", list(range(345, 371))),
                                   ("final", [20, 40, 60, 70, 80, 100, 120, 140, 160,
                                              180, 200, 220, 240, 260, 280, 300, 320,
                                              340, 350, 360, 370])):
            assert [row["frame"] for row in run["rendering"][name]] == expected_ids
    rows = []
    for arm, run in runs.items():
        probes = run["probes"]
        translation = np.asarray([row["translation_m"] for row in probes])
        rotation = np.asarray([row["rotation_deg"] for row in probes])
        initial = np.asarray([row["initial_translation_m"] for row in probes])
        persistent = run["rendering"]["persistent_return"]
        final = run["rendering"]["final"]
        frame344 = next(row for row in run["diversity"] if row["frame"] == 344)
        def mean(items, key):
            return float(np.mean([row[key] for row in items]))
        rows.append(dict(arm=arm, probe_translation_rmse_m=float(np.sqrt(np.mean(translation**2))),
                         probe_rotation_mean_deg=float(np.mean(rotation)),
                         probe_translation_p95_m=float(np.percentile(translation, 95)),
                         probe_translation_max_m=float(np.max(translation)),
                         probe_initial_rmse_m=float(np.sqrt(np.mean(initial**2))),
                         probe_failures=sum(row["failed"] for row in probes),
                         persistent_psnr=mean(persistent, "psnr"),
                         persistent_ssim=mean(persistent, "ssim"),
                         persistent_lpips=mean(persistent, "lpips"),
                         persistent_depth_mae_m=mean(persistent, "depth_mae_m"),
                         persistent_depth_completeness=mean(persistent, "depth_completeness"),
                         final_psnr=mean(final, "psnr"), final_ssim=mean(final, "ssim"),
                         final_lpips=mean(final, "lpips"),
                         final_depth_mae_m=mean(final, "depth_mae_m"),
                         final_depth_completeness=mean(final, "depth_completeness"),
                         effective_cells_344=frame344["effective_cells"],
                         occupied_cells_344=frame344["occupied_cells"],
                         top_ten_cell_mass_344=frame344["top_ten_cell_mass"],
                         effective_origins_344=frame344["effective_origins"],
                         lineage_age_quantiles_344=frame344["lineage_age_quantiles"],
                         mapping_time_s=run["mapping_time_s"],
                         ranking_time_s=run["ranking_time_s"],
                         probe_time_s=run["probe_time_s"],
                         cuda_peak_allocated_bytes=run["cuda"]["peak_allocated"],
                         cuda_peak_reserved_bytes=run["cuda"]["peak_reserved"],
                         max_rows=run["max_rows"], optimizer_steps=run["optimizer_steps"],
                         initialization_hash=run["initial_hash"],
                         schedule_hash=run["schedule_hash"]))
    with (path / "analysis.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "analysis.json").write_text(json.dumps(rows, indent=2))
    if rows[3]["effective_cells_344"] <= rows[0]["effective_cells_344"]:
        raise AssertionError("Spatial baseline failed diversity gate")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="results/controlled_retention_seed0")
    args = parser.parse_args()
    for row in summary(Path(args.directory)):
        print(row)
