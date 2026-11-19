"""Summarize the fixed room0 return triplet without rerunning SLAM."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


INTERVALS = {"pre_revisit": (0, 344), "revisit": (345, 370),
             "post_revisit": (371, 379)}


def effective_count(counts):
    values = np.asarray(list(counts), dtype=float)
    if not len(values):
        return 0.0
    p = values / values.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def analyze(run):
    directory = Path(run["run_dir"])
    frames = list(csv.DictReader((directory / "frame_diagnostics.csv").open()))
    ids = np.asarray([int(row["frame"]) for row in frames])
    errors = np.asarray([float(row["fixed_frame_error_m"]) for row in frames])
    assert ids.tolist() == list(range(380))
    events = [json.loads(line) for line in
              (directory / "retention_events.jsonl").read_text().splitlines()]
    decisions = [event for event in events if event["event"] in ("insert", "densify")]
    feedback = [event for event in events if event["event"] == "feedback"]
    pose = {}
    for name, (start, end) in INTERVALS.items():
        mask = (ids >= start) & (ids <= end)
        pose[name] = {"rmse_m": float(np.sqrt(np.mean(errors[mask] ** 2))),
                      "max_error_m": float(errors[mask].max()),
                      "mean_error_m": float(errors[mask].mean()),
                      "frames": int(mask.sum())}
    ply = PlyData.read(directory / "point_cloud/final/point_cloud.ply")["vertex"].data
    xyz = np.column_stack([ply[key] for key in ("x", "y", "z")])
    cells = np.floor(xyz / .5).astype(np.int64)
    _, cell_counts = np.unique(cells, axis=0, return_counts=True)
    last = decisions[-1]
    backend_rows = list(csv.DictReader(next(directory.glob("telemetry_backend_*.csv")).open()))
    by_frame = {}
    for row in backend_rows:
        if row["frame_idx"] and ("prune" in row["event"] or
                                 "admission" in row["event"] or
                                 row["event"].startswith("retention_")):
            by_frame.setdefault(int(row["frame_idx"]), []).append(row["event"])
    with (directory / "frame_diagnostics_complete.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(frames[0]) + ["all_pruning_admission_events"])
        writer.writeheader()
        for row in frames:
            writer.writerow({**row, "all_pruning_admission_events": ";".join(
                by_frame.get(int(row["frame"]), []))})
    feedback_rows = [row for row in backend_rows if row["event"] == "tracking_feedback"]
    result = {"policy": run["policy"], "run_dir": str(directory),
              "fixed_ate_overall_m": run["fixed_ate_rmse"],
              "max_frame_error_m": float(errors.max()),
              "intervals": pose,
              "psnr": run["psnr"], "ssim": run["ssim"], "lpips": run["lpips"],
              "row_ceiling": run["row_ceiling"], "max_live_rows": run["max_live_rows"],
              "row_violations": run["row_violations"], "final_rows": run["final_rows"],
              "backend_cuda_peak_allocated_bytes": run["backend_cuda_peak_allocated_bytes"],
              "backend_cuda_peak_reserved_bytes": run["backend_cuda_peak_reserved_bytes"],
              "policy_wall_ms_total": run["policy_wall_ms_total"],
              "frontend_feedback_wall_ms_total": run["frontend_feedback_wall_ms_total"],
              "online_runtime_s": run["runtime_s"],
              "keyframe_count": int(max(int(row["retained_keyframes"])
                                        for row in backend_rows if row["retained_keyframes"])),
              "mapping_iterations": int(max(int(row["mapping_iteration"])
                                            for row in backend_rows if row["mapping_iteration"])),
              "decision_events": len(decisions), "feedback_events": len(feedback),
              "feedback_apply_ms_total": sum(float(row["policy_wall_ms"])
                                              for row in feedback_rows),
              "pressure_events": sum(event["selected_count"] < event["candidate_count"]
                                     for event in decisions),
              "nonzero_T_fraction_at_last_decision": last["nonzero_T_fraction"],
              "T_quantiles_at_last_decision": last["T_quantiles"],
              "correlations_at_last_decision": last["component_correlations"],
              "last_decision_origin_effective_count": effective_count(
                  last["retained_origin_keyframes"].values()),
              "last_decision_spatial_effective_cells": effective_count(
                  last["retained_spatial_cells"].values()),
              "final_spatial_occupied_cells_0_5m": len(cell_counts),
              "final_spatial_effective_cells_0_5m": effective_count(cell_counts),
              "final_spatial_top10_mass_0_5m": float(np.sort(cell_counts)[-10:].sum()/len(xyz)),
              "attempted_growth": run["attempted_growth"],
              "admitted_growth": run["admitted_growth"],
              "rejected_growth": run["rejected_growth"]}
    assert abs(float(np.sqrt(np.mean(errors ** 2))) - run["fixed_ate_rmse"]) < 1e-10
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path)
    args = parser.parse_args()
    runs = json.loads((args.summary_dir / "summary.json").read_text())
    assert [run["policy"] for run in runs] == ["random", "tracking_support", "no-T"]
    assert all(run["completion"] == "completed" for run in runs)
    assert all(run["rendering_ids"] == runs[0]["rendering_ids"] for run in runs)
    results = [analyze(run) for run in runs]
    (args.summary_dir / "revisit_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["policy"], "ATE", result["fixed_ate_overall_m"],
              "return", result["intervals"]["revisit"]["rmse_m"],
              "rows", result["max_live_rows"], "keyframes", result["keyframe_count"])


if __name__ == "__main__":
    main()
