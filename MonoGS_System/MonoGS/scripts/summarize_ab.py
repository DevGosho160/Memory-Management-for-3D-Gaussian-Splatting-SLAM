"""Summarize two completed MonoGS runs without combining per-process peaks."""

import argparse
import csv
import json
from pathlib import Path


def summarize(label, run_dir):
    completion = json.loads((run_dir / "completion.json").read_text())
    online = json.loads((run_dir / "online_summary.json").read_text())
    ate = json.loads((run_dir / "plot/stats_final.json").read_text())
    rendering = json.loads((run_dir / "psnr/before_opt/final_result.json").read_text())
    paths = list(run_dir.glob("telemetry_backend_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one backend CSV in {run_dir}")
    with paths[0].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    policy = [row for row in rows if row["event"] == "extra_opacity_prune"]
    removed = sum(int(row["count_before"]) - int(row["count_after"])
                  for row in policy)
    candidates = sum(int(row["count_before"]) for row in policy)
    return {
        "run": label,
        "directory": str(run_dir),
        "completion": completion["status"],
        "final_gaussians": online["final_gaussian_count"],
        "backend_peak_allocated_bytes": max(int(r["cuda_peak_allocated_bytes"]) for r in rows),
        "backend_peak_reserved_bytes": max(int(r["cuda_peak_reserved_bytes"]) for r in rows),
        "ate_rmse_m": ate["rmse"],
        "psnr": rendering["mean_psnr"],
        "ssim": rendering["mean_ssim"],
        "lpips": rendering["mean_lpips"],
        "online_runtime_s_cpu_transfer_affected": online["online_runtime_s"],
        "extra_prune_events": len(policy),
        "extra_gaussians_pruned_cumulative": removed,
        "extra_pruned_fraction_of_event_candidates": removed / candidates if candidates else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("pruning", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [summarize("A", args.baseline), summarize("B", args.pruning)]
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
