"""Sequential fixed-view row-budget pilot. Run from MonoGS executable root."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_utils import load_config


POLICIES = ("opacity", "random", "lru", "tracking_support")


def summarize(run_dir, policy):
    path = Path(run_dir)
    completion = json.loads((path / "completion.json").read_text())
    result = {"policy": policy, "run_dir": str(path),
              "completion": completion["status"]}
    if completion["status"] != "completed":
        result["error"] = completion.get("error")
        return result
    online = json.loads((path / "online_summary.json").read_text())
    fixed = json.loads((path / "plot/stats_fixed_final.json").read_text())
    trajectory = json.loads((path / "plot/trj_fixed_final.json").read_text())
    render = json.loads((path / "psnr/before_opt/final_result.json").read_text())
    result.update(frames=online["frames"], final_rows=online["final_gaussian_count"],
                  runtime_s=online["online_runtime_s"], fixed_ate_rmse=fixed["rmse"],
                  psnr=render["mean_psnr"], ssim=render["mean_ssim"],
                  lpips=render["mean_lpips"], rendering_ids=render["view_ids"])
    if trajectory["trj_id"] != list(range(100)):
        raise RuntimeError(f"Fixed ATE pose IDs differ in {path}")
    if len(render["view_ids"]) != 17:
        raise RuntimeError(f"Fixed rendering IDs missing in {path}")
    backend = next(path.glob("telemetry_backend_*.csv"))
    with backend.open(newline="") as file:
        samples = list(csv.DictReader(file))
    result["backend_peak_allocated_bytes"] = max(
        int(row["cuda_peak_allocated_bytes"]) for row in samples)
    result["backend_peak_reserved_bytes"] = max(
        int(row["cuda_peak_reserved_bytes"]) for row in samples)
    if policy != "reference":
        result.update(json.loads((path / "retention_summary.json").read_text()))
        frontend = next(path.glob("telemetry_frontend_*.csv"))
        with frontend.open(newline="") as file:
            feedback_times = np.asarray([
                float(row["policy_wall_ms"])
                for row in csv.DictReader(file)
                if row["event"] == "tracking_feedback_accumulate"
            ])
        result["frontend_feedback_wall_ms_total"] = float(feedback_times.sum())
        result["frontend_feedback_wall_ms_median"] = float(np.median(feedback_times))
        result["frontend_feedback_wall_ms_p95"] = float(np.percentile(feedback_times, 95))
        if result["row_violations"] or result["max_live_rows"] > result["row_ceiling"]:
            raise RuntimeError(f"Strict row budget violated in {path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/office0_retention_v1.yaml")
    parser.add_argument("--budget", type=int, default=40000)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=list(POLICIES))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--no-tracking-history", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    out = Path("results") / f"retention_pilot_{datetime.now():%Y-%m-%d-%H-%M-%S}"
    out.mkdir(parents=True)
    rows = []
    names = (["reference"] if args.reference else []) + args.policies
    for name in names:
        config = json.loads(json.dumps(base))
        config["Retention"]["enabled"] = name != "reference"
        config["Retention"]["policy"] = name if name != "reference" else "tracking_support"
        config["Retention"]["max_gaussians"] = args.budget
        config["Retention"]["use_tracking_history"] = not args.no_tracking_history
        config_path = out / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        command = [sys.executable, "slam.py", "--config", str(config_path),
                   "--online-eval", "--seed", "0", "--no-memory-limits",
                   "--cpu-transfer"]
        print(f"Starting {name} at K={args.budget}", flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log_path = out / f"{name}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env)
        text = log_path.read_text(errors="replace")
        match = re.search(r"saving results in (\S+)", text)
        if not match:
            raise RuntimeError(f"{name} exited {completed.returncode}; see {log_path}")
        row = summarize(match.group(1), name)
        rows.append(row)
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
        print(f"Finished {name}: {row['completion']}, {row.get('fixed_ate_rmse')}",
              flush=True)
        if completed.returncode or row["completion"] != "completed":
            raise RuntimeError(f"{name} failed; see {log_path}")
    with (out / "summary.csv").open("w", newline="") as file:
        columns = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(out, flush=True)


if __name__ == "__main__":
    main()
