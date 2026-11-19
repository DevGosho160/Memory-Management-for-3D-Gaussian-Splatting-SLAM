"""Export sparse backend keyframe traces from one or more MonoGS run directories."""

import argparse
import csv
from pathlib import Path


def frame_rows(run_dir):
    paths = list(run_dir.glob("telemetry_backend_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one backend telemetry CSV in {run_dir}: {paths}")
    latest = {}
    with paths[0].open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["frame_idx"]:
                latest[int(row["frame_idx"])] = row
    for frame_idx in sorted(latest):
        yield latest[frame_idx]


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs", nargs="+", help="label=run_directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in args.runs:
        label, directory = entry.split("=", 1)
        rows.extend(dict(run=label, **row) for row in frame_rows(Path(directory)))
    write_csv(args.output_dir / "frame_vs_gaussians.csv",
              ["run", "frame_idx", "gaussian_count"],
              [{key: row[key] for key in ("run", "frame_idx", "gaussian_count")}
               for row in rows])
    memory_fields = ["run", "frame_idx", "cuda_allocated_bytes",
                     "cuda_reserved_bytes", "cuda_peak_allocated_bytes",
                     "cuda_peak_reserved_bytes"]
    write_csv(args.output_dir / "frame_vs_cuda_memory.csv", memory_fields,
              [{key: row[key] for key in memory_fields} for row in rows])


if __name__ == "__main__":
    main()
