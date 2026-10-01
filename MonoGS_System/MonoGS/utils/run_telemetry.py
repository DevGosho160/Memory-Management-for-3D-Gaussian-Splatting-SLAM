"""Passive, per-process MonoGS telemetry. CUDA counters are allocator counters, not device totals."""

import csv
import os
import time

import torch


FIELDS = (
    "time_wall_s", "pid", "event", "frame_idx", "mapping_iteration",
    "gaussian_count", "retained_keyframes", "window_keyframes",
    "cuda_allocated_bytes", "cuda_reserved_bytes",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    "mapping_enqueue_ms", "count_before", "count_after",
    "attempted_growth", "admitted_growth", "rejected_growth",
    "protected_rows", "policy_wall_ms", "row_ceiling", "row_violation",
)


class RunTelemetry:
    def __init__(self, save_dir, process):
        self.file = None
        if save_dir is not None:
            path = os.path.join(save_dir, f"telemetry_{process}_{os.getpid()}.csv")
            self.file = open(path, "w", newline="", encoding="utf-8", buffering=1)
            self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
            self.writer.writeheader()

    def record(self, event, gaussians, frame_idx=None, iteration=None,
               retained_keyframes=None, window_keyframes=None,
               mapping_enqueue_ms=None, count_before=None, count_after=None, **extra):
        if self.file is None:
            return
        row = dict.fromkeys(FIELDS, "")
        row.update(
            time_wall_s=time.time(), pid=os.getpid(), event=event,
            frame_idx=frame_idx, mapping_iteration=iteration,
            gaussian_count=gaussians.get_xyz.shape[0],
            retained_keyframes=retained_keyframes,
            window_keyframes=window_keyframes,
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            mapping_enqueue_ms=mapping_enqueue_ms,
            count_before=count_before, count_after=count_after,
        )
        row.update({key: value for key, value in extra.items() if key in FIELDS})
        self.writer.writerow(row)

    def close(self):
        if self.file is not None:
            self.file.close()
