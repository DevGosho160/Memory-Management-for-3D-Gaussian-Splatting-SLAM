# V1 first constrained comparison (2026-10-01)

Frozen local revision `c89f2af`; Replica office0 100-frame slice; seed 0; opt-in CPU-transfer mode; online evaluation only. All fixed ATE evaluations use IDs 0–99, and every rendering result uses the same 17 declared IDs. Reference retains upstream maintenance but has no added budget controller.

| Policy | Complete | Max / final rows | Violations | Fixed ATE (mm) | PSNR | SSIM | LPIPS | Backend allocated / reserved peak (MiB) | Policy median / p95 (ms) | Total runtime (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 100/100 | — / 55,977 | — | 0.7884 | 42.514 | 0.983454 | 0.03781 | 1096.4 / 1264.0 | — | 295.2 |
| opacity | 100/100 | 39,902 / 33,731 | 0 | 0.9056 | 40.960 | 0.976337 | 0.06495 | 976.2 / 1064.0 | 399 / 474 | 297.0 |
| random | 100/100 | 39,969 / 33,095 | 0 | 0.8811 | 41.270 | 0.978300 | 0.05632 | 983.0 / 1070.0 | 400 / 477 | 298.6 |
| lru | 100/100 | 39,944 / 30,910 | 0 | 1.0562 | 37.329 | 0.964110 | 0.11747 | 963.7 / 1048.0 | 405 / 478 | 263.9 |
| tracking_support | 100/100 | 39,930 / 32,476 | 0 | 1.0083 | 41.046 | 0.978072 | 0.05948 | 945.1 / 1028.0 | 397 / 453 | 251.4 |
| tracking_support_no_T | 100/100 | 39,864 / 32,680 | 0 | 1.0371 | 41.003 | 0.978070 | 0.05950 | 945.4 / 1028.0 | 401 / 473 | 251.5 |

Every budgeted run used K=40,000 and admitted all attempted growth by preemptively deleting existing rows: attempted = admitted, rejected = 0. Per-policy counts are in `first_comparison.csv`; attempted totals differ because online trajectories and keyframe schedules differ. Backend policy plus frontend feedback wall time is about 7% of measured online runtime in these runs. Total runtime and overhead are affected by the CPU-transfer workaround.

The proposed method does not beat the strongest simple comparator: its fixed ATE is 14.44% higher than deterministic random. T improves fixed ATE by 2.78% against the no-T ablation, but one easy 100-frame slice and seed do not establish a robust effect. Proposed fixed ATE (1.0083 mm) exceeds the preregistered reference tolerance (0.9884 mm), PSNR falls 1.468 dB (>1 dB allowed), and LPIPS rises 0.02167 (>0.01 allowed). The 5% overhead target is also missed. This is a negative/underdetermined pilot for superiority, not a successful quality gate.

Peaks are backend PyTorch allocator counters, not whole-application GPU memory. The row ceiling includes staged clone/split rows but is not an allocated-byte budget. CUDA reserved memory is diagnostic. CPU serialization changes memory copies and timing. Keyframe ATE is saved separately in each run as a secondary metric.

Main run summaries: `results/retention_pilot_2026-10-01-02-41-28/summary.json`; no-T summary: `results/retention_pilot_2026-10-01-03-05-54/summary.json`. All six run directories contain resolved config, manifest, completion, traces, fixed trajectory and rendering metrics.

## Growth and secondary metrics

| Policy | Attempted / admitted / rejected growth | Keyframe ATE (mm) | Total backend policy time (s) | Total frontend feedback time (s) |
| --- | ---: | ---: | ---: | ---: |
| opacity | 254,326 / 254,326 / 0 | 0.8169 | 20.87 | 0.29 |
| random | 255,208 / 255,208 / 0 | 0.7811 | 20.76 | 0.29 |
| lru | 229,975 / 229,975 / 0 | 1.0465 | 18.73 | 0.28 |
| tracking_support | 214,888 / 214,888 / 0 | 0.8965 | 17.50 | 0.30 |
| tracking_support_no_T | 215,005 / 215,005 / 0 | 0.9334 | 17.56 | 0.29 |

The policy event median/p95 includes backend admission/ranking, pruning and feedback application, timed with CUDA synchronization. Frontend feedback accumulation/render time is listed separately. Neither total runtime nor policy percentage is a pure GPU-kernel comparison under the CPU-transfer workaround.
