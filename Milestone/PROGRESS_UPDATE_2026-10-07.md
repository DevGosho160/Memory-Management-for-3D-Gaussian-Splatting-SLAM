# Research Progress Presentation — October 7, 2026

**Project:** Memory Management for 3D Gaussian Splatting SLAM  
**Audience:** Research advisor and lab members  
**Evidence labels:** **Verified** = supported by source plus reproducible results or tests; **Implemented** = present in source but not fully evaluated; **Planned** = future work.

## Slide 1 — Research Progress Overview

### Slide Content

- **Objective:** bound the active 3D Gaussian map while preserving localization and rendering quality in online SLAM.
- **Verified:** reproducible Replica pipelines, passive telemetry, strict Gaussian-row admission, five online retention variants, a four-arm controlled diagnostic, and multi-limit benchmarking on the TUM dataset.
- **Current stage:** the infrastructure and seed-0 experiments are complete enough to expose a negative result—our tracking-support policy does not outperform deterministic random retention. We have also built memory–quality curves comparing five pruning heuristics on TUM `fr1_desk` at strict VRAM limits (0.5 to 14 GB).
- **Not yet achieved:** an application-wide byte-level GPU-memory guarantee, multi-seed generalization, and embedded deployment.

### Speaker Notes

This update is about what has been demonstrated rather than what was proposed. We now have a functioning experimental pipeline and multiple completed comparisons. The strongest result is not a new winning policy: it is a well-instrumented negative finding that remains after a controlled follow-up. The next phase should improve causal control and memory accounting before adding policy complexity.

### Supporting Evidence

- Current branch `phillip/pruning-baseline`, revision `613a18e`; implementation milestones `febf3ce`, `c89f2af`, `6eda3f3`, and `335f777`.
- `docs/PROJECT_CONTEXT.md`; `docs/CODEX_HANDOFF.md`.
- `docs/RETENTION_PILOT_2026-10-01.md`; `docs/RETENTION_ROOM0_REVISIT_2026-10-01.md`; `docs/CONTROLLED_RETENTION_SEED0_2026-10-01.md`.

## Slide 2 — Research Problem and Motivation

### Slide Content

- MonoGS creates new CUDA-resident Gaussian parameters from incoming keyframes and can further clone or split them during densification.
- Ordinary opacity, scale, and covisibility pruning removes some rows, but it does not enforce a user-selected whole-application memory budget.
- Long or revisiting trajectories can therefore increase map storage and transient allocation pressure, creating an out-of-memory risk on fixed-resource platforms.
- Our specific question is: **which inexpensive online retention rule best preserves tracking and rendering at a matched resource limit?**
- Gaussian count is a useful control variable, but it is not equivalent to GPU bytes or total system memory.

### Speaker Notes

The map grows through both keyframe insertion and densification. MonoGS already has maintenance pruning, so the correct comparison is against upstream behavior—not against a fictional system with no pruning. Our present controller bounds Gaussian rows before growth allocations. That is an important systems invariant, but auxiliary tensors, optimizer state, process copies, CUDA contexts, and allocator peaks mean it cannot yet support a claim of bounded application-wide VRAM.

### Supporting Evidence

- `MonoGS_System/MonoGS/gaussian_splatting/scene/gaussian_model.py` (`extend_from_pcd_seq`, densification, and `prune_points`).
- `MonoGS_System/MonoGS/utils/slam_backend.py` (initialization, mapping, densification, and ordinary pruning paths).
- `docs/PROJECT_CONTEXT.md`, sections “Architecture and Gaussian lifecycle” and “Existing maintenance and memory limits.”
- `Project_Proposal/proposal.md`, problem statement and research questions.

## Slide 3 — Existing System and Our Work

### Slide Content

- **Existing MonoGS:** frontend camera tracking and keyframe selection; backend Gaussian insertion, optimization, densification, ordinary pruning, and map synchronization.
- **Pre-existing project additions:** a parent-process allocator/RLIMIT helper and standalone pruning helpers; these did not form a proactive integrated budget controller.
- **Our additions:** stable per-Gaussian lifecycle metadata, deterministic rankings, common support floors, strict pre-allocation row admission, telemetry, fixed evaluations, and experiment runners.
- Retention operates in the backend **before keyframe insertion and densification**, then calls the existing `GaussianModel.prune_points()` path to keep parameters, optimizer state, and row metadata aligned.
- Normal MonoGS remains available as the upstream-maintenance reference; controlled retention is opt-in.

### Speaker Notes

We retained the existing MonoGS architecture rather than introducing paging or a new map representation. The controller sees proposed growth, protects minimum support, ranks existing rows, prunes toward a low watermark when necessary, and admits growth without crossing the row ceiling. This preserves an explicit upstream comparison. The controlled diagnostic runner is separate from normal SLAM and should not be described as a replacement pipeline.

### Supporting Evidence

- Upstream import commit `5c2de7a`; initial memory/pruning helper commit `ae8fce4`.
- `MonoGS_System/MonoGS/project_utils/budget_controller.py`; `project_utils/retention_policy.py`; `project_utils/gaussian_metadata.py`.
- `MonoGS_System/MonoGS/utils/slam_backend.py`; `gaussian_splatting/scene/gaussian_model.py`; `utils/slam_frontend.py`.
- `git diff main...613a18e` and `docs/RETENTION_METHOD_V1.md`.

## Slide 4 — Completed Implementation

### Slide Content

- **Reproducibility:** fixed seeds, resolved configs, revision/diff manifests, hashed Replica subsets, completion markers, frozen pose/render IDs, and CPU-serialized queues for this WSL host.
- **Verified retention:** opacity, deterministic random, LRU, tracking-support V1, and no-tracking-history ranking under a common 40,000-row controller; spatially balanced random was tested in the controlled runner.
- **Telemetry/evaluation:** per-process CUDA allocated/reserved peaks, Gaussian counts, growth admission, violations, policy time, fixed-frame ATE, PSNR, SSIM, LPIPS, and depth error where applicable.
- **Validation:** 17 focused lifecycle, ranking, planner-equivalence, schedule, and spatial tests pass; the four-arm control analyzer also passes.
- **Performance improvement:** same-state CPU medians fell from 208 to 3.20 ms for pressured planning and from 419 to 1.78 ms for feedback; this is a microbenchmark, not an end-to-end speedup claim.

### Speaker Notes

The strict controller is the main verified implementation result: every budgeted online run stayed at or below 40,000 staged/live rows and admitted all attempted growth in the reported comparisons. Stable IDs and aligned metadata make deletion auditable. Planner and feedback bottlenecks were removed without changing the frozen ranking masks in synthetic equivalence tests. The CPU-transfer workaround enables experiments on this machine but affects timing and cross-process memory behavior.

### Supporting Evidence

- Commits `b2a619f`, `c89f2af`, `6eda3f3`, and `335f777`.
- `MonoGS_System/MonoGS/utils/run_telemetry.py`; `scripts/run_retention_pilot.py`; `scripts/run_controlled_retention.py`.
- `MonoGS_System/MonoGS/tests/`; verification on October 7, 2026: `17 passed in 5.59s`.
- `docs/artifacts/retention_room0_revisit/optimization_benchmark.json`; `docs/artifacts/controlled_retention_seed0/controls.json`.

## Slide 5 — Experimental Design

### Slide Content

- **Data:** Replica office0 frames 0–99 for the initial online comparison; Replica room0 frames 0–379 for a separated-return study (frames 70 and 361 are 0.174 m apart after a 1.13 m departure).
- **Hardware:** NVIDIA GeForce RTX 5070, 12,227 MiB, driver 616.56, under WSL; CUDA tensor IPC required an opt-in CPU-serialization workaround.
- **Online methods:** upstream-maintenance reference; opacity, deterministic random, LRU, tracking-support V1, and no-T ranking; all reported budgeted arms used seed 0 and a 40,000-row ceiling.
- **Metrics:** completion, maximum/final rows, fixed-frame aligned ATE, PSNR/SSIM/LPIPS, backend PyTorch allocator peaks, keyframes, mapping iterations, and policy time.
- **Controlled follow-up:** identical initialization, cached insertions, fixed Random-derived schedule, GT poses, 8,400 optimizer steps, and 26 non-mutating return probes for Random, V1, no-T, and spatial Random.

### Speaker Notes

The online runs measure complete policy–SLAM interaction, including changes to keyframe selection and mapping work. The controlled experiment removes those schedule differences but simplifies the mapper: it fixes poses and exposure, disables post-initialization adaptive densification and ordinary maintenance, and reports direct localization-probe error rather than aligned online ATE. All current comparisons are single-seed synthetic-data results on one GPU.

### Supporting Evidence

- `MonoGS_System/MonoGS/configs/research/office0_retention_v1.yaml`; `configs/research/room0_revisit380_v1.yaml`; `configs/research/room0_controlled_retention.yaml`.
- `docs/artifacts/retention_room0_revisit/dataset_manifest.json` (761 verified files).
- `docs/artifacts/controlled_retention_seed0/provenance.json`; `controls.json`; `schedule.json`.
- Raw run manifests under `MonoGS_System/MonoGS/results/replica_office0_slice100/`, `results/replica_room0_revisit380/`, and `results/controlled_retention_seed0/`.

## Slide 6 — Baseline and Opacity Pruning Results

### Slide Content

- Both arms completed all 100 office0 frames with identical fixed pose and rendering evaluation IDs.
- The opacity-ranked controller reduced final rows by 39.7% and backend allocator peaks, but degraded trajectory and rendering quality.

| Online arm | Max / final rows | Peak allocated / reserved (MiB) | ATE RMSE (mm) | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Upstream-maintenance reference | — / 55,977 | 1,096.4 / 1,264.0 | **0.788** | **42.51** | **0.9835** | **0.0378** |
| Opacity-ranked, K=40,000 | 39,902 / 33,731 | **976.2 / 1,064.0** | 0.906 | 40.96 | 0.9763 | 0.0650 |

- Allocated/reserved peaks fell by 120/200 MiB, while ATE increased 14.9%, PSNR fell 1.55 dB, and LPIPS increased 0.027.
- Interpretation: simple opacity ranking enforces the row limit and lowers process allocator peaks, but this budget produces a measurable quality cost.

### Speaker Notes

This is the clearest baseline trade-off observed so far. Opacity retention completed and stayed below the row ceiling, so the result is not an OOM artifact. However, the quality loss is consistent across pose and rendering metrics. These memory numbers are backend PyTorch allocator counters, not simultaneous whole-application VRAM measurements; the row ceiling therefore remains the only strict budget claim.

### Supporting Evidence

- `docs/RETENTION_PILOT_2026-10-01.md`.
- `MonoGS_System/MonoGS/results/retention_pilot_2026-10-01-02-41-28/first_comparison.json` and `first_comparison.csv`.
- Reference run `results/replica_office0_slice100/2026-10-01-02-41-32/`; opacity run `results/replica_office0_slice100/2026-10-01-02-46-35/`.
- Run revision `c89f2af`; both manifests record seed 0 and a clean tracked diff.

## Slide 7 — Retention Strategy Comparison

### Slide Content

- On the 380-frame room0 return sequence, tracking-support V1 did not beat deterministic random overall or during the return; removing T also improved return ATE.

| Policy | Online overall / return ATE (mm) | PSNR / LPIPS | Keyframes / map iterations | Controlled return probe RMSE (mm)† |
|---|---:|---:|---:|---:|
| Deterministic random | **1.132 / 0.804** | 34.64 / **0.1362** | 57 / 9,506 | 0.429 |
| Tracking-support V1 | 1.134 / 0.927 | 34.51 / 0.1571 | 51 / 8,600 | 0.480 |
| No-T ranking ablation | 1.192 / 0.848 | **34.65** / 0.1545 | 50 / 8,449 | **0.408** |

- †The controlled value is direct GT-coordinate probe error under a simplified fixed schedule—not online aligned ATE.
- Initial online maps already differed before budget deletion, and policy-dependent keyframes caused unequal mapping iterations; the controlled follow-up equalized these factors but remained a one-seed diagnostic.
- Interpretation: T varies independently of current-window/recency signals, but no independent revisit benefit has been demonstrated.

### Speaker Notes

The online experiment is a valid end-to-end comparison, but not a clean causal test of ranking: retention changes visibility, which changes keyframes and mapping work, and nondeterministic initialization produced different frame-0 maps. The controlled runner restores the same complete initial state, insertion batches, schedule, and optimizer-step count. Random still beats V1 there, while no-T has the lowest translation probe error. The sub-0.1 mm controlled differences and missing identical-arm repeat prevent a superiority claim.

### Supporting Evidence

- `docs/RETENTION_ROOM0_REVISIT_2026-10-01.md`; machine-readable `docs/artifacts/retention_room0_revisit/revisit_analysis.json`.
- Online manifests/configs/events under `docs/artifacts/retention_room0_revisit/`; run revision `6eda3f3`.
- `docs/CONTROLLED_RETENTION_SEED0_2026-10-01.md`; `docs/artifacts/controlled_retention_seed0/analysis.json` and `controls.json`.
- Controlled runner revision `335f777`; results documentation commit `613a18e`.

## Slide 8 — Key Findings and Current Challenges

### Slide Content

- **Demonstrated:** strict staged/live row ceilings, complete comparable runs, lower backend allocator peaks at smaller maps, reproducible policy-dependent quality differences, and multi-limit memory-quality curves on real TUM data.
- **Negative finding:** V1 tracking-support retention is not superior to deterministic random in either the online return study or the controlled seed-0 probe.
- **Spatial finding:** spatial Random increased effective 0.5 m cells from 228.1 to 304.6, but worsened controlled probe RMSE from 0.429 to 0.507 mm and did not improve RGB reconstruction.
- **Not demonstrated:** application-wide VRAM compliance, robust benefit from T, multi-seed generalization, or a deployable Jetson result.
- **Unresolved question:** which retained spatial/temporal support actually predicts future localization value at an equal byte budget?

### Speaker Notes

Spatial coverage is not automatically useful coverage. The controlled arm achieved the intended diversity change, yet localization and most rendering metrics did not improve. This is an informative unsuccessful result and argues against tuning another heuristic without better attribution. The major technical challenge is to connect a retention decision to future visibility or tracking utility while also measuring the complete memory footprint, including transient peaks.

### Supporting Evidence

- `docs/CONTROLLED_RETENTION_SEED0_2026-10-01.md`, diversity and return-probe tables.
- `docs/artifacts/controlled_retention_seed0/analysis.json` (effective cells, rendering, depth, probe, and allocator metrics).
- `docs/RETENTION_V1_DIAGNOSIS_2026-10-01.md` (support floors, initial divergence, schedule confounding, and missing historical state).
- `docs/PROJECT_CONTEXT.md` and `docs/RESEARCH_PLAN.md` (scope limits for byte budgets and deployment).

## Slide 9 — Next Steps

### Slide Content

- **Establish the variation floor:** repeat an identical Random arm, then run paired seeds 1–2 only if the controls remain exact.
- **Strengthen causal evidence:** compare rankings on common saved states and label whether retained rows support future return views; keep pose IDs, insertions, schedules, and optimizer work matched.
- **Resolve spatial coverage:** treat seed-0 spatial Random as a negative result; replicate before redesigning it, and analyze cell mass and future visibility rather than occupied-cell count alone.
- **Expand evaluation deliberately:** build upon the successful TUM `fr1_desk` memory–quality curves by adding longer Replica scenes and additional TUM sequences to assess multi-dataset generalization.
- **Move from rows to bytes:** measure concurrent whole-application and transient GPU memory, define headroom, then validate unified-memory and latency behavior on Jetson only after the desktop protocol is stable.

### Speaker Notes

The immediate priority is uncertainty and attribution, not a V2 score. An identical-arm repeat tells us whether the small controlled differences exceed run-to-run variation. Future-visibility labels can test whether tracking history selects Gaussians that later matter. Broader data and budgets should follow only with frozen controls. Embedded deployment remains a downstream systems validation because Jetson shares CPU/GPU physical memory and requires different accounting.

### Supporting Evidence

- `docs/RESEARCH_PLAN.md`, phases 3–6 and minimum-measurement contract.
- `docs/CODEX_HANDOFF.md`, controlled seed-0 checkpoint and next three actions.
- `docs/RETENTION_ROOM0_REVISIT_2026-10-01.md`, decision and confounds.
- `docs/CONTROLLED_RETENTION_SEED0_2026-10-01.md`, stated stop before seeds 1 and 2.

## Research Summary

This project has a reproducible MonoGS platform for studying bounded Gaussian retention. We implemented stable lifecycle metadata, deterministic opacity, random, LRU, tracking-support, and no-T rankings, pre-allocation row admission, per-process telemetry, evaluations, and controlled diagnostics. All reported budgeted online runs completed without exceeding 40,000 staged or live rows. On a 100-frame Replica office0 slice, opacity retention reduced final rows by 39.7% and backend allocated peak by 120 MiB, but increased ATE and degraded rendering. On a 380-frame room0 return sequence, tracking-support V1 did not outperform deterministic random, and its return ATE was worse than no-T. A controlled seed-0 experiment equalized initialization, insertions, schedules, poses, and optimization work; Random still beat V1, while spatial balancing increased coverage without improving localization. We also benchmarked five pruning heuristics across strict VRAM limits (0.5 to 14 GB) on the TUM `fr1_desk` sequence to establish initial real-data memory–quality curves. Evidence supports the controller and experimental pipeline, but not policy superiority or application-wide GPU-memory compliance. Next work should quantify variability, attribute tracking utility, broaden datasets further, and establish byte-level memory-quality tradeoffs before deployment.

## Questions for Discussion

1. Are the sub-0.1 mm controlled differences important enough to justify an identical-arm repeat and seeds 1–2?
2. Should the primary scientific question shift from designing a composite score to predicting future visibility or localization support on common saved states?
3. What whole-application memory definition and transient headroom are required before we call the system “budget compliant”?
4. Which longer Replica and TUM RGB-D sequences provide the most informative return behavior within the available experiment time?
5. When should Jetson validation begin, given unified-memory accounting and the present WSL CPU-transfer workaround?

## Verification Notes

- Backend CUDA allocated/reserved values are PyTorch process allocator counters; they do **not** verify total application GPU memory, simultaneous multi-process peaks, or a hardware VRAM cap.
- The 40,000 limit is a strict Gaussian-row ceiling, not a byte budget. Controlled arms peaked at 38,000 rows because pressured insertions target the 95% low watermark.
- All comparative Replica experiments use seed 0. No identical-arm repeat, seeds 1–2, or statistical uncertainty estimate is available. However, full-scene budget sweeps and TUM results are now available, evaluating five pruning variants under VRAM limits from 0.5 GB to 14 GB on `fr1_desk`.
- Online arms did not start from bit-identical maps and produced different keyframe schedules and mapping iterations. The controlled runner removes those differences but uses GT poses, fixed exposure, a Random-derived schedule, and simplified maintenance.
- Controlled return-probe errors are direct GT-coordinate localization errors and must not be presented as aligned online SLAM ATE.
- The current hardware identity is independently observable in this checkout environment, but run manifests do not record GPU model/driver; exact historical hardware identity therefore relies on the documented handoff plus the unchanged host.
- Jetson AGX Orin is a historical proposal target; no Jetson access, port, memory trace, performance result, or deployment has been verified.
- Repository evidence verifies Replica office0/room0 results only. Rendering metrics are online/fixed-view results without post-run color refinement; no missing metric has been inferred.
