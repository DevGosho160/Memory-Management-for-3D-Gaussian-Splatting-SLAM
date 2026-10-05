# V1 separated-return experiment: Replica room0, K=40,000

**Decision:** The three-run seed-0 comparison does not establish independent revisit value for persistent tracking support. Full V1 has slightly lower overall error than no-T, but higher return error than both no-T and deterministic random. No V2 mechanism is scientifically justified by this triplet. These are one-seed outcomes, not superiority claims.

CONFIRMED denotes source, saved telemetry or reproduced analysis; INFERENCE denotes interpretation; OPEN denotes missing causal evidence. Planner/diagnostic checkpoint: `6eda3f3`; all three SLAM manifests report that revision. The results documentation is the following local checkpoint commit. No push or V2 implementation was performed.

## Protocol and sequence verification

The executable root is `MonoGS_System/MonoGS`. Run command:

```bash
LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
  .venv/bin/python scripts/run_retention_pilot.py \
  --config configs/research/room0_revisit380_v1.yaml \
  --budget 40000 --revisit-triplet
```

CONFIRMED: the available local TUM dataset directory has no sequence data, and the original local Replica archive is incomplete. The NICE-SLAM Replica archive was queried for poses of all eight supported scenes. Room0 gives the clearest practical separated return found in a contiguous segment of a few hundred frames: source frames 70 and 361 are 0.174 m and about 6 degrees apart, after a maximum 1.13 m departure from frame 70. ORB ratio matches with frame 70 are 488 at frame 361 versus 260 at frame 200 (same 1,000-feature detector). The first visit is approximately frames 60–90, the main departure 120–320, and return 345–370. The post-return tail is only 371–379. Middle views retain some image overlap; this is a geometric return with stronger temporal separation than `office0_slice100`, not proof of complete loss and reacquisition of the same Gaussian support.

The acquisition script saved contiguous source frames 0–379. All 761 files in the local dataset manifest have matching SHA-256 hashes. The three runs use seed 0, the same 40,000 staged/live backend-map row ceiling, 38,000 low watermark, densification cap, support floors, half-life, utility weights, upstream maintenance thresholds, CPU-transfer workaround and both single-thread scheduling flags. Resolved configs differ only in output directory, ranking policy and no-T's `use_tracking_history=false`. Every run completed 380 frames; fixed pose IDs are 0–379, saved GT poses match exactly, and all use the same 21 explicit rendering view IDs. Keyframe selection remained online and map-dependent. No-T removes the T contribution from **ranking only**; the common spatial floor still uses T.

For interval metrics, each entire estimated trajectory is rigidly aligned once to its identical GT, then errors are sliced into pre-return 0–344, return 345–370 and post-return 371–379. This reproduces each saved overall fixed-frame ATE to less than 1e-10 m. The post interval has only nine frames. Rendering metrics use the final online map at estimated poses, without post-run refinement.

## Planner equivalence and overhead

Fourteen controller/policy/metadata tests passed, including 500 randomized exact comparisons of retained/pruned row masks and growth accounting to the frozen old planner, tiny/floor-bound cases, atomic two-child splits, and scalar-versus-batched feedback equality. Since stable IDs are tied to the same input row order, equal masks give identical retained/pruned Gaussian IDs. `compileall` and `git diff --check` passed.

Paired CPU microbenchmark, 32,476 rows and synthetic masks/metadata, five repetitions each; median milliseconds:

| Path | Old | Optimized | Reduction |
| --- | ---: | ---: | ---: |
| No-pressure planner | 280.98 | 2.72 | 99.0% |
| Pressured insertion planner | 207.60 | 3.20 | 98.5% |
| Metadata feedback update | 419.09 | 1.78 | 99.6% |

The planner selects needed unprotected rows from the unchanged priority permutation after one protected count. A target equal to existing count returns all rows directly. Feedback batches unique-ID Torch updates; duplicate feedback IDs retain the old sequential path. Score weights, half-life, support floors, growth/atomic admission, ranking, watermark and pruning thresholds were not changed. These CPU figures measure the avoidable path, not a matched old/new end-to-end room0 run.

Measured backend policy plus frontend feedback wall time is 5.669/5.261/5.133 s for random/V1/no-T, or 0.557/0.554/0.551% of each run's online elapsed interval. This denominator includes queue waits and other work and is not the preregistered compute-time overhead fraction. Optional decision diagnostics were enabled for analysis; their own cost is not isolated in these timers. The prior unchanged V1 office0 pilot recorded 17.8 s (7.08% of its elapsed interval), but the datasets and online schedules differ, so that is contextual rather than a paired end-to-end speedup.

## Three-run results

ATE and maximum frame error are millimeters. All rows use the same strict controller; CUDA columns are **backend PyTorch process** peak allocated/reserved MiB, not application-wide memory.

| Policy | Overall ATE | Pre | Return | Post | Max error | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic random | 1.132 | 1.157 | **0.804** | **0.954** | 3.368 | 34.641 | **0.9416** | **0.1362** |
| Full V1 tracking support | **1.134** | **1.126** | 0.927 | 1.807 | 4.111 | 34.508 | 0.9350 | 0.1571 |
| No-T ranking ablation | 1.192 | 1.209 | 0.848 | 1.349 | 3.986 | **34.652** | 0.9396 | 0.1545 |

Random's overall ATE is numerically 0.0028 mm below V1; that tiny one-run gap is not a meaningful superiority result. The return gap is larger: V1 is 15.3% above random and 9.3% above no-T. V1 is 4.8% below no-T overall, with a 6.9% advantage in the long pre-return interval; the direction reverses on the return and short tail. V1 has lower SSIM and higher LPIPS than both comparators. These differences cannot be assigned solely to ranking because the online map, keyframes and optimization work diverge.

| Policy | Max staged/live / violations | Final rows | CUDA allocated / reserved MiB | Keyframes | Mapping iterations | Policy + frontend feedback s | 0.5 m effective cells / occupied cells | Effective origin count at last decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 40,000 / 0 | 31,032 | 1609.3 / 1752.0 | 57 | 9,506 | 5.669 | 169.6 / 453 | 20.5 |
| V1 | 40,000 / 0 | 24,243 | 1488.1 / 1604.0 | 51 | 8,600 | 5.261 | 127.9 / 487 | 14.6 |
| No-T | 40,000 / 0 | 24,781 | 1464.5 / 1580.0 | 50 | 8,449 | 5.133 | 133.4 / 484 | 18.0 |

Effective cells are exp(entropy of final row mass), origin effective counts are from the final **decision-time** retained distribution, not the exported PLY. V1 occupies more spatial cells than random, but concentrates row mass into fewer effective cells and has fewer effective origins at that decision. It also receives six fewer keyframes and 906 fewer cumulative mapping iterations than random. This is consistent with a coupled allocation/work explanation, not a demonstrated causal mechanism. Growth attempted/admitted/rejected totals are in the machine-readable summary and event sidecars; all policies have budget deletion events and zero row-ceiling violations.

## Return-specific decisions and T interpretation

At the V1 return densification pressure events (frames 349, 354, 359, 365), 2,703–3,628 existing rows are deleted per event. Candidate nonzero-T fractions are 0.669–0.676; median T falls from 0.434 to 0.297. Pairwise candidate Pearson correlations are T–W about 0.39–0.41 and T–R about 0.26–0.30. Thus T has measurable variation and is not identical to either W or R in these decision states. The corresponding no-T run still computes T for shared floors and diagnostics. Its online state is different, so comparing its event rows by position would be invalid.

**Inference:** persistent support may contribute to pre-return tracking in this seed, but no independent revisit advantage is observed. The return ATE ordering is random, no-T, V1. The event summaries do not contain future visibility labels or a same-state counterfactual selection/retention overlap, so they cannot establish whether T picked rows that mattered later. The return remains relatively easy in absolute pose error, and its intervening images retain nonzero overlap with the first visit.

## Confounds and decision

- The first synchronized maps already differ at frame 0 (random 32,833; V1 32,744; no-T 32,715 rows), before a budget deletion in initialization. Equal seed did not enforce deterministic CUDA state. Config equality alone cannot remove that confound.
- Retention changes map-dependent keyframe selection, insertion opportunities, maintenance timing and mapping iterations. These are legitimate end-to-end policy effects but obstruct attribution to the T score alone.
- The CPU-transfer workaround affects process transfer timing, CPU RAM and cross-process GPU allocations. A row ceiling and backend allocator peaks do not prove an application-wide GPU-memory budget.
- The post-return interval contains nine frames; there is one run per policy. Diagnostics are optional and were enabled equally, but their standalone cost was not timed.
- Fixed-frame ATE uses saved final keyframe-refined poses and saved non-keyframe tracking poses; rendering at estimated poses combines map and pose quality. No future visibility labels or same-state alternative selections were saved.

**Decision:** Stop at these three runs. T shows empirical variation and an overall V1-versus-no-T difference, but no demonstrated independent value during the separated return. V2 is **not scientifically justified** by this evidence. A stronger claim would require replicated runs and common-state/future-visibility attribution, outside this authorized stop point.

## Artifacts

- [Machine-readable results](artifacts/retention_room0_revisit/revisit_analysis.json), [paired CPU benchmark](artifacts/retention_room0_revisit/optimization_benchmark.json), [dataset hashes](artifacts/retention_room0_revisit/dataset_manifest.json).
- Per-policy manifests, resolved configs, per-frame diagnostics and optional event sidecars are in `docs/artifacts/retention_room0_revisit/` with `random_`, `v1_` and `no_t_` prefixes. Original local run directories under `MonoGS_System/MonoGS/results/replica_room0_revisit380/` preserve telemetry, trajectories, final PLYs and rendered metrics; they remain ignored by Git.
