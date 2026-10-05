# Controlled room0 retention diagnostic: seed 0

**Status: CONFIRMED single-seed diagnostic; stop before seeds 1 and 2.** Runner revision `335f7776a011d85d4f5fd0a4e6446b0b674ca8c6` on `phillip/pruning-baseline`. Normal `slam.py` and V1 weights/policy semantics were not changed. The preceding online MonoGS room0 triplet remains the upstream-maintenance comparison; this controlled runner is a deliberately simplified mapper and its probe errors are **not ordinary online SLAM ATE**.

## Protocol and control validity

- Replica room0 frames 0–379, all 761 source files matching the existing local manifest (`sha256 1f77e47818d5d27aca706bdf9db0103b2cf9ceeba6a7675614dea2f6a7771ceb`). One frame-0 initialized Gaussian and optimizer state was serialized and restored for every arm. Its complete state digest, including parameter arrays, Adam state and settings, row metadata, visibility and counters, is `8a3522a44f506536609f00a187fd909055b2768e12cb2096761e281e9d68900d`. Restored optimizer bindings were checked.
- Random's existing 57 keyframe IDs, latest-10 FIFO mapping windows, and precomputed older-view IDs share schedule digest `5dfc20c0798b8d6325cf3c98034097651782e4423c689246cae485b334097fd2`. Each arm executed exactly 56 × 150 = 8,400 post-initialization Gaussian optimizer steps. Mapping cameras stayed at GT pose with zero exposure; every retained camera's pose/exposure signature was asserted during the run.
- Fifty-six cached RGB-D Gaussian batches had individual SHA-256 digests checked before every insertion. Every arm admitted all **713,944** candidate rows in identical batch order. At each event, post-retention and post-insertion occupancy matched across arms; pressured insertions finished at 38,000 rows. Maximum observed rows were 38,000, below K=40,000, with zero violations. No adaptive densification, ordinary opacity/size/covisibility deletion or opacity reset ran after initialization.
- A frozen copy of each arm's frame-344 map rendered return frames 345–370 and supported exactly 26 independent 100-iteration localization probes. Each started from GT pose at f−1. Every probe passed parameter, optimizer, metadata and RNG non-mutation checks; probe outputs were discarded and never supplied to mapping or retention. No probe failed by the run's nonfinite-output criterion. Initial translation RMSE was **19.379 mm** in every arm.
- The preflight probe and rendering evaluation completed before comparative execution. Three focused schedule/spatial tests, Python compilation and `git diff --check` passed. `scripts/analyze_controlled_retention.py` passed all four-arm ledger, hash, event, probe and spatial-diversity assertions. Detailed per-event, per-probe, per-view, allocator and hash data are in the ignored local `MonoGS_System/MonoGS/results/controlled_retention_seed0/` directory; the compact summary is committed under `docs/artifacts/controlled_retention_seed0/`.

## Primary return results

Translation is direct error in the common GT coordinate system; no per-policy alignment. Lower is better for all error metrics and LPIPS.

| Arm | Return probe RMSE mm | p95 / max mm | Mean rotation deg | Persistent PSNR / SSIM / LPIPS | Persistent depth MAE mm / completeness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.429 | 0.558 / 0.633 | 0.033 | 36.60 / .9581 / .1065 | 9.16 / 1.000 |
| V1 | 0.480 | 0.808 / 0.987 | 0.028 | 36.16 / .9575 / .1086 | 12.90 / 1.000 |
| No-T | **0.408** | 0.607 / 0.766 | **0.021** | 35.82 / .9567 / .1094 | 14.18 / 1.000 |
| Spatial Random | 0.507 | 0.762 / 0.869 | 0.024 | 36.19 / .9553 / .1127 | 9.89 / 1.000 |

The persistent metrics render the **frame-344 map** at all 26 return views before return geometry is inserted. No-T has the lowest probe RMSE, while Random has the strongest persistent PSNR, SSIM, LPIPS and depth MAE among these arms. Rotation and translation rankings differ; no single arm dominates every localization metric.

## Final reconstruction and diversity

Final reconstruction uses the existing 21 fixed rendering IDs at GT poses. Effective cells are `exp(Shannon entropy)` of row population in origin-anchored 0.5 m world cells. The departure mean uses scheduled keyframe states in frames 120–320. Lineage age is in camera frames, reported as p10 / median / p90 at frame 344.

| Arm | Final PSNR / SSIM / LPIPS | Final depth MAE mm | Effective cells: departure / f344 / final | Occupied cells f344 | Top-10 cell mass f344 | Effective origins f344 | Lineage age p10 / median / p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 34.62 / .9418 / .1221 | 11.62 | 226.4 / 228.1 / 218.3 | 418 | .1329 | 25.21 | 2 / 86 / 297 |
| V1 | 30.96 / .9005 / .1748 | 22.05 | 165.6 / 165.2 / 143.9 | 434 | .1815 | 18.85 | 2 / 124 / 344 |
| No-T | 31.03 / .9003 / .1796 | 25.89 | 159.4 / 155.0 / 135.6 | 414 | .2041 | 23.49 | 2 / 51 / 277 |
| Spatial Random | 34.29 / .9390 / .1284 | **11.58** | **288.3 / 304.6 / 300.3** | **482** | **.0879** | 25.19 | 2 / 81 / 293 |

All final rendered-depth completeness values are 1.000 under the run's `depth > 0.01 m` completeness definition. The raw per-view file records each depth MAE and completeness. Spatial Random has 33.5% more effective cells at f344 than Random and less concentration, but worse return probe RMSE by 0.078 mm (18.2% relative to Random), slightly worse persistent RGB metrics, and no meaningful final reconstruction advantage except a 0.04 mm lower final depth MAE. V1 occupies *more* cells than Random (434 versus 418) yet has much lower effective-cell count; row-mass concentration, not simply occupied-cell count, differs.

## Cost and interpretation

| Arm | Mapping s | Ranking/admission s | Probe s | CUDA peak allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| Random | 584.1 | 2.2 | 19.3 | 1177 / 1298 |
| V1 | 599.3 | 2.4 | 19.8 | 1215 / 1430 |
| No-T | 589.7 | 2.4 | 19.7 | 1249 / 1424 |
| Spatial Random | 583.5 | 5.9 | 19.2 | 1123 / 1360 |

These are sequential single-process wall and PyTorch allocator observations. Mapping time includes geometry-dependent rasterizer work; allocated/reserved peaks do **not** establish an application-wide GPU-memory budget. Ranking/admission time includes planning, deletion, synchronization and upload planning, so it is not pure score computation. The current runner does not separate those costs further.

**A.** Random still beats V1 on the controlled return probe (0.429 versus 0.480 mm) and on persistent/final rendering. Equalized scheduling and mapping work therefore do not erase the seed-0 Random advantage in this simplified setting. They also do not prove retention is the sole cause of the old online ATE gap; optimizer and visibility state still diverge downstream of retention.

**B.** Spatial Random increases diversity: 304.6 versus 228.1 effective cells at f344, with 482 versus 418 occupied cells and less top-ten concentration.

**C.** That extra diversity does not improve return localization or persistent RGB reconstruction over ordinary Random in seed 0. It worsens probe RMSE and most RGB metrics; final depth MAE differs by only 0.04 mm in its favor. This does not justify proposing a spatial-retention method.

**D.** V1 underperforms no-T on controlled return translation RMSE (0.480 versus 0.408 mm), although V1 has better return rendering and no-T has lower mean rotation error. T remains present in the common support floor for both; this is a ranking ablation, not complete removal of tracking history.

**Limit:** all final probe RMSEs are 0.4–0.5 mm, and several differences are below 0.1 mm. One seed, one synthetic GT-pose room, one Random-derived schedule, frozen exposure, disabled ordinary maintenance, and possible residual CUDA variation limit causal generalization. No identical-arm repeat was run, so the run-to-run variation floor is unmeasured. The seed-0 result is a useful negative diagnostic for spatial balancing and V1, not a new method or multi-seed superiority claim. Stop here for review before seeds 1 and 2.
