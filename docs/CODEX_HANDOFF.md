# Codex handoff

Date: 2026-09-30. Audited source: `ae8fce4` on main, matching live origin HEAD. One-time research audit complete; only documentation added. No algorithm changes, installs, training, dataset downloads, commits or pushes.

## Confirmed state

- MonoGS core matches upstream `6c9254c`; Devon imported it at `5c2de7a` and reports running on RTX 5060 Ti, but no run artifacts here.
- `ae8fce4` adds parent-process memory-limit call and unused pruning helpers. No proactive budget controller, queue, score policy, memory logging, sweep runner or reversible offload.
- Existing upstream opacity/size pruning executes; window covisibility deletion is monocular-only. n_obs means current-window visibility, unique_kfIDs means origin keyframe.
- Proposal assigns Devon infrastructure and Phillip heuristics/ablations/evaluation; confirm current agreement before duplicating helpers.
- Jetson AGX Orin is explicit in original PDF/presentation; revised proposal omits it. Current target/access unresolved.

## Current task / decisions

Prepare reproducible no-budget reference and passive telemetry before policy novelty. No-budget retains ordinary MonoGS pruning. Recommend Replica office0_sp first; it still has two processes. Do not implement learned policy, CPU/disk hierarchy or loop-closure integration by assumption.

## Relevant files / blockers

- Executable root `MonoGS_System/MonoGS`; start with slam.py, utils/slam_backend.py, utils/slam_frontend.py, gaussian_splatting/scene/gaussian_model.py, project_utils/*, utils/eval_utils.py, configs/rgbd/replica/office0_sp.yaml.
- No dataset/results/environment found at obvious configured paths; system Python3 lacks Torch. WSL GPU is RTX 5070 12,227 MiB; environment/extension compatibility unverified.
- Default memory limits are 64 GiB; CLI/YAML do not expose them. GPU limit is parent per-process allocator setting; Linux CPU limit is RLIMIT_AS, not RSS. Need budget scope, child setup and transient headroom.
- Density helper .cuda() on NumPy fails (isolated reproduction); voxel helper is stub; masked renderer unpack mismatch; keep unused paths untouched until separately integrating/testing.
- --eval forces W&B and 26,000 post-run refinement iterations; avoid for smoke. Online rendering is before_opt; FPS includes throttling/waits. Seeding helper exists but is never called.
- New literature overlap: Pocket-SLAM contribution + tile budgets. Novelty remains unproven; see RESEARCH_PLAN.md.

## Next three actions

1. Ask Devon: “Can you share your successful run's command, resolved config, MonoGS revision, environment/build versions, dataset path and logs/ATE/rendering outputs? Are you still owning budget/queue/instrumentation/runner, and should I extend map_pruning.py? Does eviction mean permanent deletion or offload/reload, and is AGX Orin still required/available?”
2. Recover/validate that environment and dataset; record a local run manifest and reproduce an unbounded headless Replica office0_sp smoke run with existing maintenance. No --eval/refinement; explicitly disable custom limits. If original validated sequence is different, recover it first.
3. Add passive per-process Gaussian/memory/peak/timing logs around initialization, insertion, mapping, densification and sync; repeat full office0 reference, preserving online ATE/rendering and completion status. Capture CPU retained-keyframe growth and simultaneous GPU observation. Implement no policy yet.

Validation completed in audit: AST parsing 32 Python files; config inheritance/scheduling/path inspection; Git blob provenance comparison; stubbed limiter checks; isolated NumPy density failure. No CUDA SLAM execution or baseline metrics obtained.

## 2026-09-30 implementation milestone: branch, data, and telemetry (in progress)

- Revision: working branch `phillip/pruning-baseline` from `ae8fce4`; no commit or push. Pre-existing untracked `AGENTS.md` and `docs/` preserved.
- CONFIRMED: the public NICE-SLAM Replica archive is reachable. `scripts/prepare_replica_office0_slice.py` extracted frames 0–99 of office0, RGB and depth, plus the original trajectory into ignored `datasets/replica/office0_slice100/`; `slice_manifest.json` records source and SHA-256 hashes. This is a declared 100-frame slice, not a full-office0 result. `configs/rgbd/replica/office0_slice100_sp.yaml` selects it.
- Code changes in progress: `utils/run_telemetry.py` writes per-process CUDA allocated/reserved/cumulative peaks, Gaussian count, keyframe counts, iteration, event and CPU enqueue time. Backend records insertion, initialization, mapping, upstream densify/prune, covisibility prune and synchronization. Frontend records frame and completion. `slam.py` adds seeded runs, online-only rendering evaluation without the existing 26,000-iteration color refinement, a run manifest, completion marker and online summary. Default `--eval` retains refinement.
- Validation: Python `compileall` passed for edited files. **No MonoGS run or baseline metrics yet.** Local system lacked Torch, CUDA extensions, C++ compiler and development headers; an ignored `.venv`/`.toolchain` was assembled from existing Torch 2.12.1+cu132 and local packages. Open3D, torchmetrics and W&B now import with local library path. Compiling simple-knn is in progress; the rasterizer is not yet built.
- Decision: first comparable result will use the 100-frame slice and the same resolved config/seed for existing MonoGS and an additional opacity-pruning variant. The complete office0 scene remains a later validation. The user explicitly authorized downloading public data and directed implementation here, superseding earlier suggestion to defer infrastructure ownership clarification.

Next three actions:

1. Complete both CUDA extension builds; run application import and a headless online baseline smoke, correcting only environment/config/runtime failures.
2. Run the full 100-frame slice with upstream MonoGS and capture completion, ATE, online rendering, Gaussian/CUDA traces and runtime.
3. Then integrate the simplest opacity helper through `GaussianModel.prune_points`, preserve visibility alignment and compare the same slice/config/seed. Do not implement voxel, learned or reversible policies.

### Environment validation and baseline launch update

- CONFIRMED: both `simple-knn` and `diff-gaussian-rasterization` built locally for RTX 5070 (sm_120), and `simple_knn._C.distCUDA2` ran on CUDA. A full `slam` import and Replica parser check returned 100 frames. Two minimal vendor compatibility fixes were needed for CUDA 13.4/C++20: removed an unused scalar `lerp` overload that collides with `std::lerp`, and included `<cstdint>` for integer types.
- CONFIRMED blocker: launched `slam.py --config configs/rgbd/replica/office0_slice100_sp.yaml --online-eval --seed 0 --no-memory-limits` twice. Both failed **before mapping** while unpickling the initial CUDA tensor in the spawned backend with `CUDA error: invalid resource handle`; a standalone 10-element CUDA tensor `torch.multiprocessing` spawn test reproduced it. Removing the 13.4 library path did not help. `cudaMallocAsync` allocator cannot share IPC handles; expandable-segments allocator failed with `pidfd_getfd: Operation not permitted`. These observations support a WSL CUDA IPC limitation, not a MonoGS algorithm failure. Failed run directories: `results/replica_office0_slice100/2026-09-30-18-29-51` and `2026-09-30-18-30-34`; both have `completion.json` marked failed. No ATE, rendering or complete Gaussian trace exists yet.
- OPEN QUESTION sent to user: allow a clearly labeled local communication workaround, which changes transfer/runtime/memory accounting, or retain standard two-process baseline and move execution to a compatible environment. Do not present a workaround as upstream memory behavior.
- CONFIRMED independent smoke: a one-iteration, in-process first-frame Replica office0 initialization using the existing `GaussianModel`, `BackEnd.add_next_kf`, CUDA rasterizer, backward/optimizer, and upstream `densify_and_prune` completed. Gaussian rows: 25,461 inserted, 25,677 after initialization; PyTorch CUDA peak allocated 128,137,216 bytes. This is a kernel/runtime smoke only, **not** an online baseline trace or a representative memory result. The exact script/log are ignored local files `.venv/smoke_init.py` and `.venv/smoke-init.log`.
- Validation after telemetry edits: `compileall`, `git diff --check`, and a CSV writer smoke passed. Frontend now reports a dead backend instead of waiting indefinitely, and SLAM cleans up its queues on that failure path.

Next three actions now:

1. Resolve the CUDA IPC execution choice; retain the exact two-process architecture if a compatible runtime/host is available.
2. Complete one upstream-behavior office0 slice run and summarize its trace/metrics before adding optional pruning.
3. Add safe opacity pruning through `GaussianModel.prune_points` and repeat the same configuration/seed, then compare completion, ATE, online rendering, counts and CUDA memory.

## 2026-10-01 local checkpoint

- CONFIRMED: `febf3ce` on `phillip/pruning-baseline` commits the reviewed setup, passive telemetry, 100-frame slice tooling/config, CUDA build compatibility fixes, and documentation. `main` was not modified; nothing was pushed. Git author used Phillip's most recent repository identity for this commit only because this checkout lacked a configured author.
- CONFIRMED: staged paths were inspected before commit; no dataset, results, environment, compiler, archive, or build artifact was included. `compileall` and `git diff --cached --check` passed.
- Decision: user authorized an explicitly labeled CPU-transfer compatibility workaround for this WSL CUDA IPC failure, with the same mechanism for A and B. Its transfer time, CPU memory, end-to-end FPS, and cross-process GPU memory effects cannot support final system-level claims.

Next three actions:

1. Implement an opt-in serialized CPU transfer for only the frontend/backend queues and initialize the empty backend model in the child so startup also avoids CUDA IPC. Verify it with a minimal multiprocessing test.
2. Complete Phase A on 100-frame office0 slice with seed 0, online evaluation, no custom pruning or memory limit; preserve exact artifacts and summarize before B.
3. Add one deterministic opacity policy via `prune_points`, keep the same transfer/config/seed, complete Phase B, and produce comparison and plotting CSVs.

### CPU-transfer and Phase A progress

- Opt-in `--cpu-transfer` wraps only the frontend/backend multiprocessing queues with `torch.save`/`torch.load` through CPU bytes, retaining each tensor's device tag. The backend constructs its initially empty GaussianModel/background in the child, avoiding startup CUDA IPC. Default mode remains the original queue/model transfer. A standalone spawned-process test passed CUDA and CPU tensors in both directions and preserved values/devices.
- First CPU-transfer run reached frontend frame 100 and produced a full mapping trace, but final ATE failed because local `evo==1.37.1` removed `trajectory.align_trajectory`. Run `results/replica_office0_slice100/2026-10-01-00-38-51` is marked failed, so it is not Phase A. The repository's `evo==1.11.0` was incompatible with Python 3.14; we returned to 1.37.1 and changed only ATE alignment to the current `PosePath3D.align` API. Posthoc evaluation of its saved 19-pose trajectory succeeded, and LPIPS model initialization succeeded. Phase A is being rerun for a complete record.
- `scripts/export_frame_traces.py` has been smoke tested on the failed run; it exports measured backend keyframe samples to `frame_vs_gaussians.csv` and `frame_vs_cuda_memory.csv`. These are sparse keyframe samples, not an inferred value for every camera frame.

### Phase A completed before pruning integration

- CONFIRMED complete run: `results/replica_office0_slice100/2026-10-01-00-44-52/`, `completion.json` status `completed`, 100 frames, seed 0, 19 retained keyframes. Exact command from `MonoGS_System/MonoGS`: `PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" .venv/bin/python slam.py --config configs/rgbd/replica/office0_slice100_sp.yaml --online-eval --seed 0 --no-memory-limits --cpu-transfer`. Resolved `config.yml`, `run_manifest.json`, per-process telemetry CSV, trajectory/ATE files, online rendering metrics, final PLY and `online_summary.json` are in that run directory. No 26k refinement was run.
- Final Gaussian count **54,918**. Backend CUDA peak allocated **1,148,377,088 bytes**, peak reserved **1,302,331,392 bytes** (cumulative PyTorch allocator counters from the backend CSV). Online runtime **286.886375 s**, end-to-end FPS 0.34857. ATE RMSE **0.0006086733 m** on 19 keyframes. Before-opt PSNR **42.3556616**, SSIM **0.98283933**, LPIPS **0.03959721**. Final PLY row count matches 54,918. Backend telemetry has 3,856 data rows and upstream densification/pruning events.
- Limit: CPU serialization affects transfer time, CPU RAM, end-to-end FPS, and cross-process GPU memory behavior. These values compare A/B under the same workaround; they are not final system-level memory/throughput benchmark claims. Allocator peak is backend-process scope, not whole-application/device peak.
- Decision for B: fixed opacity threshold 0.5, applied after existing keyframe mapping/maintenance, preserving at least one Gaussian. New Gaussians enter at opacity 0.5, and the final A map has 0.91% below 0.5. This is a simple, conservative extra pruning baseline, not a budget controller or novel policy.

Next three actions:

1. Add only the opt-in opacity helper/backend hook, filter visibility caches with the same pre-prune mask, and validate optimizer/metadata alignment on a small model.
2. Run B with the same scene, seed, CPU transfer, no memory limits, and online evaluation; preserve all artifacts.
3. Export A/B frame/count and frame/backend CUDA traces, compute matched metrics and deletion counts, report limitations and Git status without pushing.

### Phase B launched

- B uses exact A command plus only `--opacity-prune-threshold 0.5`; run directory `results/replica_office0_slice100/2026-10-01-00-51-48/`. `config.yml` and manifest record the enabled policy. Same 100-frame data, seed 0, CPU transfer, no custom memory limit, headless online metrics, and no refinement.
- The existing `project_utils.map_pruning.prune_by_opacity` now returns a one-dimensional pre-prune mask, preserves the highest-opacity row if all would be removed, and calls `GaussianModel.prune_points()` for actual deletion. The backend applies it after keyframe mapping and filters window visibility arrays by that same mask. A three-row real GaussianModel test with populated Adam state confirmed row alignment of optimizer moments, origin IDs, and observation counts, including the keep-one case. `compileall` and `git diff --check` passed.
- Early B telemetry confirms `extra_opacity_prune` events at keyframes 4, 8, and 13, removing 195, 217, and 245 rows respectively. Completion and quality are pending; do not claim improvement yet.

### Final comparable pair correction (2026-10-01)

- The first completed A/B pair above is retained for diagnosis but superseded: B's last keyframe at frame 99 was still queued when the frontend finalized at frame 100. Its backend later reached 54,920 Gaussians while the evaluated frontend map had 52,464. `FrontEnd.run` now waits for any outstanding final keyframe before final save/evaluation. This changes end-of-sequence synchronization, not tracking/mapping equations. Both A and B are being rerun with the fix.
- A second reproducibility issue was found before accepting the pair: Open3D's `random_down_sample` used its own RNG. Backend now seeds `o3d.utility.random` with the same seed 0 as Python/NumPy/Torch. Repeated Open3D downsampling with seed 0 selected identical points in an isolated check. A partial rerun before this fix was stopped and is not a result.
- CONFIRMED final-pair A: `results/replica_office0_slice100/2026-10-01-01-01-20/` completed 100 frames with frontend/backend final counts both **55,365** at frame 99. Backend CUDA peak allocated **1,149,759,488 bytes**, reserved **1,302,331,392 bytes**; runtime **286.66946875 s**; ATE **0.0007402668 m**; PSNR **42.37517817**, SSIM **0.98324368**, LPIPS **0.03846904**. Exact command is the Phase A command above. Resolved config, manifest, completion, telemetry, trajectory, rendering and PLY are saved. B with the same code/data/seed/transfer/evaluation plus opacity threshold 0.5 has been launched; results pending.
- Prior A/B keyframe IDs diverged after frame 36. Existing MonoGS ATE and rendering evaluation therefore used policy-dependent keyframe/non-keyframe sets. This remains a quality-comparison limitation unless the final pair happens to select identical IDs; the final report must check and disclose it. No new evaluation viewpoint freeze was added to this first result.

### Final 100-frame A/B result (2026-10-01)

- CONFIRMED final pair: A `results/replica_office0_slice100/2026-10-01-01-01-20/`, B `results/replica_office0_slice100/2026-10-01-01-06-28/`, relative to `MonoGS_System/MonoGS`. Both `completion.json` files report `completed`; both processed 100 frames, finished backend mapping through frame 99, and have matching frontend/backend final map counts and 19 retained keyframes. Each directory contains `config.yml`, `run_manifest.json`, backend/frontend telemetry, online summary, trajectory/ATE and before-opt rendering metrics. The final pair supersedes the earlier completed pair and the partial/failed runs above.
- Exact command, from executable root, with working directory `MonoGS_System/MonoGS`: `PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" .venv/bin/python slam.py --config configs/rgbd/replica/office0_slice100_sp.yaml --online-eval --seed 0 --no-memory-limits --cpu-transfer` for A; append `--opacity-prune-threshold 0.5` for B. Both were run headless, without W&B or post-run 26k refinement. The resolved configs match apart from output directory and B's opacity threshold. The dataset's ignored `slice_manifest.json` records source archive and file hashes.
- A: final Gaussians **55,365**; backend peak CUDA allocated **1,149,759,488 B**; peak reserved **1,302,331,392 B**; ATE RMSE **0.0007402668 m**; online PSNR **42.375178**, SSIM **0.98324368**, LPIPS **0.03846904**; online runtime **286.6695 s**.
- B: final Gaussians **54,989**; backend peak CUDA allocated **1,148,273,152 B**; peak reserved **1,302,331,392 B**; ATE RMSE **0.0007055720 m**; online PSNR **42.264529**, SSIM **0.98299556**, LPIPS **0.03896548**; online runtime **289.6299 s**. Eighteen additional opacity-prune events removed **6,396 cumulative rows** (0.894% of the summed candidate counts at those events). Final map has **376 fewer rows (0.679%)** than A; backend peak allocated is **1,486,336 B (1.42 MiB)** lower and peak reserved is unchanged. Cumulative removals include rows that would later have been replaced/densified/pruned by upstream behavior; they are not unique final-map savings.
- CONFIRMED quality comparison detail: A retained keyframe 52 where B retained 53; the other 18 retained keyframe IDs match. ATE therefore evaluates a slightly different keyframe set. The evaluator samples every fifth non-keyframe; all **17 rendering viewpoint IDs match exactly** between A and B. The two runs are one seed on a declared 100-frame slice; the small peak allocated difference does not establish a memory-budget effect.
- CSV artifacts: `results/replica_office0_slice100/comparison_final_2026-10-01/comparison.csv`, `frame_vs_gaussians.csv`, and `frame_vs_cuda_memory.csv` (plus `comparison.json`). Frame plots contain **measured backend keyframe samples** only; they do not interpolate each camera frame. Per-process telemetry also records mapping time, retained keyframe count, iteration, upstream densification/pruning and extra-opacity events. Artifacts are ignored locally and intentionally absent from Git.
- LIMITATION: the opt-in CPU serialization changes CPU/GPU transfer timing, total CPU memory, end-to-end FPS/runtime, and cross-process GPU memory behavior. Neither runtime/FPS nor these backend allocator peaks are final application-wide system benchmark claims. No custom memory limit was active. No obvious crash or failed final synchronization occurred in either final run; only modest Gaussian/allocated-peak reduction and a slight rendering-quality decrease were observed.

Next three actions:

1. Keep the first A/B result and scripts on the local branch; reproduce on a CUDA IPC-capable environment to establish ordinary cross-process memory/throughput measurements.
2. Freeze evaluation viewpoint IDs or report common-keyframe ATE on the saved trajectories before interpreting small ATE changes; repeat the same single policy on a full scene or second seed for robustness.
3. Decide with Devon how this measured opacity baseline fits the budget/controller work, without treating a Gaussian-count cap as an application-wide GPU-memory budget.

### 2026-10-01 research-design session: first retention method specified

- Design artifact: `docs/RETENTION_METHOD_V1.md`, against clean current revision `df425c85f34561ec4330148d18f9a25c59c39b2e`. No implementation, SLAM experiment, environment change, full repository re-audit or another-machine reproduction performed. This user-authorized algorithm-design session supersedes the preceding next-action suggestion to prioritize reproduction on another machine.
- CONFIRMED saved-telemetry count decomposition: both final runs inserted 254,584 rows. A initialization adds 1,089 net / B 1,088; ordinary mapping densify/prune removes 200,308 net / B 194,287. B's 6,021 fewer ordinary net deletions offsets nearly all its 6,396 extra removals, giving -376 final rows. Gross clone/split/deletion components and row identities are absent, so more densification versus fewer ordinary deletions remains unresolved; do not assert direct regeneration of specific deleted rows.
- CONFIRMED peak localization: keyframe 93, ordinary mapping iteration 3490 (A) / 3478 (B), after insertion at iteration 3466 and before upstream densify/prune at 3500, extra B pruning at 3617 and frontend synchronization. Maximum sampled counts are 65,583 / 65,286 at frame 99; no new allocated peak there. Sub-operation attribution inside mapping is unmeasured. Final reserved peak first appears at frame 93 mapping iteration 3468 / 3469.
- Saved-trajectory analysis only: common 18-keyframe rigid-aligned ATE is A=0.0007549439273923975 m / B=0.0007205770070285981 m. No new SLAM execution. This does not establish policy superiority. The design freezes all-frame pose IDs and the existing 17 shared rendering views in upcoming experiments.
- PLANNED method: common preemptive admission controller with a strict backend-map row ceiling including gross clone/split staging, separately calibrated/measured backend allocated-byte guard, stable per-row metadata and aggregated frontend tracking feedback. Candidate retention score is 0.50 tracking-hit EMA + 0.25 current-window visibility fraction + 0.15 last-seen recency + 0.10 opacity, with common keyframe/actual-view/coarse-world-cell support floors. Visibility remains a tracking-support proxy, not loss sensitivity or proven future utility.
- PLANNED bounded matrix: 50k/40k live-row ceilings; opacity/random/LRU/proposed through the same controller; one fixed-evaluation unbounded reference; no-history/no-protection ablations at 40k; proposed versus strongest simple comparator at 95% of observed backend allocated peak. 13 primary runs, at most 16 with paired second-seed/reference. All constants, forecast limitations, exact hook files/functions, invariants and proposed command/config contract are in the design artifact. Those new runner/config files do not yet exist.
- Primary-source targeted reread: Pocket-SLAM already combines contribution ranking with tracking-gradient tile budgets; DiskChunGS already uses preemptive chunk loading/LRU and Gaussian residency budgets; MemGS already merges geometric redundancy in voxels. Novelty remains unproven; the differentiated hypothesis is temporal tracking-support retention beyond recency/window/opacity under common feasible constraints, not budget plumbing or coverage alone.

Next three actions for implementation, authorized only when the user starts that session:

1. Give GPT-6 Sol Medium `docs/RETENTION_METHOD_V1.md`; implement pure controller/ranking and stable metadata lifecycle first, coordinating the interface with Devon. Validate synthetic optimizer/row/EMA/staging invariants.
2. Integrate tracking feedback, pre-allocation hooks, phase peak measurement and fixed evaluation on the current local environment; run a declared short smoke and one 40k proposed pilot. Preserve disabled upstream behavior and identical CPU transport.
3. Run the small comparator/ablation matrix, inspect actual compliance before quality, and record preliminary positive/negative results before Thursday October 8. No learned policy, new hardware reproduction, paging or full literature implementation is required for this pilot.

### 2026-10-01 V1 implementation: metadata checkpoint

- CONFIRMED source change: `GaussianMetadata` now owns CPU stable IDs, creation and lineage frames, last-seen, tracking EMA, probation expiry, observation flag, and map/keyframe counters. `GaussianModel` appends it for insertion/clone/split, filters it in `prune_points`, initializes it after PLY load, and checks alignment after mutation. Split child metadata follows the same parent `repeat(N)` order as parameter rows. IDs do not recycle after deletion.
- CONFIRMED validation: `LD_LIBRARY_PATH` local environment, `pytest tests/test_gaussian_metadata.py` passed 3 tests, including a CUDA GaussianModel with populated Adam state, clone, split, pruning and preserved moments/step. This checks lifecycle plumbing, not tracking feedback integration or budgeted execution.
- Decision: retain the user's existing uncommitted design artifact and its previous handoff addition; no A/B result artifacts are modified. This checkpoint precedes controller integration.

Next three actions:

1. Implement pure deterministic policy rankings, common support protection and row admission planning with focused tests.
2. Integrate preemptive insertion and gross densification admission into backend/model, preserving the disabled upstream path.
3. Deliver frontend tracking feedback and fixed-frame evaluation, then run a declared smoke and first constrained comparison.

### 2026-10-01 V1 controller and first 40k pilot

- CONFIRMED implementation: one strict row admission planner now drives opacity, deterministic SplitMix64 random, last-seen/recency and the tracking-support score. Shared view, origin and world-cell protection is selected before ranking. Candidate admission is decided from the sampled CPU point cloud before CUDA upload; densification freezes parent IDs and budgets gross clone/two-child split staging. `GaussianModel` asserts the row ceiling at append. Frontend final-pose tracking hits are aggregated once per camera frame and applied by ID/version at the next keyframe. Fixed all-100-frame ATE and the 17 declared rendering views are available while the default evaluator remains available.
- CONFIRMED validation: seven focused policy/metadata tests pass, including real CUDA optimizer compaction and pre-pruned clone/split staging; `compileall` and `git diff --check` pass. One full proposed-policy correctness pilot, `results/replica_office0_slice100/2026-10-01-02-36-11/`, completed 100 frames at 40k with maximum staged/live rows 39,968, final rows 32,344 and zero row violations. Its fixed all-frame ATE is 0.0010362998 m; fixed-view PSNR/SSIM/LPIPS are 41.03818/0.9779206/0.0604378; backend peak allocated/reserved are 991,242,240/1,077,936,128 B. CPU-transfer timing and allocator scope limitations remain.
- LIMITATION: this pilot was launched while subsequent telemetry/manifest and timing edits were still being made, so its manifest records an in-progress working tree. It proves the earlier controller path completed, but is not the frozen-revision comparator result. The first attempt failed at snapshot deepcopy of a bound callback; the second reached frame 63 and exposed a stale-gradient padding bug after pre-pruning. Both were repaired, and the successful third run completed. Their failed artifacts remain in ignored results directories.
- Decision: the user reduced V1's primary enforced budget to a strict Gaussian row ceiling. CUDA allocated/reserved peaks and timing remain measured diagnostics; no whole-application allocated-byte controller claim is made. Keep CPU transfer opt-in and common across comparisons.

Next three actions:

1. Commit the tested controller/comparator integration, then rerun the fixed-evaluation unbounded reference and four policies at 40k from that one revision.
2. Verify every run's completion, fixed pose/view IDs, staged maximum, zero row violations, growth counts and CUDA peaks before interpreting quality.
3. Run the no-T ablation through the same controller, summarize whether tracking history adds value and checkpoint the results locally without pushing.

### 2026-10-01 frozen 40k comparison and no-T ablation

- CONFIRMED frozen implementation revision: `c89f2af7e3967bf75eeb45986aa8025cd5570109` on `phillip/pruning-baseline`; all six run manifests report this revision, an empty tracked diff, seed 0, the same Replica office0 100-frame slice, opt-in CPU transfer, and online-only evaluation. Main summaries/configs/logs: `MonoGS_System/MonoGS/results/retention_pilot_2026-10-01-02-41-28/`; no-T: `.../retention_pilot_2026-10-01-03-05-54/`. Combined local CSV/JSON/Markdown are `first_comparison.*` in the main summary directory. A committed readable result is `docs/RETENTION_PILOT_2026-10-01.md`. Raw per-run directories are listed there and remain ignored locally.
- CONFIRMED validation: all six runs completed 100 frames. Every fixed trajectory contains exactly IDs 0–99 and every rendering evaluation exactly `[5,10,15,20,25,30,35,45,50,55,60,65,70,75,80,90,95]`. All five budgeted runs had maximum staged/live rows below 40,000 and zero row violations. Attempted/admitted/rejected gross growth totals are recorded for every policy; all attempted growth was admitted by preemptive deletion, so rejected growth was zero. The synthetic/CUDA lifecycle and controller tests had passed (7 tests), with `compileall` and `git diff --check`, before the frozen runs.
- CONFIRMED preliminary outcome: fixed-frame ATE (m) reference 0.00078837, opacity 0.00090564, deterministic random 0.00088107, last-seen 0.00105617, tracking-support 0.00100830, no-T 0.00103713. At the same K, proposed tracking-support is 14.44% worse in ATE than the strongest simple comparator (random). Removing T worsens ATE by 2.78% relative to the proposed policy, a small one-seed effect. Proposed PSNR/SSIM/LPIPS are 41.0461/0.978072/0.059475 versus reference 42.5139/0.983454/0.037806; it misses preregistered ATE, PSNR and LPIPS tolerances. Backend policy plus frontend feedback wall time is about 7% of online runtime, above the 5% target. This is a negative/underdetermined pilot for method superiority, not a successful quality gate.
- CONFIRMED memory scope: proposed maximum staged/live rows 39,930, final 32,476; backend PyTorch allocated/reserved peaks 991,017,984/1,077,936,128 B versus reference 1,149,611,520/1,325,400,064 B. This is a strict backend-map row ceiling and backend allocator observation, not an application-wide GPU-byte budget. CPU-transfer timing, CPU RAM and cross-process accounting remain workaround-affected. Reference carries the new passive row metadata but no controller or extra pruning; its SLAM objectives/maintenance are unchanged.
- Decision: do not claim that persistent tracking support beats simple retention at 40k. The requested first comparator and no-T ablation are complete; do not expand to the full proposed matrix or tune weights merely to seek a positive result.

Next three actions:

1. Profile and vectorize the shared CPU ID feedback join/ranking path to address roughly 400 ms median backend policy events, then repeat matched policies if timing claims matter.
2. Validate the same fixed protocol at a less severe common ceiling (50k) or a harder trajectory/second seed before making a general research conclusion; keep all policies under the same controller.
3. Coordinate the row-controller/telemetry interface and observed negative pilot with Devon; retain allocated/reserved peaks as process-scoped diagnostics until a separate byte controller is justified.
