# Research plan

Audit date: 2026-09-30; baseline source snapshot `ae8fce4`. This document contains recommendations/hypotheses, not implemented features or results. See PROJECT_CONTEXT.md for source evidence.

## Question and scope decisions

Which inexpensive online Gaussian retention policy preserves localization and rendered-view quality under a measured application memory budget, including allocation peaks?

The current proposal studies a bounded active map using deletion; it does not require CPU/disk hierarchy. Confirm with Devon whether eviction means permanent budget-driven deletion or reversible removal from GPU residency. Treat reversible eviction as a separate, larger architecture task until agreed.

Confirmed documented ownership: Phillip algorithms/evaluation; Devon controller/instrumentation/runner. Reconcile Devon's pruning helpers with this split before changing them. Decide current Jetson commitment and publication ambitions versus course deliverable.

## Baselines

- CONFIRMED code reference: MonoGS importer `5c2de7a`; mapper/tracker/model/evaluator match upstream `6c9254c`. It already performs ordinary pruning, so no-budget does not mean no-pruning.
- OPEN QUESTION measured reference: Devon reports a run in commit message; no reproducible outputs in checkout. No measured unbounded or budgeted baseline established here.
- First scene recommendation: Replica office0, `configs/rgbd/replica/office0_sp.yaml`, because both scheduling flags agree and synthetic depth/poses support controlled diagnosis. Config still spawns a backend. Use TUM RGB-D fr1_desk as fallback if Devon already has it validated; do not invent a new data acquisition task during audit.
- First run recipe must use executable root, exact revision/config/environment/data manifest, fixed seeds in parent AND spawned backend, GUI/W&B disabled, results enabled, eval_rendering disabled for smoke. A short slice must be explicitly implemented/configured later if needed; current loader has no general max-frame option.
- Full reference: complete one fixed scene without new policy, retain existing maintenance; passive telemetry plus online ATE and rendering. Preserve genuinely unbounded limits through a reviewed runner/constructor None override; CLI currently cannot disable custom defaults.
- Start budget comparison at largest feasible fraction, then 75/50/25% as proposed. Fixed camera/renderer/snapshot overhead may exceed low budgets. Mark those budgets infeasible rather than assigning failure to one policy.
- Same insertion/densification/pruning settings, frame schedule and quality-viewpoint IDs across policies; record scheduling-dependent differences. FIFO creation order and LRU least-recently-observed are different; proposal conflates them. unique_kfIDs are origin IDs, not last_seen or stable Gaussian IDs.

## Minimum measurements

| Measure | Why / contract |
| --- | --- |
| Gaussian count vs frame/iteration | Growth and removal effects; not a substitute for bytes. |
| CUDA allocated/reserved and running peaks per PID | Distinguish tensors, allocator retention and transient growth; parent-only logging misses backend. |
| Concurrent device/application memory observation | Include process copies/context; isolate unrelated GPU load, report observer limitations. Do not sum independent per-process peaks or double-count shared IPC allocations. |
| Completion/OOM and budget violations | A quality score from an incomplete run is not success. Record max violation/duration and phase. |
| Aligned ATE RMSE | Existing evaluator uses keyframes; freeze evaluation pose IDs when selection changes to enable fair comparison. |
| Online PSNR/SSIM, LPIPS if feasible | Freeze non-keyframe viewpoint IDs; existing rendering couples map quality to pose error. Existing LPIPS is pretrained evaluation, not policy learning. |
| Mapping update time + policy time | Overall FPS includes throttle/waits/initialization; cannot replace mapping throughput. Separate CPU wall timing and GPU timing without per-iteration synchronization overhead. |
| Lightweight CPU RSS and retained keyframes | Diagnose whether count control leaves image/snapshot memory growing, especially unified-memory targets. Not an independent primary research axis initially. |

Sample telemetry in initialization, keyframe insertion, map update, densification/pruning, and synchronization. Reset/capture CUDA peak counters within defined run phases; boundary samples alone miss interior peaks. Distinguish live/online phase from export, LPIPS and post-run refinement. No need for RPE, disk latency, reload latency, power sweeps or feature metrics before corresponding mechanisms/targets exist.

Existing outputs to recover: resolved config.yml, plot/{trj_final,stats_final}.json, point_cloud/final/point_cloud.ply; rendering results in psnr/{before_opt,after_opt}/final_result.json; console/W&B FPS table. Preserve online map snapshot separately because final save path can be overwritten by refinement.

## Policy progression and invariants

1. Controller (Devon): explicit measurement scope and high/low watermark, reserve headroom before insertion/densification/synchronization, apply one ranking interface, log reasons and bytes/count before/after. A count target is acceptable for smoke testing, not a hard byte-budget claim.
2. Small baselines: budgeted seeded random, opacity ranking, and accurately named FIFO creation-age or LRU last-seen. Use same controller/deletion operator, not different opacity thresholds in each run.
3. Candidate (Phillip): cheap recency/window visibility/contribution proxy + coverage safeguard + infrequent expensive reevaluation. Visibility and opacity are proxies, not actual future tracking utility. No loop-closure event implementation exists; start with keyframe/periodic events.
4. Ablate each score signal, coverage constraint and trigger cadence; evaluate trajectories with revisits before claims about retaining future utility.

Model mutation stays backend-owned. New per-Gaussian arrays must be initialized/appended/copied in extend/densification, inherited or reset deliberately for split/clone children, and filtered by prune_points. Update/remap all visibility caches before sync. Enforce minimum viable nonempty map; renderer returns None for an empty map. Pruning itself allocates compact tensors/moments, so controller requires working headroom.

No learning required by proposal. Adaptive statistics can be deterministic. A small predictor is optional only after equal-budget heuristic comparisons, a meaningful target (e.g. loss/localization impact on future observations), independent train/validation scenes, and accounted policy training/inference costs. Repeated from-scratch training and RL are premature; long-horizon reward/credit assignment is unestablished.

## Novelty assessment and targeted reading

- LIKELY BASELINE / WELL EXPLORED: opacity/scale thresholds, basic visibility/contribution ranking, generic importance scores, FIFO/LRU/random.
- LIKELY ENGINEERING: wiring callbacks/config/logging, CPU/GPU copies and caches, disk storage, chunking/paging, hardware port. A new controller with demonstrable guarantees may support a systems contribution; implementation alone does not establish novelty.
- POTENTIALLY NOVEL, currently UNPROVEN: online utility estimates that protect future localization/revisits, explicitly account for insertion/densification/snapshot allocation peaks, and yield a better matched-budget trade-off with low overhead.
- NOVELTY UNCLEAR: normalized hybrid + coverage, adaptive hardware-aware policy, learned prediction. Combining familiar scores or running on Jetson is insufficient.
- Compression/merging and learned pruning already have substantial prior art; they add confounds and are not required to start this project.

Targeted primary-source checks during audit (not an exhaustive review):

- [DiskChunGS](https://arxiv.org/abs/2511.23030): out-of-core spatial chunks, disk-backed inactive regions, Jetson evaluation; GPU-memory scaling/edge deployment are established themes.
- [MemGS](https://arxiv.org/abs/2509.13536): voxel-space geometric merging in SLAM; distinguish it from deleting sparse points or keeping an arbitrary voxel representative.
- [PUP 3D-GS](https://arxiv.org/abs/2406.10219), [SafeguardGS](https://arxiv.org/abs/2405.17793): static sensitivity and pixel-aware pruning; a rendering score is not automatically new.
- [Pocket-SLAM](https://arxiv.org/html/2606.24796v1), June 2026: contribution-based online pruning plus image-tile survival budgets. Direct overlap with contribution/coverage hybrid; compare before claiming novelty. Its stated global target is Gaussian count, which does not itself prove a whole-application byte cap.
- [LP-3DGS](https://arxiv.org/abs/2405.18784): learned masks for pruning already exist; a learned score alone is not novelty.

Next bounded literature task: compare budget definition, peak-accounting scope, retention signal, spatial safeguard, revisit behavior and policy cost in Pocket-SLAM/DiskChunGS/MemGS; search `Gaussian SLAM online pruning recency coverage strict VRAM budget revisit tracking utility` and `Gaussian SLAM learned retention policy budget`. Inspect whether any method anticipates peak allocations and protects future localization at the same application byte budget. Do not claim absence of such work from this limited search.

## Minimum sequence / acceptance gates

| Phase | Objective / relevant files / expected change | Output and success | Complexity / Codex |
| --- | --- | --- | --- |
| 0 | Recover Devon run and ownership; validate data/environment and upstream behavior. environment files, configs, slam.py. Only necessary runner/config adjustments. | Manifest + successful baseline smoke; no claim of reproduction from syntax checks. | Small if existing setup; otherwise medium / Sol Medium. |
| 1 | Passive telemetry and full reference. BackEnd.initialize_map/add_next_kf/map/push_to_frontend, SLAM process setup, eval_utils. Add local machine-readable records without changing method. | Complete office0 trace + quality outputs, accounted peaks, map/camera/snapshot growth separated; overhead checked. | Medium / Sol Medium. |
| 2 | Devon's controller + ranking interface and simplest random/opacity/age baselines. backend, project_utils, GaussianModel mutation. | Comparable full budgeted runs, count/state alignment, headroom/budget violations recorded; smoke tests for empty/tiny/split maps. | Medium / 6.1 Sol High design, Sol Medium coding. |
| 3 | Benchmark simple policies on fixed small Replica/TUM subset. runner/configs/eval_utils. | Equal-budget quality/memory/overhead curves with completion status; confirm budgets are feasible. | Medium / Sol Medium; Luna/cheap for established plotting. |
| 4 | One differentiated candidate after literature comparison. backend metadata/project_utils. | Candidate demonstrably beats strongest simple comparator or useful negative result; no superiority assumed. | Medium–large / 6.1 Sol High design, Sol Medium implementation. |
| 5 | Ablations and repeated seeds/revisit sequences. configs/runner/analysis. | Contribution of each signal and constraint, variability reported, frozen evaluation IDs. | Medium / Sol Medium; Luna/cheap for mechanical run summaries. |
| 6 | Resource/Jetson evaluation if confirmed and accessible. environment/deployment/telemetry configs. | Validated CUDA extensions, unified-memory accounting and latency on named hardware; power only if a claim requires it. | Large setup / 6.1 Sol High initial diagnosis, Sol Medium execution. |
| 7 | Paper-quality experiments after novelty and feasibility gates. runner/results manifest/docs. | Broader matched comparisons and failure cases, reproducible tables/figures, defensible limited claims. | Large / 6.1 Sol High research synthesis, Sol Medium experiments. |

Start implementation only on baseline plumbing/measurement. Do not start research-policy implementation until phase 1 and controller ownership/interface are resolved.
