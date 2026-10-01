# Project context

Source snapshot audited: `ae8fce4b45747c94cee8314c885b5bba5e39f0e9`, 2026-09-30. Source can supersede this document.
Labels: CONFIRMED = source/history/results evidence; INFERENCE = reasoned interpretation; PLANNED = proposed, absent from execution; OPEN QUESTION = unresolved.

## Objective and ownership

- PLANNED objective: bound online MonoGS GPU-memory use while retaining localization and rendering quality; compare policies at matched budgets (proposal.md:16–66).
- CONFIRMED documented assignment, not confirmation of today's working agreement: Devon Goshorn owns budget/queue/instrumentation/experiment infrastructure; Phillip Minghao Li owns pruning heuristics, ablations, trade-off analysis, and results (Project_Proposal/proposal.md:9–10; newer PDF p.1).
- OPEN QUESTION: division of baseline reproduction and any changes to that agreement. Devon has already added preliminary pruning helpers, so coordinate rather than duplicate them.
- Original September 8 PDF identifies this as a CSCE 585 ML Systems project. Publication novelty is a further goal provided by Phillip; not a demonstrated repository outcome.

## Repository and provenance

- Root: `/home/phillubt/projects/3dGS`; origin: `https://github.com/DevGosho160/Memory-Management-for-3D-Gaussian-Splatting-SLAM.git`.
- Audit began on clean `main`; origin/main and live remote HEAD matched `ae8fce4`; only remote branch main existed. Proposal tags: proposal-v1=`8ad691c`, proposal-v2=`36c1fed`.
- Executable root is `MonoGS_System/MonoGS/`, not the repository root. Relative configuration inheritance and dataset paths require running there.
- Underlying system: [muskie82/MonoGS](https://github.com/muskie82/MonoGS), Gaussian Splatting SLAM, with Graphdeco Gaussian optimization and a camera-pose-differentiable rasterizer.
- Read-only upstream Git blob comparison against MonoGS `6c9254c319d8bff5caeef65259e6bb0941a9b9f6` matched 69 non-submodule imported files exactly, including mapper, tracker, Gaussian model, renderer, evaluator, and RGB-D configs. This identifies matching provenance; the team did not record an explicit upstream pin.
- Import differences: `.gitignore` Windows artifacts, monocular TUM GUI disabled, updated TUM download URLs, added `DevGosho_environment.yml`. Vendor submodules were not exhaustively hash-verified.
- Upstream gitlinks at that revision: pose rasterizer `43e21bff91cd24986ee3dd52fe0bb06952e50ec7`, simple-knn `44f764299fa305faf6ec5ebd99939e0508331503`. Team checkout vendors source instead of retaining these root gitlinks; do not claim exact vendor revisions without comparison.

## Team history

| Commit | Author metadata / actual change |
| --- | --- |
| `66f7d9f` | Devon: uploaded proposal presentation. |
| `2abd26d` through `986c5ad` | Devon: created/revised Markdown proposal. |
| `8ad691c`, `36c1fed` | Phillip identities: original PDF, then revised PDF and Markdown proposal; no implementation commits. |
| `5c2de7a` | Devon: imported MonoGS and environment/config/download adjustments. Commit message reports installing/running on RTX 5060 Ti; no saved run evidence in checkout. |
| `ae8fce4` | Devon: `project_utils/{memory_limit,map_pruning,__init__}.py` and memory-limit call in `slam.py`; mapper/model unchanged. Message says semantic pruning, but code has no semantic labels. |

## Architecture and Gaussian lifecycle

Paths below are relative to `MonoGS_System/MonoGS/`.

| Stage | Evidence / data / ownership |
| --- | --- |
| Input | `utils/dataset.py:257–278`: load RGB to CUDA, depth as CPU NumPy, pose to CUDA. `Camera.init_from_dataset` retains them. |
| Track / select keyframe | `utils/slam_frontend.py:128–196,198–286`: render map; optimize camera rotation/translation deltas and exposure; choose keyframes using translation/covisibility. First pose uses ground truth at :120. |
| Initialize / insert | FrontEnd.initialize/add_new_keyframe/request_keyframe -> queue -> `BackEnd.run:395–417`, `add_next_kf:67` -> `GaussianModel.extend_from_pcd_seq:235`. |
| Create | `gaussian_model.py:107–203`: RGB-D point cloud on CPU/Open3D, random downsampling; CUDA xyz/SH/scales/quaternions/logit opacity. RGB-D uses observed depth; monocular uses seeded depth from noisy initial/rendered estimates, not a depth network. |
| Append | `extend_from_pcd:208`, `densification_postfix:557`, `cat_tensors_to_optimizer:523`: concatenate trainable CUDA rows and Adam moments. CPU int32 `unique_kfIDs` and `n_obs` remain row-aligned. |
| Optimize | `BackEnd.initialize_map:86`, `map:142`; Gaussian Adam groups from `training_setup:245`; local camera poses/exposure also optimized. |
| Densify | `densify_and_clone:643`, `densify_and_split:593`, `densify_and_prune:674`: gradient/scale-based cloning and splitting; split parents deleted. Children inherit origin keyframe IDs and n_obs. |
| Observe | Renderer returns CUDA `radii`, `visibility_filter`, `n_touched`; backend derives window visibility and CPU `n_obs`. `n_obs` is recomputed over current window, not lifetime frequency. |
| Remove | `prune_points:505` filters all parameter rows, Adam moments, CUDA stats, CPU metadata. Permanent deletion, no archive/return. Reset removes all map rows. |
| Synchronize | `BackEnd.push_to_frontend:355` deep-copies GaussianModel through `clone_obj`; frontend receives a detached map snapshot. Copies retain optimizer objects/state; ownership and IPC allocation need measurement. |
| Export | `save_ply:326` and `load_ply:377` export/load scene parameters. No automatic offload/reactivation or optimizer/usage-preserving checkpoint. |

CONFIRMED: a CUDA-resident dense Gaussian map; no resident/inactive map pool, CPU/disk cache, paging, chunks, reactivation, or separate learned memory policy. The camera optimization window is not a Gaussian residency window. Existing masks express visibility, valid pixels, or permanent deletion, not memory eviction.

## Existing maintenance and memory limits

- Initialization periodically calls densify_and_prune with opacity threshold; mapping calls it when iteration_count % gaussian_update_every == gaussian_update_offset, with opacity and size thresholds (`slam_backend.py:122–128,286–296`). Replica defaults: init threshold 0.005, mapping threshold 0.7, update interval 150/offset 50, screen radius 20; world scale threshold is 0.1 * extent when size filtering enabled.
- Full-window covisibility pass (`slam_backend.py:244–275`): odometry removes n_obs<3; slam removes n_obs<=3 among recent origin keyframes; actual deletion is gated on monocular. RGB-D retains opacity/size pruning but not this covisibility deletion.
- Opacity reset is optimization maintenance, not deletion; nonvisible reset uses 0.4 and can affect later threshold deletion.
- `project_utils/map_pruning.py` defines opacity, max-axis-scale (named volume), window-visibility and density deletion helpers; no caller/import connects them to SLAM. Voxel-grid is a pass stub. Density mask is a NumPy ndarray and `.cuda()` at :60 fails; reproduced with real NumPy and a stub KDTree, without CUDA. It removes sparse points, not a merge of redundant dense points.
- `SLAM.__init__:28–30` calls memory limiter with defaults of 64.0 GiB GPU and CPU. CLI exposes only config/eval; no YAML budget integration. Passing None programmatically disables limits.
- GPU helper sets PyTorch per-process allocator fraction; limits at/above total device capacity are skipped. Set in parent before spawning backend/GUI, not explicitly initialized in children. No proactive budget controller or OOM handling.
- Linux CPU helper uses RLIMIT_AS (virtual address space, not physical RSS); Windows uses per-process Job Object memory. Neither is an aggregate SLAM memory policy.
- `render(mask=...)` is currently unsafe: wrapper at renderer :117 unpacks four values; rasterizer returns five at submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py:105; n_touched is unset on masked branch. No current SLAM caller supplies a render mask. A render mask also leaves full CUDA storage allocated.

## Datasets, evaluation, hardware

- Supported loader: TUM RGB-D (mono/depth), Replica RGB-D, experimental EuRoC stereo, live RealSense (`dataset.py:522`). No KITTI loader.
- Configs: TUM fr1_desk/fr2_xyz/fr3_office; Replica room0–2/office0–4 with _sp variants; EuRoC mh02. Proposal's selected targets: Replica + TUM, not a committed scene selection.
- Replica expects NICE-SLAM-style `results/frame*.jpg`, `results/depth*.png`, and traj.txt; download script points to that archive, not raw Replica assets.
- `_sp` is scheduling coordination: still a spawned backend process. Frontend reads Training.single_thread; backend reads Dataset.single_thread. office0_sp has both true; office0 has frontend true/backend false.
- Evaluator: aligned translation ATE on selected keyframes, monocular scale correction; PSNR/SSIM/LPIPS on every fifth non-keyframe at estimated poses; overall elapsed FPS. Not RPE, geometric reconstruction error, mapping-only throughput, or pure render FPS.
- --eval forces headless/save/rendering/W&B and invokes 26,000 post-run refinement iterations. Treat before_opt rendering as online metric; after_opt is separate refinement. Refinement steps all Gaussian Adam groups despite its color name. Do not use --eval for an inexpensive smoke test.
- No saved experiment outputs, datasets, trained checkpoints, custom tests, memory traces, or sweep runner found in this checkout. Existing media are upstream demonstrations. This does not disprove runs elsewhere; datasets/results are ignored.
- Original PDF (proposal-v1, Sept 8, p.1) and presentation slide 4 explicitly target NVIDIA Jetson AGX Orin and 4/8 GB. Newer PDF/Markdown proposal-v2 (Sept 11) omits a device and uses 25/50/75% peak budgets. CONFIRMED historical target; OPEN QUESTION current commitment/access. No Jetson-specific implementation/results.
- Hardware observed during audit: WSL-visible RTX 5070, 12,227 MiB, driver 616.56. System Python3 lacked Torch/NumPy/SciPy; no obvious repo environment/dataset. No environment installed. Devon's 5060 Ti environment is Python3.10/Torch2.1.2/CUDA11.8; runtime/extension compatibility on the chosen machine remains unverified.
- 2026-09-30 update: a hashed public 100-frame Replica office0 slice now exists in the ignored `datasets/replica/office0_slice100/` directory, with a committed download script/config in progress on `phillip/pruning-baseline`. A local ignored Python 3.14/Torch 2.12.1+cu132 environment built both CUDA extensions for sm_120; one in-process first-frame initialization iteration completed. Standard spawned execution still fails before mapping at CUDA tensor IPC on this host, reproduced with a standalone tensor. No completed online baseline or pruning result exists yet. See `CODEX_HANDOFF.md` for commands and status.
- 2026-10-01 update: `phillip/pruning-baseline` has a local checkpoint at `febf3ce`. An opt-in CPU-serialized frontend/backend queue bypassed CUDA IPC on this WSL host; both existing MonoGS and a deterministic extra opacity-pruning policy completed the same 100-frame office0 slice at seed 0. The policy calls `GaussianModel.prune_points()` and keeps visibility/metadata aligned; voxel pruning remains a stub. A/B have 55,365/54,989 final Gaussian rows and backend peak CUDA allocated 1,149,759,488/1,148,273,152 bytes, respectively; peak reserved is unchanged at 1,302,331,392 bytes. The workaround affects CPU/GPU transfer timing, CPU RAM, FPS and cross-process GPU memory, so these are local comparison measurements, not final system-level benchmark claims. Full commands, evaluation metrics, trace paths, and limitations are in `CODEX_HANDOFF.md`.
- 2026-10-01 current research milestone: frozen revision `c89f2af` completed an upstream-maintenance reference plus five retention/ablation runs at seed 0 on this slice. All five obey a strict 40,000 staged/live backend-map row ceiling. Tracking-support V1 fixed-frame ATE is 1.0083 mm versus deterministic random 0.8811 mm and no-T 1.0371 mm; superiority and quality/overhead gates are not achieved. Offline diagnosis confirms small support-floor fractions, different initialized maps/keyframe schedules, greater final V1 row concentration, and avoidable CPU controller/feedback overhead. Historical per-row T/visibility/ranking snapshots are absent, so independent tracking information and the causal reason random wins remain OPEN. This short local trajectory weakly tests persistent/revisit utility. See `RETENTION_V1_DIAGNOSIS_2026-10-01.md`; no V2 was implemented during diagnosis.
- Jetson CPU/GPU share physical DRAM: moving data to CPU need not reduce total system memory. [NVIDIA memory documentation](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).
