# First research method: preemptive retention with tracking-history support

Design only; no implementation or new SLAM runs in this session. Written 2026-10-01 against current clean revision `df425c85f34561ec4330148d18f9a25c59c39b2e`. This design supersedes the old research-plan instruction to remain in baseline plumbing for this user-authorized session. Phillip owns the retention hypothesis, heuristics, ablations and evaluation; the small controller contract below is the interface to coordinate with Devon, not a reassignment of all infrastructure.

Evidence labels: **CONFIRMED** means current code or saved artifacts; **INFERENCE** means an interpretation; **PLANNED** means this specification; **OPEN** identifies an unresolved measurement or hypothesis. Defaults below are preregistered pilot choices, not experimentally optimized constants.

## 1. What the saved A/B pair establishes

Artifact root, relative to the executable root `MonoGS_System/MonoGS`:

- A: `results/replica_office0_slice100/2026-10-01-01-01-20/`.
- B: `results/replica_office0_slice100/2026-10-01-01-06-28/`.
- Existing summary: `results/replica_office0_slice100/comparison_final_2026-10-01/`.

Both completion files report 100 completed frames; backend and frontend final maps agree. Both use seed 0, the same CPU queue serialization, SH degree 0 and online evaluation without refinement. Their manifests record base revision `febf3ce` plus a tracked-diff hash; do not describe these runs as a clean run of current `df425c8` or of `febf3ce` alone.

| Quantity | A | B |
|---|---:|---:|
| Final rows | 55,365 | 54,989 |
| Maximum sampled rows, before ordinary maintenance | 65,583 | 65,286 |
| Total rows inserted at 19 keyframes | 254,584 | 254,584 |
| Initialization densify/prune **net** additions | 1,089 | 1,088 |
| Mapping densify/prune **net** deletions | 200,308 | 194,287 |
| Extra opacity deletions, 18 events | 0 | 6,396 |
| Backend peak allocated bytes | 1,149,759,488 | 1,148,273,152 |
| Backend peak reserved bytes | 1,302,331,392 | 1,302,331,392 |

The exact count ledger is:

```
A: 254584 + 1089 - 200308        = 55365
B: 254584 + 1088 - 194287 - 6396 = 54989
B - A = -1 + 6021 - 6396        = -376
```

**CONFIRMED:** B's ordinary mapping maintenance has 6,021 fewer net deletions, offsetting most extra removals. Every later keyframe still inserts about 12,726–12,731 rows; extra pruning never restricts admission. B is sometimes larger even after extra pruning: frame 4 ends at 30,055 versus A's 29,829; frame 8 at 33,679 versus 33,441. The final count difference is therefore not a running sum of extra deletions.

**Important correction to the causal hypothesis:** insertion replenishment occurs in both runs, but insertion totals are identical. The saved events combine cloning, splitting, split-parent deletion and opacity/size deletion. They do **not** distinguish more densification from fewer ordinary deletions, identify the same Gaussian across runs, or prove that a particular removed row was recreated. The offset is confirmed; its gross lifecycle decomposition is OPEN. Different trajectories, masks, floating-point execution and the keyframe 52/53 substitution can change optimization and maintenance outcomes. Initialization differs by one row before extra pruning even starts, so a fixed seed has not established bitwise CUDA reproducibility.

Extra opacity deletion is also small per event: 6,396 / summed pre-prune candidate counts = 0.894%. Its threshold is 0.5; admission initializes opacity at 0.5; mapping already uses 0.7 plus size pruning. The extra hook executes after keyframe mapping and its final maintenance pass. It selects a small remaining tail and arrives after the expensive work.

**No inference of ATE improvement:** retained IDs differ at A=52/B=53. Reanalysis of the saved trajectories on their 18 common IDs, with independent rigid alignment and no scale correction, gives A=0.000754943927 m, B=0.000720577007 m. This computation is analysis of saved poses, not another SLAM run. Both rendering evaluations use the same 17 viewpoint IDs. One seed and submillimeter errors are insufficient to establish superiority.

## 2. Peak timing and its measurement limits

| Frame 93 event | A | B |
|---|---:|---:|
| Insertion completes, iteration 3466 | 49,109 → 61,836 | 48,919 → 61,646 |
| Allocated immediately after insertion | 432,138,240 B | 432,069,632 B |
| First row recording final allocated peak | map iteration **3490** | map iteration **3478** |
| Next upstream densify/prune, iteration 3500 | 61,836 → 52,857 | 61,646 → 53,050 |
| Extra pruning, iteration 3617 | absent | 53,050 → 52,560 |
| Sync begins, iteration 3617 | 285,397,504 B | 282,470,912 B |
| Sync enqueued | 297,883,648 B | 294,888,448 B |

**CONFIRMED:** both global allocated peaks occur during ordinary mapping after admission and before iteration 3500's maintenance, before extra pruning and before synchronization. A's peak is about 23 mapping iterations after insertion; B's about 11. The counter is cumulative, so the final peak never decreases after later pruning. Frame 99 has the largest sampled count but does not set a new byte peak. Reserved memory first reaches its final maximum during frame 93 mapping at iteration 3468 in A / 3469 in B.

The sampler records boundaries after backward and optimization. It can identify this peak's **mapping iteration interval**, not whether the instantaneous maximum was inside a particular camera rasterization, backward, Adam or an overlapping live tensor lifetime. A's maximum sampled live allocation is only 493,755,904 B, substantially below the 1,149,759,488 B running peak. Rendering several window views plus up to two older views accumulates autograd/rasterizer buffers inside each mapping iteration. This is the leading code-grounded explanation for the transient peak, not a measured per-kernel attribution. Synchronization does allocate copies, but it does not establish the global peak in either saved run.

Pruning row tensors cannot remove all other memory: image/depth/loss buffers, camera data, rasterization workspaces, visibility arrays and model snapshots matter. The same allocator can retain freed blocks in its reserved pool; B's unchanged reserved high-water mark is not evidence that `prune_points` failed. Neither backend allocator metric is application/device-wide memory. The CPU-transfer workaround further prevents system-wide cross-process conclusions.

## 3. Research question and claim boundary

**PLANNED primary pilot question:** With identical preemptive admission control and the same maximum live Gaussian count, does retaining Gaussians with persistent recent tracking visibility and multi-view support yield lower fixed-frame ATE than opacity, random and last-seen retention, at acceptable online rendering quality and policy cost?

**Secondary question:** At an identical measured backend PyTorch allocated-byte ceiling, does the same ranking give a better localization/quality trade-off while completing without observed peak violations?

Call the score a **tracking-support proxy**, not expected Fisher information, actual localization sensitivity or a prediction of loop closure. Rendering visibility may occur at pixels excluded by the tracking gradient/depth masks. The proposed temporal hypothesis is that repeated recent tracking visibility is more useful for near-future localization than instantaneous opacity or one last-seen timestamp. This must be tested; the 100-frame slice cannot establish revisit retention or long-horizon behavior.

## 4. First controller: two explicit contracts

### 4.1 Strict row/state contract

Use a hard live-row ceiling `K` for the mutable backend map. It applies **during append and split staging**, not just after maintenance. Snapshot duplicates are accounted in the byte guard, not treated as additional distinct map Gaussians under K. Define `L=floor(0.95*K)` as a reclamation target. If a planned operation with gross additions `G` would exceed `K`, retain at most `L-G` existing rows before allocating children/new row tensors. If that target conflicts with support floors, reduce admission first; report infeasibility if the existing protected map itself cannot fit. Do not exceed the ceiling to honor an unlimited grace period.

Track exact parameter/Adam/statistic bytes too, using tensor shapes/dtypes and unique storages rather than a universal bytes-per-row constant. Here, SH0/nonisotropic FP32 parameters contain 14 scalars = 56 B/row; two Adam moments add 112 B; a full parameter gradient adds 56 B; the three densification/radius statistics add 12 B. Thus resident state is approximately 180 B/row without gradients, 236 with them, excluding cameras, visibility, policy metadata and transient copies. SH3 is different. A row cap with constant formats is an explicit map-state budget; it does not prove a process-memory budget.

### 4.2 Backend allocated-byte guard

Let `B` denote an **online backend PyTorch allocated-byte ceiling**. At each allocation-heavy phase, predict:

```
predicted_peak = measured_live_before_phase
               + upper_estimate_of_additional_live_bytes(phase, N, G)
               + safety_margin
```

Use the maximum requirement among candidate upload/initialization, pruning/compaction, append, mapping, clone/split and snapshot copy. Do not add mutually exclusive phase peaks as if they were simultaneous; do include buffers genuinely live through a mutation. Byte-limit enforcement is empirical admission control with measured compliance, not a mathematical CUDA/whole-device guarantee.

For the same 100-frame pilot, initialize the mapping increment estimate from A's trace:

```
H_map(N) = ceil(1.10 * 14224 * N) + 16 MiB
```

14,224 B/row is a rounded maximum of `(cycle-end cumulative allocated peak - post-insertion live allocated)/post-insertion N` over A's keyframe cycles, including its full window. The maximum is frame 62, approximately 14,223.6. Cumulative peaks upper-bound individual cycle peaks; this is intentionally conservative **within the observed configuration**. Linear scaling with N, different Gaussian sizes/coverage, new trajectories and budgets is an INFERENCE, not a proven upper bound. Keep the full-window coefficient even during early initialization for simplicity. Do not extrapolate this calibration to another resolution/SH degree/window or scene without labeling it uncalibrated.

For compaction/append/sync, compute bounds from the actual tensor ledger; allow old and new parameters/moments to coexist. A simple conservative initial append/prune allowance is two full target row-state copies plus candidate tensors and the same 16 MiB margin. For `clone_obj`, account for deepcopy of optimizer/parameters and the additional detached top-level clones. Candidate CUDA preparation includes KNN and features; use a separate conservative ledger allowance, not only final candidate parameter bytes. These estimates must be logged and checked against phase peaks.

At an event, if the projected map does not fit B, solve for a smaller row target using `H_map` plus measured resident row bytes, compact once, **remeasure live allocation**, and recheck. Do not assume old parameter graphs immediately release after a prune. At most two corrective passes; then reduce incoming growth or mark the run `budget_infeasible`. Check before each mapping iteration too, including `prune=True`, as other live allocations change. If support floors cannot fit the predicted phase, stop with a recorded infeasibility rather than changing resolution/window/mapping iterations for one policy.

Initialization and every synchronization require the same checks. Reset/export/evaluation are logged as separate phases; exclude frontend LPIPS evaluation from the online backend ceiling, and report the scope explicitly.

Measure actual phase peaks. If actual allocated peak exceeds B, record a violation and stop/mark budget failure; post-hoc pruning cannot erase it. Keep a separate host-side running maximum if phase peak counters are reset. Update the phase envelope upward after any underprediction, identically for all rankings. The first model can over-reject feasible budgets; its conservative failures are a controller result, not an algorithm-quality conclusion.

Reserved memory is a diagnostic, not the control variable: the caching allocator need not shrink reserved blocks. Do not call `empty_cache()` every policy event or count its apparent savings as retention. No claim of device/application compliance follows from B.

## 5. Forecast growth without learning

### Insertion

Refactor point-cloud preparation into a CPU stage and the existing CUDA parameter stage. Preserve the existing Open3D projection and seeded downsampling. Obtain the **actual** candidate count `G` from the sampled CPU point cloud before uploading xyz/features or invoking CUDA KNN. Existing raw-depth pixel count divided by 64 (32 for initialization) is a fallback upper planning estimate, not the final admission count.

Keep all original candidates when feasible. Otherwise select a smaller candidate subset **on CPU** using a policy-independent, deterministic spatial round-robin over 0.5 m world cells, with original candidate order as tie-break. This is admission throttling, not the proposed utility ranking; record generated/admitted/rejected counts. All comparators share it. Reserve room for accepted candidates before concatenating optimizer state. New Gaussians are not permanently exempt from budget enforcement.

### Densification

The existing gradient/scale masks give exact candidate counts before child allocation. Compute them once at the normal maintenance event after accumulating current gradients. For clone count C and split-parent count S with two children:

```
gross staged additions = C + 2*S
net post-split growth  = C + S
staged live rows       = N + C + 2*S
```

Budget against **gross staged additions**, because parents exist while children are appended. Do not subtract opacity/size deletions expected to happen later.

Limit admitted gross densification additions to `floor(0.05*K)` per event in v1, including initialization. Select candidate parent operations by densification gradient descending, stable ID ascending, admitting a split atomically only if both child slots fit. This admission rule is identical for all rankings. Temporarily protect admitted parents while retaining existing rows to `L-G` when required. Reduce growth if parent plus support protections cannot fit. Remap frozen parent IDs/masks through pre-pruning rather than recomputing gradients from reset statistics. Disable controller branch entirely for upstream reference.

An existing upstream maintenance event replaces parameters after backward, already dropping current gradients on mutated parameter rows. Put pre-retention at this same mutation event, after stats collection; do not add arbitrary after-backward pruning on non-maintenance iterations. Before ordinary mapping, prune only before building that iteration's graph. Preserve the original thresholds, opacity reset and ordinary deletion; protections apply to **extra budget retention**, not to overriding upstream validity/opacity rules.

## 6. Cheap signals and the first score

| Signal | Current availability | V1 decision |
|---|---|---|
| Opacity | `get_opacity`, CUDA | Use as a small reliability term |
| Origin keyframe | CPU `unique_kfIDs` | Use for support floors; not age or last-seen |
| Current-window observations | `occ_aware_visibility`; `n_obs` recomputed only on full-window maintenance | Derive W directly from aligned latest window masks; do not assume `n_obs` is cumulative |
| Renderer visibility | CUDA `radii>0`, `n_touched>0` | Reuse `n_touched>0` without another render |
| Renderer contribution | `n_touched` count exists; alpha-weighted contribution not exposed | Count is a proxy only; no new rasterizer work |
| Actual tracking visibility | Frontend final tracking render already returns `n_touched` | Accumulate per camera frame and return aggregate at next keyframe |
| Birth/age, last seen | Absent as policy metadata | Add row creation/lineage birth and tracking last-seen |
| Visibility EMA | Absent | Maintain cheap EMA over **distinct frontend frames**, not optimizer iterations |
| Lifetime visibility count | Absent | Optional debug counter; not required or used in score |
| Spatial coverage | XYZ exists; no integrated coverage policy | Recompute coarse cells at retention events |
| Tracking Jacobian/Fisher, leave-one-out loss, semantic utility, exact masked loss contribution | Not maintained | Exclude v1: expensive or requires renderer/gradient changes |

**Renderer caveat from current CUDA source:** `n_touched` is incremented only for accepted pixel contributions with `test_T > 0.5`. It is not all projected pixels or total alpha/transmittance-weighted contribution. Binary hits inherit this limitation.

Let f be the latest feedback frame; for Gaussian i:

```
v_i(f) = 1 if final tracking render n_touched_i > 0 else 0
beta = 2^(-1/20)                         # 20 camera-frame half-life
T_i <- beta*T_i + (1-beta)*v_i(f)        # tracking visibility EMA
W_i = number of current-window views seeing i / number of valid window views
R_i = 2^(-(f-last_seen_i)/20)             # 0 if never actually seen
O_i = sigmoid(opacity_logit_i)
U_i = 0.50*T_i + 0.25*W_i + 0.15*R_i + 0.10*O_i
```

Retain highest U, stable Gaussian ID ascending on ties. All terms are bounded in [0,1], with fixed interpretable scales; no fitted weights or policy-specific thresholds. NaN/Inf scores get lowest rank and a diagnostic flag. Age is not a direct score penalty; genuinely useful older map rows should survive.

The weights and half-life are hypotheses. T rewards persistent tracking support; W favors multi-view overlap; R avoids discarding newly useful regions while EMA catches up; opacity is a small reliability prior. W must be recomputed after window changes, including immediately before insertion using available aligned masks for surviving previous views. The new camera has no backend observation yet and is excluded from W's denominator until rendered. No union hit is counted hundreds of times because mapping uses 150 optimizer iterations.

New inserted rows start T=0, `last_seen=-1`, with a **soft** one-keyframe probation: use R=1 until their first completed keyframe cycle. Clone/split children inherit parental T/last_seen as a provisional prior; flags identify unobserved children. No extra additive lifetime evidence is created by copying. Probation expires after one keyframe event, independent of row creation frame, and is not renewed by repeated cloning. Keep this common to all rankings via shared admission/protection, not a hidden exemption. If a child has inherited real last-seen, label it inherited rather than an actual child observation.

## 7. Tracking feedback and stable identity

Required CPU row metadata in `GaussianModel`:

- `gaussian_id`: int64, unique monotonically allocated backend ID.
- `row_created_frame`: int32, actual row creation event.
- `lineage_birth_frame`: int32, original insertion frame, inherited by descendants.
- `last_seen_frame`: int32, -1 initially.
- `tracking_ema`: float32, 0 initially.
- `probation_until_kf_event`: int32, common short lifetime; descendants inherit the parent's expiry.
- `observed_since_creation`: bool, distinguishes an actual row observation from an inherited parental prior.
- Model scalars: `next_gaussian_id`, `map_version`, `kf_event_index`.

These are non-optimizer metadata and total about 29 B/row CPU, roughly 1.45 MB at 50k, excluding container overhead. `observed_since_creation` starts false on insertion/clone/split and becomes true only on a returned tracking hit for that exact ID. Store metadata in a named container with one append/inherit/filter operation to avoid scattering row logic. CPU policy sorting at insertion/maintenance events is acceptable at 40–50k; measure transfers/sorting cost. Do not sort every tracking iteration.

Frontend uses its received stable IDs and map version. Once per final tracking render, accumulate a vector `s <- beta*s + (1-beta)*v`, latest seen-frame vector and a count m of distinct camera frames. At the next keyframe request include IDs, version, m, s, last-seen. Backend joins IDs to currently surviving rows before insertion/retention and applies:

```
T_new = beta^m * T_old + s
last_seen_new = max(last_seen_old, received_last_seen)
```

This matches framewise EMA without queueing one message per camera or changing the renderer. A single CUDA-to-CPU transfer per feedback block suffices; vectors can accumulate on frontend CUDA. Flush/reset the block at feedback transmission; never double count it. Include block sequence and start/end frame IDs for validation. First version supports the existing `_sp` scheduling only: both frontend/backend single-thread flags must be true. Version mismatches are an explicit error in this mode; ID joining is still needed for deletions and future compatibility. Asynchronous stale-feedback semantics are outside v1.

Update statistics for all enabled comparator policies, even if their ranking does not use T, to share instrumentation/protection costs. Finish/final-frame feedback may be logged without another map mutation; do not introduce a final special retention step that changes only one policy.

## 8. Support floors

At a retention event construct a protected union P using a **policy-independent** selector:

1. For each current-window view, protect up to 128 currently visible rows, highest opacity then lowest stable ID. This protects actual observed keyframe support, not only origin labels.
2. For each represented origin keyframe, protect up to 64 rows, highest opacity then ID. These origin quotas are provenance support, not proof that the origin remains geometrically visible.
3. For every 0.5 m world cell with a currently observed row (W>0 or T>=0.2), protect up to two eligible rows, highest opacity then ID. Recompute cells from current XYZ only at policy events. Admission spatial round-robin gives incoming regions initial coverage; do not treat rejected candidates as occupied map support.

Select the union once; fill remaining target slots by the chosen ranking. Every comparator shares P except the named no-protection ablation. Record occupied/covered cell counts and actual per-view survivor counts. For a group smaller than its quota, retain all available eligible members. These floors guard against deleting **all** represented support; they do not guarantee pose observability or localization.

If |P| exceeds the required retained count, reduce incoming clone/split/admission requests first. Do not silently relax the floors, arbitrarily choose some cells, exceed K, or hide a failed feasibility condition. A existing protected map that cannot fit is `support_budget_infeasible`. Empty input cells/views need no synthetic support. Ordinary upstream deletion can still remove protected rows; refresh masks/floors at the next decision and distinguish its removals in telemetry.

## 9. Row lifecycle and optimizer invariants

| Operation | Required metadata/cache behavior |
|---|---|
| Insertion | Fresh IDs, both birth fields=f, origin existing kf_id, T=0, last_seen=-1, common probation expiry; append zero visibility for previously rendered views |
| Clone | Fresh child ID; row creation=f; inherit lineage birth, origin, tracking prior/last-seen and parent's probation expiry; no invented real observation |
| Split | Fresh IDs for both children; same inheritance; parent removed; order metadata exactly as existing `repeat(N,...)`, not `repeat_interleave` unless parameters also change |
| Prune | One boolean pre-prune keep mask filters every parameter, Adam moment, densification statistic, existing CPU metadata, new metadata and every cached visibility vector |
| Optimizer compaction | Retained moment values and scalar Adam step preserved; new rows have zero moments as upstream; metadata is not an optimizer group; param-group references match new model tensors |
| Snapshot | Copy IDs and metadata, retain version; frontend never changes backend-owned parameters or counters |
| Reset | Clear arrays/feedback/caches and event counters; IDs must not alias across a reset generation; version/generation identifies reset |
| PLY load/export | Existing PLY is geometry only; initialize missing metadata conservatively for a fresh run; refuse history-dependent resume unless a separate metadata checkpoint exists |

All mutation paths, including upstream clone/split/prune, emit a `RowTransform` describing surviving old rows and newly appended child/insert rows. Filter cached old visibility; appended rows initially have false visibility because inherited parent visibility is not a measured child observation. Backend's existing next mapping render refreshes it; ensure `map(prune=True)` refreshes before sync. Stable IDs make tracking history independent of mutable row index. Never retain stale boolean masks after compaction.

## 10. Exact implementation work for GPT-6 Sol Medium

All paths in this section are relative to `MonoGS_System/MonoGS/`. Implement this small method only; no learned score, reversible eviction, rasterizer modification, merging, masked render, keyframe storage architecture or hardware migration.

| File / functions | Changes |
|---|---|
| **New** `project_utils/retention_policy.py` | Pure deterministic score/rank functions; protection union; stable-ID tie breaks; comparator strategies and two ablation toggles |
| **New** `project_utils/budget_controller.py` | Row/byte contract; phase forecasts; admission plans; feasibility status; ledger accounting; no Gaussian mutation inside ranking functions |
| `gaussian_splatting/scene/gaussian_model.py`: `__init__`, `create_pcd_from_image`, `create_pcd_from_image_and_depth`, `extend_from_pcd_seq`, `extend_from_pcd` | CPU candidate preparation hook before CUDA upload; admission callback; metadata initialization; disabled branch preserves old path |
| Same: `densification_postfix`, `densify_and_clone`, `densify_and_split`, `densify_and_prune` | Frozen parent IDs/selected masks; explicit growth planning; gross staging cap; metadata inheritance; RowTransform return/callback, no children allocated before admission |
| Same: `prune_points`, `_prune_optimizer`, `cat_tensors_to_optimizer`, `load_ply` | Central metadata/caches alignment, ledger checks, preserve existing moment/scalar-step behavior; initialize load metadata |
| `utils/slam_backend.py`: `set_hyperparams`, `run`, `add_next_kf`, `initialize_map`, `map`, `push_to_frontend`, `reset` | Instantiate controller; consume feedback before mutation; checks before insertion/mapping/densification/sync; own masks/caches; isolated timing and structured events; retain existing post-map opacity option only for diagnostic reference |
| `utils/slam_frontend.py`: `tracking`, `run`, `request_keyframe`, `sync_backend`, `initialize` | Frame-level final-render accumulation, versioned feedback payload, fixed-frame pose recording; preserve final synchronization fix |
| `utils/run_telemetry.py`: `FIELDS`, `record` | Phase start/end live+local peak, global max, predictions/errors, admission counts, budget/protection counts, lifecycle component removals, policy wall/GPU timing |
| `utils/eval_utils.py`: `eval_ate`, `eval_rendering`; callers in frontend / `slam.py` | Optional explicit pose/render IDs; preserve default upstream evaluator; evaluate all 100 fixed final poses in research runs; preserve tracked pose-only scalars when Camera.clean removes images |
| `slam.py`: parser/config/manifest/online summary | Research policy config, reject simultaneous extra-threshold and budget ranking; config hash and feedback/evaluation/budget contract in manifest; budget failure is visible |
| **New** `configs/research/office0_retention_v1.yaml` | Inherit existing slice config; fixed `_sp` scheduling, policy/budget/ablation/evaluation settings |
| **New** `scripts/run_retention_pilot.py`, extend `scripts/summarize_ab.py` or new summarizer | Sequential small matrix, per-run config manifests, validate completion and compliance before comparing quality |
| **New** `tests/test_retention_policy.py`, `tests/test_gaussian_metadata.py` | Only algorithm/lifecycle/contract tests described below |

No change is needed in `project_utils/memory_limit.py`, `utils/cpu_transfer_queue.py` or CUDA rasterizer for the research method. CPU transport remains identical. Preserve the upstream reference with `Retention.enabled: false`; extra opacity threshold remains separately available. Do not reinterpret `n_obs` or `unique_kfIDs`.

**Pruning sequence:** consume latest feedback → prepare growth candidates → forecast row/byte demands → build P and rank existing rows → prune once if needed → apply RowTransform → remeasure → reduce admission if needed → allocate accepted candidates → assert staging ceiling → run normal mapping/maintenance → refresh visibility → check snapshot → synchronize. At densification, the plan occurs at its existing stats/mutation point and includes all temporary child rows. Between these events use only a cheap phase guard; ranking is triggered by forecasted pressure, not an arbitrary every-frame deletion schedule.

**Invariants and meaningful tests:**

- Every live parameter/statistic/metadata/cache length is N; IDs unique; all optimizer moment shapes agree with the corresponding parameter; same retained values and scalar Adam step after compaction.
- Synthetic three-row Adam test with insertion, clone, split, prune and reset; both uninitialized and populated optimizer states; verify split child order/history, deleted IDs never reappear and zero child moments.
- After all mutations, including temporary append, N<=K; clone=+1, two-child split uses two gross slots before -1 parent. Test a near-full map where using net growth would overflow.
- EMA aggregate equals explicit framewise update; never-seen last_seen and missing/deleted IDs; duplicate/incorrect feedback block/version rejected; mapping iteration count cannot inflate visibility evidence.
- Equal-score deterministic ordering, seeded random independent of global NumPy/Open3D/Torch RNG, nonfinite scores, all-low-opacity, tiny maps, zero candidate growth.
- Protection union satisfies each feasible group floor; |P|>target rejects growth/reports infeasibility; no empty map sent to renderer. Minimum viable map floor is P plus at least one row; count-only keep-one is not claimed sufficient for SLAM.
- Forecast recognizes prune/snapshot copies and **interior** phase peak; violation remains recorded even if later allocation drops. Phase peak resets preserve cumulative global maximum.
- Disabled path retains existing thresholds/schedules and default evaluator. Smoke test enabled path for ten frames locally, then run the declared pilot; do not demand another-machine reproduction.

## 11. Pilot budgets and comparators

Use the measured final 55,365 and maximum sampled 65,583, not the proposal's arbitrary 25/50/75% byte fractions:

| Strict live-row ceiling | Fraction of final baseline | Reduction from sampled maximum |
|---|---:|---:|
| 50,000 | 90.31% | 23.76% |
| 40,000 | 72.25% | 39.01% |
| 33,000, stretch only | 59.60% | 49.68% |

Both main ceilings exceed initialization's 25,461 candidates / ~26,550 initialized map. They should first apply meaningful pressure during later insertion, without making initialization deliberately infeasible. Floors/byte guards can still make them infeasible; do not assume success. Report both maximum live count and count at sync; they are different.

For the row-comparison stage, common backend guard `B=ceil(1.10*P_A)=1,264,735,437 B` (~1,206.15 MiB). It reserves a modest absolute ceiling while testing the row constraint; it is **not** a claim of 10% byte savings. Any byte-guard intervention is reported and can make realized counts differ below K. No superiority inference if some runs violate their common budget.

For the small byte-focused extension, keep K=50,000 and use common `B=floor(0.95*P_A)=1,092,271,513 B` (~1,041.67 MiB). Compare proposed method and the strongest simple comparator. Optional 90% `1,034,783,539 B` only after 95% completes feasibly. A row reduction cannot predict these byte savings: experiment decides. Never set 25%/50% of A's byte peak this week without showing non-map/scratch feasibility.

Policies:

- Upstream/no extra pruning: controller disabled, ordinary MonoGS maintenance retained. Saved A remains the original evidence; a new local reference with fixed evaluation/profiling serves the upcoming matrix.
- Existing extra opacity threshold 0.5: saved B is a diagnostic uncontrolled reference, not an equal-budget comparator.
- Budgeted opacity: highest opacity retained through the **same controller**, no new opacity threshold.
- Budgeted random: fixed SplitMix64 of `seed XOR stable_id` defines priority; no Python randomized `hash()`, no use of algorithm RNG, no newly sampled permutation at each event.
- Budgeted LRU: last-seen descending, lineage birth descending then ID for ties/never-seen. Name it last-seen/recency, not FIFO. FIFO creation-age can be a stretch comparator, not required.
- Proposed: U plus shared support P.

All enabled rankings use the same growth plan, byte model, metadata updates, floors, thresholds, camera/optimizer schedule and seed. Their Gaussian subsets can change trajectories/keyframes naturally; log IDs and work counts. Do not freeze keyframe admission just for one policy. Shared protection partially benefits simple comparators deliberately; the experiment isolates ranking conditional on a common viable support mechanism.

## 12. Minimum ablation and run count

Core seed-0 set: four budgeted rankings × two row ceilings = **8 runs**, plus one new local unbounded reference with fixed evaluation.

At K=40k, seed 0 add two ablations:

1. **No tracking-history score:** set T coefficient to zero; keep W/R/O coefficients unchanged. Do not renormalize or retune. Last-seen/R and the common T-based cell eligibility floor still exist; this isolates T's ranking contribution beyond recency, rather than removing all tracking-history use from the system.
2. **No support floors:** same U, no P; keep admission/controller/soft probation unchanged. This tests the safeguard independently of score.

Then two runs at the 95% backend ceiling: proposed vs strongest simple comparator (choose by seed-0 ATE among feasible runs, rendering as tie-break). Total **13 runs** including local reference. If time permits, repeat proposed and that comparator at K=40k seed 1 (plus seed-1 reference for degradation comparisons), reaching **16 runs**. These are preliminary ablations; no need for all weight/cadence/voxel-size sweeps, full literature implementations or all seeds × all budgets. The existing ~287 s per run implies roughly 1–1.3 hours of raw online runtime for 13–16 runs, but allow hours for validation, evaluation and debugging; overhead/quality failures can extend it.

If any budget is infeasible for all rankings, record that result and move to the next less constrained common budget; do not silently tune each policy. If tracking metadata delivery cannot be validated, stop research comparison until fixed; replacing T with backend visibility would test a different hypothesis.

## 13. Evaluation and pilot success gates

Freeze pose IDs `[0,...,99]` for final all-frame ATE with rigid alignment; save GT/estimated poses and IDs. Final keyframes use their final backend-refined poses, other frames their saved tracking poses; use this same rule in every run. Also retain conventional keyframe ATE as secondary. Do not use saved A's keyframe-only ATE as an all-frame reference. Freeze rendering IDs to A/B's actual shared non-keyframe IDs `[5,10,15,20,25,30,35,45,50,55,60,65,70,75,80,90,95]`; evaluate them even if a future run selects one as a keyframe, and label them fixed evaluation views rather than always held-out. Run online map only, no refinement. Estimated-pose rendering is primary; GT-pose rerendering is optional if map/pose confounding becomes material.

For each run record:

- Completion/frames and feasibility; **zero strict row violations**, including staged append/split counts.
- Exact state bytes and resident N min/mean/max/final, post-insert and sync traces; accepted/rejected candidates and every gross clone/split/upstream/budget deletion component.
- Measured online backend allocated peak, reserved peak, predicted versus actual local phase peaks, violation bytes/count/phases. Boundary samples cannot certify interior compliance; phase max counters cover allocator peaks.
- Fixed-frame ATE RMSE and per-frame errors, PSNR/SSIM/LPIPS at fixed IDs; lost/nonfinite tracking events and keyframe IDs.
- CPU ranking/feedback/compaction wall time and completed GPU event time where relevant, not only enqueue time. Queue transfer/overall FPS remains a workaround measurement. Measure total **added** policy time, median/p95 per event, and fraction of comparable backend+frontend online work; instrument every enabled comparator alike.

Preregister practical pilot quality gates against the new same-seed all-frame unbounded reference: ATE <= max(1.25*ATE_reference, ATE_reference + 0.0002 m), PSNR drop <=1 dB, SSIM drop <=0.01, LPIPS increase <=0.01. These are pilot tolerance choices tailored to the unusually easy submillimeter slice, not general SLAM standards. Added policy overhead target <=5% of online compute time, and report p95 event cost; do not use end-to-end FPS alone.

**Controller success:** every successful comparable run honors K and its backend B without observed allocator peak violations, and all frames complete. A violated/infeasible run cannot earn a quality win.

**Research signal:** proposed method beats the strongest feasible simple heuristic at the same budget in fixed-frame ATE while meeting the quality/overhead gates; a >=10% ATE reduction is a useful effect-size target, not a statistical claim. Check consistency at both budgets or on the extra paired seed. No-T ablation should worsen localization if persistent tracking history contributes; no-P ablation should reveal coverage/support loss or quality failure. If all policies have nearly identical ATE, if simple LRU wins, or if ablations do not matter, report a negative/underdetermined pilot. A 100-frame easy trajectory may be insufficient to discriminate utility even with correct control.

## 14. Novelty risk, primary-source comparison

- **Pocket-SLAM** already uses online rendering-area ranking, tracking-gradient tile budgets, regional survival constraints and evaluates localization. Its paper ranks after mapping; newly added Gaussians are exempt until tracking allocation, and its explicit global target is Gaussian count. Thus contribution/coverage/“tracking-aware” language and a weighted visibility score are high-overlap ideas. This v1 uses temporal tracking-hit history, actual window-observation/origin/world-cell support and pre-allocation gross-growth checks; those are testable differences from the described method, not novelty proof. [Paper, Sections III-B–C](https://arxiv.org/html/2606.24796v1).
- **DiskChunGS** already anticipates loading, uses spatial chunks, LRU eviction and a configurable Gaussian residency target before new chunks are loaded; it also manages keyframe memory. Preemptive admission, recency and spatial organization are therefore known. Its reversible disk residency and ORB-SLAM3 pose architecture differ from this permanent row retention in coupled MonoGS; GPU-memory/Jetson positioning is not a new contribution. [Paper, Optimization / LRU](https://arxiv.org/html/2511.23030v1).
- **MemGS** already addresses Gaussian redundancy in SLAM through voxel-based geometric merging, including custom CUDA pairwise similarity machinery and initialization sampling. A coarse-cell deletion floor is a different operation, but voxel coverage and redundancy reduction are established themes. This v1 avoids merging and does not claim equivalent reconstruction preservation. [Paper, Section II-C](https://arxiv.org/html/2509.13536v1).

Likely baseline/engineering: count ceilings, allocator measurement, headroom/watermarks, opacity/LRU/random, visibility EMA, stable metadata, coverage floors and callback wiring. **Differentiated hypothesis, unproven:** at matched feasible budgets, temporally persistent actual frontend tracking support identifies retention worth preserving beyond recency/window visibility/opacity, and makes preemptive permanent deletion less harmful to localization. The systems hypothesis is separately whether gross growth plus mapping-phase headroom reduces measured peaks relative to post-hoc maintenance. Do not claim absence of either idea from prior art; this was a targeted reread of three identified papers, not an exhaustive novelty search. A stronger future method may require loss-conditioned support or revisit evidence, but neither is necessary to test this first hypothesis.

## 15. Proposed commands/config contract (not available yet)

Working directory: `/home/phillubt/projects/3dGS/MonoGS_System/MonoGS`. Retain existing `.venv`, library path and CPU transport. The following runner/config names are **to implement**, not commands run in this design session.

Proposed config additions:

```yaml
inherit_from: configs/rgbd/replica/office0_slice100_sp.yaml
Retention:
  enabled: true
  policy: tracking_support # opacity | random | lru | tracking_support
  max_gaussians: 40000
  backend_allocated_budget_bytes: 1264735437
  low_watermark_fraction: 0.95
  max_densify_gross_fraction: 0.05
  visibility_half_life_frames: 20
  utility_weights: [0.50, 0.25, 0.15, 0.10]
  protect_support: true
  use_tracking_history: true
  spatial_cell_m: 0.5
  spatial_min_rows: 2
  origin_min_rows: 64
  window_view_min_rows: 128
  mapping_extra_bytes_per_row: 14224
  forecast_multiplier: 1.10
  safety_margin_bytes: 16777216
  max_corrective_passes: 2
  feedback_mode: single_thread_keyframe_aggregate
Evaluation:
  pose_ids: all
  rendering_ids: [5,10,15,20,25,30,35,45,50,55,60,65,70,75,80,90,95]
```

The runner generates separate resolved configs for each ranking/budget/ablation, validates both scheduling flags, invokes the existing CLI and records independent output directories. No new CLI budget flag is necessary; YAML is the source of truth. A research config with `Retention.enabled: false` selects upstream comparison, with the same fixed evaluator. Reject conflicting `--opacity-prune-threshold` and enabled Retention.

```bash
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  .venv/bin/python -m pytest tests/test_retention_policy.py tests/test_gaussian_metadata.py

# Single research run after implementing the config/controller:
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  .venv/bin/python slam.py --config configs/research/office0_retention_v1.yaml \
  --online-eval --seed 0 --no-memory-limits --cpu-transfer

# Proposed runner contract; core includes its fixed-evaluation upstream reference.
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  .venv/bin/python scripts/run_retention_pilot.py --stage core --seed 0 \
  --budgets 50000 40000 --policies opacity random lru tracking_support

# Two additional runs at 40k: no T and no P.
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  .venv/bin/python scripts/run_retention_pilot.py --stage ablations --seed 0 --budgets 40000

# Runner requires explicit comparator selected from completed feasible core results.
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH="$PWD/.toolchain/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
  .venv/bin/python scripts/run_retention_pilot.py --stage bytes --seed 0 \
  --budgets 50000 --backend-budget-bytes 1092271513 --comparator lru
```

The final command's `lru` is an example; use the actual strongest feasible core comparator. The runner must echo the selected policy and record how it was selected. Any ten-frame smoke must be a separately declared dataset/config slice; do not silently alter this 100-frame dataset for core comparisons.

## 16. Work schedule through Thursday October 8

1. Oct 1–2: controller/ranking pure functions and metadata lifecycle; validate synthetic tests. Coordinate controller hook contract with Devon without blocking Phillip's ranking design.
2. Oct 3–4: integrate versioned tracking feedback, growth hooks, phase peak accounting and fixed evaluation; ten-frame local smoke, then one 40k proposed run to catch invariants. No hardware migration.
3. Oct 5–6: eight-run core + fixed-evaluation reference; two minimum ablations; inspect compliance before quality tables.
4. Oct 7: two byte-ceiling runs and optional paired seed; summarize evidence/negative results, update handoff, preserve configs and artifacts. No additional scene or novelty claim required before Thursday.

Keep scope bounded: postpone learned policies, merging, disk/CPU hierarchy, exact per-Gaussian tracking sensitivity, Jetson accounting, asynchronous operation and an exhaustive literature benchmark. Controller prediction calibration, actual loss utility and long-horizon revisit benefit remain explicit open questions.
