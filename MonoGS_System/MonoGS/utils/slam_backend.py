import random
import time
import json
import os

import numpy as np
import open3d as o3d
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from project_utils.budget_controller import RowBudgetController, BudgetInfeasible
from project_utils.retention_policy import rank_rows, protected_rows

from project_utils.map_pruning import prune_by_opacity
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.run_telemetry import RunTelemetry
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.telemetry = None
        self.frame_idx = None
        self.retention = None
        self.policy_times_ms = []
        self.retention_diagnostic_file = None

    def record_telemetry(self, event, **kwargs):
        if self.telemetry is not None:
            if self.retention is not None:
                kwargs.setdefault("row_ceiling", self.retention.maximum)
                kwargs.setdefault("row_violation", self.retention.violations)
            self.telemetry.record(
                event, self.gaussians, frame_idx=self.frame_idx,
                iteration=self.iteration_count,
                retained_keyframes=len(self.viewpoints),
                window_keyframes=len(self.current_window), **kwargs
            )

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.opacity_prune_threshold = self.config.get("Experiment", {}).get(
            "opacity_prune_threshold"
        )
        retention_config = self.config.get("Retention", {})
        if retention_config.get("enabled", False):
            if self.opacity_prune_threshold is not None:
                raise ValueError("Budget retention and extra opacity threshold conflict")
            if not self.single_thread or not self.config["Training"]["single_thread"]:
                raise ValueError("V1 retention requires both single-thread scheduling flags")
            self.retention = RowBudgetController(
                retention_config["max_gaussians"],
                retention_config.get("low_watermark_fraction", .95))
            self.retention_config = retention_config

    def _retention_inputs(self):
        model = self.gaussians
        n = model.get_xyz.shape[0]
        opacity = model.get_opacity.detach().flatten().cpu().numpy()
        masks = [self.occ_aware_visibility[k].detach().bool().cpu().numpy()
                 for k in self.current_window
                 if k in self.occ_aware_visibility and
                 len(self.occ_aware_visibility[k]) == n]
        if masks:
            window_fraction = np.mean(np.stack(masks), axis=0)
        else:
            window_fraction = np.zeros(n)
        meta = model.metadata
        protected = protected_rows(
            meta.gaussian_id.numpy(), opacity, model.unique_kfIDs.numpy(),
            model.get_xyz.detach().cpu().numpy(), masks,
            meta.tracking_ema.numpy(),
            cell_m=self.retention_config.get("spatial_cell_m", .5),
            cell_min=self.retention_config.get("spatial_min_rows", 2),
            origin_min=self.retention_config.get("origin_min_rows", 64),
            view_min=self.retention_config.get("window_view_min_rows", 128),
        ) if self.retention_config.get("protect_support", True) else np.zeros(n, bool)
        order = rank_rows(
            self.retention_config["policy"], meta.gaussian_id.numpy(), opacity,
            meta.last_seen_frame.numpy(), meta.lineage_birth_frame.numpy(),
            meta.tracking_ema.numpy(), window_fraction, self.frame_idx,
            meta.probation_until_kf_event.numpy() > meta.kf_event_index,
            seed=self.config.get("Experiment", {}).get("seed", 0),
            use_tracking_history=self.retention_config.get("use_tracking_history", True),
            half_life=self.retention_config.get("visibility_half_life_frames", 20))
        if self.retention_config.get("event_diagnostics", False):
            self._retention_diagnostic_state = dict(
                ids=meta.gaussian_id.numpy(), opacity=opacity,
                origins=model.unique_kfIDs.numpy(),
                xyz=model.get_xyz.detach().cpu().numpy(),
                last_seen=meta.last_seen_frame.numpy(),
                tracking=meta.tracking_ema.numpy(), window=window_fraction,
                probation=meta.probation_until_kf_event.numpy() > meta.kf_event_index,
                order=order)
        return order, protected

    def _record_retention_decision(self, event, plan, protected, growth, elapsed):
        """Optional CPU summary; performed after policy timing and before the next mutation."""
        if not self.retention_config.get("event_diagnostics", False):
            return
        state = self._retention_diagnostic_state
        n = len(plan.keep)
        seen = state["last_seen"] >= 0
        recency = np.zeros(n, dtype=np.float64)
        recency[seen] = np.exp2(-np.maximum(0, self.frame_idx - state["last_seen"][seen]) /
                                 self.retention_config.get("visibility_half_life_frames", 20))
        recency[state["probation"] & ~seen] = 1.0
        components = dict(T=state["tracking"], W=state["window"],
                          R=recency, O=state["opacity"])
        quantiles = [0, .1, .25, .5, .75, .9, 1]
        def distribution(values):
            values = np.asarray(values)
            return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                            np.quantile(values, quantiles).tolist())) if len(values) else {}
        correlations = {}
        names = list(components)
        for i, left in enumerate(names):
            for right in names[i+1:]:
                a, b = components[left], components[right]
                correlations[f"{left}_{right}"] = (float(np.corrcoef(a, b)[0, 1])
                    if n > 1 and np.std(a) > 0 and np.std(b) > 0 else None)
        selected = plan.keep
        origin, origin_counts = np.unique(state["origins"][selected], return_counts=True)
        cells = np.floor(state["xyz"][selected] /
                         self.retention_config.get("spatial_cell_m", .5)).astype(np.int64)
        unique_cells, cell_counts = np.unique(cells, axis=0, return_counts=True)
        record = dict(event=event, frame=self.frame_idx, iteration=self.iteration_count,
                      map_version=self.gaussians.metadata.map_version,
                      candidate_count=n, protected_count=plan.protected_count,
                      selected_count=int(selected.sum()), rejected_growth=growth-plan.admitted_growth,
                      admitted_growth=plan.admitted_growth, policy_decision_ms=elapsed,
                      component_distributions={key: distribution(value) for key, value in components.items()},
                      component_correlations=correlations,
                      nonzero_T_fraction=float(np.mean(components["T"] != 0)) if n else 0,
                      T_quantiles=distribution(components["T"]),
                      last_seen_quantiles=distribution(state["last_seen"]),
                      retained_origin_keyframes=dict(zip(map(str, origin.tolist()), origin_counts.tolist())),
                      retained_spatial_cells={",".join(map(str, cell)): int(count)
                                              for cell, count in zip(unique_cells.tolist(), cell_counts.tolist())})
        self.retention_diagnostic_file.write(json.dumps(record, allow_nan=False) + "\n")

    def _apply_row_plan(self, plan):
        if not np.all(plan.keep):
            keep = torch.from_numpy(plan.keep).to("cuda")
            self.gaussians.prune_points(~keep)
            self.occ_aware_visibility = {
                key: value[keep] for key, value in self.occ_aware_visibility.items()
                if len(value) == len(keep)}
        self.retention.observe(self.gaussians.get_xyz.shape[0])

    def _candidate_admission(self, xyz):
        started = time.perf_counter()
        count = len(xyz)
        old = self.gaussians.get_xyz.shape[0]
        order, protected = self._retention_inputs()
        plan = self.retention.plan(old, count, order, protected)
        self._apply_row_plan(plan)
        admitted = plan.admitted_growth
        if admitted == count:
            indices = np.arange(count)
        else:
            # Deterministic spatial round-robin; original candidate order breaks ties.
            cells = np.floor(xyz / self.retention_config.get("spatial_cell_m", .5)).astype(np.int64)
            groups = {}
            for i, cell in enumerate(cells):
                groups.setdefault(tuple(cell), []).append(i)
            indices = []
            depth = 0
            while len(indices) < admitted:
                for group in groups.values():
                    if depth < len(group):
                        indices.append(group[depth])
                        if len(indices) == admitted:
                            break
                depth += 1
            indices = np.asarray(indices, dtype=np.int64)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        self.policy_times_ms.append(elapsed)
        self.record_telemetry("retention_insert_plan", count_before=old,
                              count_after=self.gaussians.get_xyz.shape[0],
                              attempted_growth=count, admitted_growth=admitted,
                              rejected_growth=count-admitted, policy_wall_ms=elapsed,
                              protected_rows=plan.protected_count)
        self._record_retention_decision("insert", plan, protected, count, elapsed)
        return indices

    def _densify_admission(self, grads, clone_mask, split_mask):
        started = time.perf_counter()
        model = self.gaussians
        n = model.get_xyz.shape[0]
        ids = model.metadata.gaussian_id.numpy()
        gradient = grads.detach().flatten().cpu().numpy()
        clone = clone_mask.detach().cpu().numpy()
        split = split_mask.detach().cpu().numpy()
        candidates = [(i, 2 if split[i] else 1) for i in range(n)
                      if clone[i] or split[i]]
        candidates.sort(key=lambda item: (-gradient[item[0]], int(ids[item[0]])))
        gross_cap = max(0, int(self.retention_config.get(
            "max_densify_gross_fraction", .05) * self.retention.maximum))
        selected = []
        gross = 0
        for index, cost in candidates:
            if gross + cost > gross_cap:
                break
            selected.append((index, cost))
            gross += cost
        order, protected = self._retention_inputs()
        for index, _ in selected:
            protected[index] = True
        plan = self.retention.plan(n, gross, order, protected,
                                   atomic_costs=[cost for _, cost in selected])
        total_attempted = 2 * int(split.sum()) + int(clone.sum())
        self.retention.attempted_growth += total_attempted - gross
        self.retention.rejected_growth += total_attempted - gross
        accepted = []
        used = 0
        for index, cost in selected:
            if used + cost > plan.admitted_growth:
                break
            accepted.append((int(ids[index]), cost))
            used += cost
        self._apply_row_plan(plan)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        self.policy_times_ms.append(elapsed)
        self.record_telemetry("retention_densify_plan", count_before=n,
                              count_after=model.get_xyz.shape[0],
                              attempted_growth=total_attempted,
                              admitted_growth=used,
                              rejected_growth=total_attempted-used,
                              protected_rows=plan.protected_count,
                              policy_wall_ms=elapsed)
        self._record_retention_decision("densify", plan, protected, total_attempted, elapsed)
        return ([value for value, cost in accepted if cost == 1],
                [value for value, cost in accepted if cost == 2])

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        before = self.gaussians.get_xyz.shape[0]
        self.gaussians.current_frame = frame_idx
        if self.retention is not None:
            self.gaussians.candidate_admission_callback = self._candidate_admission
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )
        self.gaussians.candidate_admission_callback = None
        if self.retention is not None:
            self.retention.observe(self.gaussians.get_xyz.shape[0])
        self.record_telemetry("keyframe_insert", count_before=before,
                              count_after=self.gaussians.get_xyz.shape[0])

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        if self.retention is not None:
            self.gaussians.row_limit = self.retention.maximum
            self.gaussians.densify_admission_callback = self._densify_admission
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            iteration_start = time.perf_counter()
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    before = self.gaussians.get_xyz.shape[0]
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    self.record_telemetry("upstream_init_densify_prune",
                                          count_before=before,
                                          count_after=self.gaussians.get_xyz.shape[0])

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            self.record_telemetry("initialize_iteration",
                                  mapping_enqueue_ms=(time.perf_counter() - iteration_start) * 1000)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            iteration_start = time.perf_counter()
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            before = self.gaussians.get_xyz.shape[0]
                            self.gaussians.prune_points(to_prune.cuda())
                            self.record_telemetry("upstream_coviz_prune",
                                                  count_before=before,
                                                  count_after=self.gaussians.get_xyz.shape[0])
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    self.record_telemetry("map_prune_iteration",
                                          mapping_enqueue_ms=(time.perf_counter() - iteration_start) * 1000)
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    before = self.gaussians.get_xyz.shape[0]
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    self.record_telemetry("upstream_map_densify_prune",
                                          count_before=before,
                                          count_after=self.gaussians.get_xyz.shape[0])
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
            self.record_telemetry("map_iteration",
                                  mapping_enqueue_ms=(time.perf_counter() - iteration_start) * 1000)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.record_telemetry("sync_begin")
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        # Bound backend callbacks must not enter a frontend snapshot: deepcopy
        # would recursively capture the process, queues and authentication key.
        densify_callback = self.gaussians.densify_admission_callback
        admission_callback = self.gaussians.candidate_admission_callback
        self.gaussians.densify_admission_callback = None
        self.gaussians.candidate_admission_callback = None
        try:
            snapshot = clone_obj(self.gaussians)
        finally:
            self.gaussians.densify_admission_callback = densify_callback
            self.gaussians.candidate_admission_callback = admission_callback
        msg = [tag, snapshot, self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)
        self.record_telemetry("sync_enqueued")

    def run(self):
        seed = self.config.get("Experiment", {}).get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            o3d.utility.random.seed(seed)
        if self.config.get("Experiment", {}).get("cpu_transfer", False):
            # Only the transport changes: this is the same initially empty
            # GaussianModel that the parent normally passes through CUDA IPC.
            sh_degree = 3 if self.config["Training"]["spherical_harmonics"] else 0
            self.gaussians = GaussianModel(sh_degree, config=self.config)
            self.gaussians.init_lr(6.0)
            self.gaussians.training_setup(self.opt_params)
            self.background = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.telemetry = RunTelemetry(self.config["Results"].get("save_dir"), "backend")
        if self.retention is not None and self.retention_config.get("event_diagnostics", False):
            self.retention_diagnostic_file = open(os.path.join(
                self.config["Results"]["save_dir"], "retention_events.jsonl"),
                "w", encoding="utf-8", buffering=1)
        self.record_telemetry("backend_start")
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    self.frame_idx = cur_frame_idx
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    self.frame_idx = cur_frame_idx
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    if self.retention is not None:
                        feedback_start = time.perf_counter()
                        block = data[5]
                        if self.retention_config.get("event_diagnostics", False):
                            prior_t = self.gaussians.metadata.tracking_ema.numpy().copy()
                            prior_seen = self.gaussians.metadata.last_seen_frame.numpy().copy()
                        self.gaussians.metadata.apply_feedback(
                            ids=block["ids"], version=block["version"],
                            sequence=block["sequence"], frames=block["frames"],
                            weighted_hits=block["weighted_hits"],
                            last_seen=block["last_seen"],
                            beta=2 ** (-1 / self.retention_config.get(
                                "visibility_half_life_frames", 20)))
                        feedback_elapsed = (time.perf_counter() - feedback_start) * 1000
                        self.policy_times_ms.append(feedback_elapsed)
                        self.record_telemetry("tracking_feedback",
                                              count_before=self.gaussians.get_xyz.shape[0],
                                              count_after=self.gaussians.get_xyz.shape[0],
                                              policy_wall_ms=feedback_elapsed)
                        if self.retention_diagnostic_file is not None:
                            meta = self.gaussians.metadata
                            self.retention_diagnostic_file.write(json.dumps({
                                "event": "feedback", "frame": cur_frame_idx,
                                "start_frame": block["start_frame"],
                                "end_frame": block["end_frame"],
                                "frames": block["frames"], "sequence": block["sequence"],
                                "map_version": block["version"],
                                "feedback_ids": len(block["ids"]),
                                "nonzero_weighted_hit_fraction": float(np.mean(
                                    np.asarray(block["weighted_hits"]) != 0)),
                                "last_seen_changed_count": int(np.count_nonzero(
                                    meta.last_seen_frame.numpy() != prior_seen)),
                                "T_before_quantiles": np.quantile(prior_t,
                                    [0, .1, .25, .5, .75, .9, 1]).tolist(),
                                "T_after_quantiles": np.quantile(meta.tracking_ema.numpy(),
                                    [0, .1, .25, .5, .75, .9, 1]).tolist(),
                                "feedback_apply_ms": feedback_elapsed,
                            }) + "\n")

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    if self.opacity_prune_threshold is not None:
                        before = self.gaussians.get_xyz.shape[0]
                        with torch.no_grad():
                            prune_mask = prune_by_opacity(
                                self.gaussians, self.opacity_prune_threshold
                            )
                            if self.gaussians.get_xyz.shape[0] != before:
                                keep = ~prune_mask
                                self.occ_aware_visibility = {
                                    kf_id: visibility[keep]
                                    for kf_id, visibility in self.occ_aware_visibility.items()
                                }
                        after = self.gaussians.get_xyz.shape[0]
                        assert self.gaussians.unique_kfIDs.shape[0] == after
                        assert self.gaussians.n_obs.shape[0] == after
                        assert all(v.shape[0] == after for v in self.occ_aware_visibility.values())
                        self.record_telemetry("extra_opacity_prune", count_before=before,
                                              count_after=after)
                    self.push_to_frontend("keyframe")
                    if self.retention is not None:
                        self.gaussians.metadata.kf_event_index += 1
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        self.record_telemetry("backend_stop")
        if self.retention is not None and self.config["Results"].get("save_dir"):
            timings = np.asarray(self.policy_times_ms)
            with open(os.path.join(self.config["Results"]["save_dir"],
                                   "retention_summary.json"), "w", encoding="utf-8") as file:
                json.dump({
                    "policy": self.retention_config["policy"],
                    "row_ceiling": self.retention.maximum,
                    "max_live_rows": max(self.retention.max_observed,
                                         self.gaussians.max_live_rows),
                    "final_rows": int(self.gaussians.get_xyz.shape[0]),
                    "row_violations": self.retention.violations,
                    "attempted_growth": self.retention.attempted_growth,
                    "admitted_growth": self.retention.admitted_growth,
                    "rejected_growth": self.retention.rejected_growth,
                    "policy_wall_ms_median": (float(np.median(timings)) if len(timings) else 0),
                    "policy_wall_ms_p95": (float(np.percentile(timings, 95)) if len(timings) else 0),
                    "policy_wall_ms_total": float(timings.sum()),
                    "backend_cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "backend_cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }, file, indent=2)
        self.telemetry.close()
        if self.retention_diagnostic_file is not None:
            self.retention_diagnostic_file.close()
        return
