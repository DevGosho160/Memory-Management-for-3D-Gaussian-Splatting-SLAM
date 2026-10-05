"""Single-process frozen-pose, fixed-schedule retention diagnostic.

Run from MonoGS_System/MonoGS. Seed 0 only; analysis gate stops expansion.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from munch import munchify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from utils.camera_utils import Camera
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.pose_utils import update_pose
from utils.slam_backend import BackEnd
from utils.slam_utils import get_loss_mapping, get_loss_tracking
from project_utils.controlled_retention import (
    assert_optimizer_bound, digest, fixed_schedule, model_state, restore_verified,
    rng_state, spatially_balanced_order,
)

ARMS = ("random", "v1", "no_t", "spatial_random")
PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def cuda_stats():
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


def camera(dataset, frame, projection):
    item = Camera.init_from_dataset(dataset, frame, projection)
    item.update_RT(item.R_gt, item.T_gt)
    item.exposure_a.requires_grad_(False)
    item.exposure_b.requires_grad_(False)
    return item


def camera_signature(item):
    return digest((item.R, item.T, item.exposure_a, item.exposure_b))


def init_backend(config, pipe, opt, background, model):
    backend = BackEnd(config)
    backend.gaussians = model
    backend.pipeline_params = pipe
    backend.opt_params = opt
    backend.background = background
    backend.cameras_extent = 6.0
    backend.set_hyperparams()
    backend.gaussians.row_limit = backend.retention.maximum
    return backend


def build_batch(model, item, frame, seed):
    random.seed(seed + frame * 1009)
    np.random.seed(seed + frame * 1009)
    torch.manual_seed(seed + frame * 1009)
    torch.cuda.manual_seed_all(seed + frame * 1009)
    result = model.create_pcd_from_image(item)
    if result is None:
        raise RuntimeError(f"Empty insertion batch at frame {frame}")
    batch = tuple(t.detach().cpu().clone() for t in result)
    return batch


def map_step(model, config, pipe, background, views, window, older, iteration):
    loss = 0
    visibility = {}
    for frame in window + older:
        package = render(views[frame], model, pipe, background)
        loss = loss + get_loss_mapping(config, package["render"], package["depth"],
                                      views[frame], package["opacity"])
        if frame in window:
            visibility[frame] = (package["n_touched"] > 0).long()
    scaling = model.get_scaling
    loss = loss + 10 * torch.abs(scaling - scaling.mean(dim=1).view(-1, 1)).mean()
    loss.backward()
    with torch.no_grad():
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model.update_learning_rate(iteration)
    return visibility


def probe(model, dataset, frame, projection, config, pipe, background, steps):
    before_model = digest(model_state(model))
    before_rng = digest(rng_state())
    item = camera(dataset, frame, projection)
    previous = camera(dataset, frame - 1, projection)
    item.update_RT(previous.R_gt, previous.T_gt)
    item.exposure_a.requires_grad_(True)
    item.exposure_b.requires_grad_(True)
    item.compute_grad_mask(config)
    groups = [dict(params=[item.cam_rot_delta], lr=config["Training"]["lr"]["cam_rot_delta"]),
              dict(params=[item.cam_trans_delta], lr=config["Training"]["lr"]["cam_trans_delta"]),
              dict(params=[item.exposure_a], lr=.01), dict(params=[item.exposure_b], lr=.01)]
    optimizer = torch.optim.Adam(groups)
    def errors():
        translation = torch.linalg.norm(item.T - item.T_gt).item()
        angle = torch.acos(torch.clamp((torch.trace(item.R @ item.R_gt.to(item.R.dtype).T) - 1) / 2,
                                      -1, 1)).item() * 180 / np.pi
        return translation, angle
    initial_translation, initial_rotation = errors()
    started = time.perf_counter()
    for _ in range(steps):
        package = render(item, model, pipe, background)
        optimizer.zero_grad()
        loss = get_loss_tracking(config, package["render"], package["depth"],
                                 package["opacity"], item)
        # Probe gradients are restricted to camera variables. Clear model gradients below.
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            update_pose(item)
    torch.cuda.synchronize()
    final_translation, final_rotation = errors()
    for name in PARAMETERS:
        getattr(model, name).grad = None
    if digest(model_state(model)) != before_model or digest(rng_state()) != before_rng:
        raise AssertionError("Localization probe mutated mapper or RNG state")
    return dict(frame=frame, initial_translation_m=initial_translation,
                initial_rotation_deg=initial_rotation, translation_m=final_translation,
                rotation_deg=final_rotation, iterations=steps,
                time_s=time.perf_counter() - started,
                failed=not np.isfinite(final_translation + final_rotation))


def render_metrics(model, dataset, frames, projection, pipe, background):
    from gaussian_splatting.utils.image_utils import psnr
    from gaussian_splatting.utils.loss_utils import ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
    rows = []
    with torch.no_grad():
        for frame in frames:
            item = camera(dataset, frame, projection)
            package = render(item, model, pipe, background)
            image = package["render"].clamp(0, 1)
            gt = item.original_image
            valid_rgb = gt > 0
            depth = package["depth"]
            gt_depth = torch.from_numpy(item.depth).to(depth.device)[None]
            valid_depth = gt_depth > .01
            rendered = depth > .01
            rows.append(dict(frame=frame,
                             psnr=float(psnr(image[valid_rgb][None], gt[valid_rgb][None])),
                             ssim=float(ssim(image[None], gt[None])),
                             lpips=float(lpips(image[None], gt[None])),
                             depth_mae_m=float(torch.abs(depth[valid_depth] - gt_depth[valid_depth]).mean()),
                             depth_completeness=float((valid_depth & rendered).sum() / valid_depth.sum())))
    return rows


def diversity(model, frame):
    xyz = model.get_xyz.detach().cpu().numpy()
    cells, counts = np.unique(np.floor(xyz / .5).astype(np.int64), axis=0, return_counts=True)
    probs = counts / counts.sum()
    origins, origin_counts = np.unique(model.unique_kfIDs.numpy(), return_counts=True)
    origin_prob = origin_counts / origin_counts.sum()
    lineage_age = frame - model.metadata.lineage_birth_frame.numpy()
    return dict(occupied_cells=len(cells), effective_cells=float(np.exp(-(probs * np.log(probs)).sum())),
                top_ten_cell_mass=float(np.sort(probs)[-10:].sum()),
                effective_origins=float(np.exp(-(origin_prob * np.log(origin_prob)).sum())),
                lineage_age_quantiles=np.quantile(lineage_age, [0, .1, .5, .9, 1]).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/room0_controlled_retention.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/controlled_retention_seed0")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Seed expansion requires review of complete seed-0 results")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    control = config["ControlledRetention"]
    config["Training"]["monocular"] = False
    config["Results"]["use_gui"] = False
    config["Retention"]["event_diagnostics"] = False
    config["Retention"]["max_gaussians"] = control["ceiling"]
    model_params = munchify(config["model_params"])
    model_params.sh_degree = 0
    opt = munchify(config["opt_params"])
    pipe = munchify(config["pipeline_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    if len(dataset) != control["frames"]:
        raise AssertionError("Dataset does not match frozen 380-frame protocol")
    projection = getProjectionMatrix2(znear=.01, zfar=100, fx=dataset.fx, fy=dataset.fy,
                                      cx=dataset.cx, cy=dataset.cy, W=dataset.width,
                                      H=dataset.height).transpose(0, 1)
    background = torch.zeros(3, device="cuda")
    reference = json.loads(Path(control["reference_trajectory"]).read_text())
    keyframes = [int(value) for value in reference["trj_id"]]
    if len(keyframes) != 57 or keyframes[-1] >= 380:
        raise AssertionError("Random reference keyframes differ from approved 57")
    schedule = fixed_schedule(keyframes, window_size=config["Training"]["window_size"],
                              steps=control["mapping_steps"], seed=args.seed)
    schedule_hash = digest(schedule)
    save_json(output / "schedule.json", dict(keyframes=keyframes, schedule=schedule, hash=schedule_hash))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = GaussianModel(0, config=config)
    model.init_lr(6.0)
    model.training_setup(opt)
    backend = init_backend(config, pipe, opt, background, model)
    first = camera(dataset, 0, projection)
    backend.frame_idx = 0
    backend.add_next_kf(0, first, init=True)
    backend.initialize_map(0, first)
    if backend.retention.attempted_growth != backend.retention.admitted_growth or backend.retention.max_observed > control["ceiling"]:
        raise AssertionError("Shared initialization used policy-dependent budget deletion")
    checkpoint = output / "initialized.pt"
    torch.save(dict(model=model, iteration=backend.iteration_count,
                    visibility=backend.occ_aware_visibility), checkpoint)
    initial_hash = digest(model_state(model, iteration=backend.iteration_count,
                                      visibility=backend.occ_aware_visibility))
    del backend, model, first
    torch.cuda.empty_cache()
    for _ in ARMS:
        restore_verified(checkpoint, initial_hash)
    batches = {}
    batch_hashes = {}
    cache_model = restore_verified(checkpoint, initial_hash)["model"]
    for frame in keyframes[1:]:
        item = camera(dataset, frame, projection)
        batch = build_batch(cache_model, item, frame, args.seed)
        batches[frame] = batch
        batch_hashes[frame] = digest(batch)
        del item
    del cache_model
    shared_rng = rng_state()
    save_json(output / "controls.json", dict(seed=args.seed, initialization_hash=initial_hash,
              schedule_hash=schedule_hash, batch_hashes=batch_hashes,
              keyframe_ids=keyframes, batch_sizes={k: len(v[0]) for k, v in batches.items()}))
    if args.validate_only:
        test_model = restore_verified(checkpoint, initial_hash)["model"]
        probe_result = probe(test_model, dataset, control["return_probe_start"],
            projection, config, pipe, background, control["probe_steps"])
        render_result = render_metrics(test_model, dataset,
            [control["return_probe_start"]], projection, pipe, background)
        save_json(output / "preflight.json", dict(probe=probe_result, rendering=render_result,
                                                   initial_hash=initial_hash))
        print(f"Validated shared checkpoint and {len(batches)} cached batches: {output}", flush=True)
        return
    baseline_counts = None
    for arm in ARMS:
        started = time.perf_counter()
        random.setstate(shared_rng["py"])
        np.random.set_state(shared_rng["numpy"])
        torch.set_rng_state(shared_rng["torch"])
        torch.cuda.set_rng_state_all(shared_rng["cuda"])
        saved = restore_verified(checkpoint, initial_hash)
        model = saved["model"]
        torch.cuda.reset_peak_memory_stats()
        backend = init_backend(config, pipe, opt, background, model)
        backend.iteration_count = saved["iteration"]
        backend.occ_aware_visibility = saved["visibility"]
        backend.retention.observe(len(model.get_xyz))
        backend.retention_config["policy"] = "random" if arm in ("random", "spatial_random") else "tracking_support"
        backend.retention_config["use_tracking_history"] = arm != "no_t"
        views = {0: camera(dataset, 0, projection)}
        signatures = {0: camera_signature(views[0])}
        feedback_sequence = 0
        beta = 2 ** (-1 / config["Retention"]["visibility_half_life_frames"])
        events, probes, render_sets, diversity_rows = [], [], {}, []
        mapping_time = ranking_time = probe_time = 0.0
        optimizer_steps = 0
        frozen_return_map = None
        for frame in range(control["frames"]):
            item = views.get(frame) or camera(dataset, frame, projection)
            with torch.no_grad():
                hits = (render(item, model, pipe, background)["n_touched"] > 0).cpu()
            metadata = model.metadata
            metadata.apply_feedback(ids=metadata.gaussian_id.clone(),
                version=metadata.map_version, sequence=feedback_sequence, frames=1,
                weighted_hits=hits.float() * (1 - beta),
                last_seen=torch.where(hits, torch.tensor(frame, dtype=torch.int32),
                                      torch.tensor(-1, dtype=torch.int32)), beta=beta)
            feedback_sequence += 1
            if frame == control["return_map_frame"]:
                frozen_return_map = copy.deepcopy(model)
                assert_optimizer_bound(frozen_return_map)
                diversity_rows.append(dict(frame=frame, **diversity(model, frame)))
                render_sets["persistent_return"] = render_metrics(frozen_return_map, dataset,
                    range(control["return_probe_start"], control["return_probe_end"] + 1),
                    projection, pipe, background)
            if control["return_probe_start"] <= frame <= control["return_probe_end"]:
                row = probe(frozen_return_map, dataset, frame, projection, config, pipe, background,
                            control["probe_steps"])
                probes.append(row)
                probe_time += row["time_s"]
            if frame in schedule:
                event_start = time.perf_counter()
                backend.frame_idx = frame
                backend.current_window = schedule[frame]["window"]
                backend.viewpoints = views
                old_count = len(model.get_xyz)
                order, protected = backend._retention_inputs()
                if arm == "spatial_random":
                    order = spatially_balanced_order(model.metadata.gaussian_id.numpy(),
                        model.get_xyz.detach().cpu().numpy(), protected, args.seed,
                        cell_m=control["spatial_cell_m"])
                batch = batches[frame]
                if digest(batch) != batch_hashes[frame]:
                    raise AssertionError("Cached batch changed")
                plan = backend.retention.plan(old_count, len(batch[0]), order, protected)
                if plan.admitted_growth != len(batch[0]):
                    raise AssertionError(f"Incomplete cached admission: {arm} frame {frame}")
                backend._apply_row_plan(plan)
                post_retention = len(model.get_xyz)
                torch.cuda.synchronize()
                ranking_time += time.perf_counter() - event_start
                model.current_frame = frame
                model.extend_from_pcd(*(part.cuda().clone() for part in batch), kf_id=frame)
                backend.retention.observe(len(model.get_xyz))
                model.metadata.kf_event_index += 1
                backend.occ_aware_visibility = {
                    key: mask for key, mask in backend.occ_aware_visibility.items()
                    if key in backend.current_window}
                occupancy = len(model.get_xyz)
                event = dict(frame=frame, before=old_count, post_retention=post_retention,
                             after=occupancy,
                             admitted=len(batch[0]), protected=plan.protected_count,
                             batch_hash=batch_hashes[frame], window=backend.current_window)
                events.append(event)
                if baseline_counts is not None and occupancy != baseline_counts[len(events)-1]:
                    raise AssertionError("Row occupancy diverged across arms")
                views[frame] = item
                signatures[frame] = camera_signature(item)
                for older in schedule[frame]["older"]:
                    step_start = time.perf_counter()
                    backend.iteration_count += 1
                    backend.occ_aware_visibility = map_step(model, config, pipe, background,
                        views, backend.current_window, older, backend.iteration_count)
                    optimizer_steps += 1
                    mapping_time += time.perf_counter() - step_start
                diversity_rows.append(dict(frame=frame, **diversity(model, frame)))
            if frame in views:
                for view_id, view in views.items():
                    if camera_signature(view) != signatures[view_id]:
                        raise AssertionError("Mapping pose or exposure mutated")
            if frame % 20 == 0:
                print(f"{arm}: frame {frame}, rows {len(model.get_xyz)}", flush=True)
        if optimizer_steps != control["mapping_steps"] * (len(keyframes) - 1):
            raise AssertionError("Optimizer-step count mismatch")
        if baseline_counts is None:
            baseline_counts = [event["after"] for event in events]
        render_sets["final"] = render_metrics(model, dataset,
            config["Evaluation"]["rendering_ids"], projection, pipe, background)
        summary = dict(arm=arm, seed=args.seed, initial_hash=initial_hash,
                       schedule_hash=schedule_hash, batch_hashes=batch_hashes,
                       optimizer_steps=optimizer_steps, occupancy=baseline_counts,
                       row_violations=backend.retention.violations,
                       max_rows=backend.retention.max_observed,
                       events=events, probes=probes, rendering=render_sets,
                       diversity=diversity_rows, mapping_time_s=mapping_time,
                       ranking_time_s=ranking_time, probe_time_s=probe_time,
                       elapsed_s=time.perf_counter() - started, cuda=cuda_stats())
        save_json(output / f"{arm}.json", summary)
        print(f"Completed {arm}: {output / (arm + '.json')}", flush=True)
        del backend, model, frozen_return_map, views
        torch.cuda.empty_cache()
    print("Seed-0 four-arm run complete. Stop for analysis and review.", flush=True)


if __name__ == "__main__":
    main()
