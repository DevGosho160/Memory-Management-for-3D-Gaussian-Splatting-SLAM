"""Controls and validity checks for the opt-in frozen-pose retention diagnostic."""

import hashlib
import heapq
import io
import json
import random

import numpy as np
import torch

from project_utils.retention_policy import rank_rows


def digest(value):
    """Stable digest of nested tensors and ordinary Python state."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(b"tensor")
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            visit(torch.from_numpy(np.ascontiguousarray(item)))
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def model_state(model, *, iteration=0, visibility=None):
    groups = []
    for group in model.optimizer.param_groups:
        groups.append({key: value for key, value in group.items() if key != "params"})
    return dict(parameters={name: getattr(model, name) for name in
                            ("_xyz", "_features_dc", "_features_rest", "_scaling",
                             "_rotation", "_opacity")},
                optimizer=model.optimizer.state_dict(), groups=groups,
                statistics={name: getattr(model, name) for name in
                            ("max_radii2D", "xyz_gradient_accum", "denom", "unique_kfIDs", "n_obs")},
                metadata={name: getattr(model.metadata, name) for name in model.metadata.FIELDS},
                counters={name: getattr(model.metadata, name) for name in
                          ("next_gaussian_id", "map_version", "kf_event_index")},
                iteration=iteration, visibility=visibility or {})


def assert_optimizer_bound(model):
    expected = {id(getattr(model, name)) for name in
                ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")}
    actual = {id(param) for group in model.optimizer.param_groups for param in group["params"]}
    if expected != actual:
        raise AssertionError("Restored optimizer is not bound to restored parameters")


def restore_verified(path, expected_hash):
    # This checkpoint is written and read locally by this runner; never load untrusted files.
    saved = torch.load(path, weights_only=False)
    model = saved["model"]
    assert_optimizer_bound(model)
    actual = digest(model_state(model, iteration=saved["iteration"],
                                visibility=saved["visibility"]))
    if actual != expected_hash:
        raise AssertionError("Initialized state differs from shared checkpoint")
    return saved


def fixed_schedule(keyframes, *, window_size=10, steps=150, seed=0):
    if not keyframes or keyframes[0] != 0 or keyframes != sorted(set(keyframes)):
        raise ValueError("Invalid frozen keyframe schedule")
    rng = np.random.default_rng(seed + 19073)
    result = {}
    for position, frame in enumerate(keyframes[1:], 1):
        prior = keyframes[:position + 1]
        window = list(reversed(prior[-window_size:]))
        older = [item for item in prior if item not in window]
        result[frame] = dict(window=window,
                             older=[rng.permutation(older)[:2].tolist() for _ in range(steps)])
    return result


def spatially_balanced_order(ids, xyz, protected, seed, cell_m=0.5):
    """Lowest retained cell population, then existing Random priority and stable ID."""
    ids = np.asarray(ids, dtype=np.int64)
    protected = np.asarray(protected, dtype=bool)
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(ids) != len(protected) or len(ids) != len(xyz):
        raise ValueError("Spatial ranking length mismatch")
    empty = np.zeros(len(ids))
    random_order = rank_rows("random", ids, empty, empty, empty, empty,
                             empty, 0, np.zeros(len(ids), bool), seed=seed)
    cells = np.floor(xyz / cell_m).astype(np.int64)
    keys = [tuple(cell) for cell in cells]
    groups, counts = {}, {}
    for index, key in enumerate(keys):
        if protected[index]:
            counts[key] = counts.get(key, 0) + 1
    for priority, index in enumerate(random_order):
        if not protected[index]:
            groups.setdefault(keys[index], []).append((priority, int(ids[index]), int(index)))
    heap = []
    for key, group in groups.items():
        heapq.heappush(heap, (counts.get(key, 0), *group[0][:2], key, 0))
    result = []
    while heap:
        population, _, _, key, offset = heapq.heappop(heap)
        result.append(groups[key][offset][2])
        offset += 1
        if offset < len(groups[key]):
            next_priority, next_id, _ = groups[key][offset]
            heapq.heappush(heap, (population + 1, next_priority, next_id, key, offset))
    return np.asarray(np.flatnonzero(protected).tolist() + result, dtype=np.int64)


def rng_state():
    return dict(py=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
