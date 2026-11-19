"""CPU row metadata with the same append/filter order as Gaussian parameters."""

import torch


class GaussianMetadata:
    FIELDS = {
        "gaussian_id": torch.int64,
        "row_created_frame": torch.int32,
        "lineage_birth_frame": torch.int32,
        "last_seen_frame": torch.int32,
        "tracking_ema": torch.float32,
        "probation_until_kf_event": torch.int32,
        "observed_since_creation": torch.bool,
    }

    def __init__(self):
        for name, dtype in self.FIELDS.items():
            setattr(self, name, torch.empty(0, dtype=dtype))
        self.next_gaussian_id = 0
        self.map_version = 0
        self.kf_event_index = 0

    def __len__(self):
        return self.gaussian_id.numel()

    def append(self, count, frame, parent_indices=None):
        if count == 0:
            return
        if parent_indices is not None:
            parent_indices = parent_indices.to(device="cpu", dtype=torch.long)
            assert parent_indices.numel() == count
            assert torch.all((parent_indices >= 0) & (parent_indices < len(self)))
        ids = torch.arange(self.next_gaussian_id, self.next_gaussian_id + count,
                           dtype=torch.int64)
        self.next_gaussian_id += count
        values = {
            "gaussian_id": ids,
            "row_created_frame": torch.full((count,), frame, dtype=torch.int32),
            "lineage_birth_frame": (self.lineage_birth_frame[parent_indices].clone()
                                    if parent_indices is not None else
                                    torch.full((count,), frame, dtype=torch.int32)),
            "last_seen_frame": (self.last_seen_frame[parent_indices].clone()
                                if parent_indices is not None else
                                torch.full((count,), -1, dtype=torch.int32)),
            "tracking_ema": (self.tracking_ema[parent_indices].clone()
                             if parent_indices is not None else
                             torch.zeros(count, dtype=torch.float32)),
            "probation_until_kf_event": (
                self.probation_until_kf_event[parent_indices].clone()
                if parent_indices is not None else
                torch.full((count,), self.kf_event_index + 1, dtype=torch.int32)),
            "observed_since_creation": torch.zeros(count, dtype=torch.bool),
        }
        for name in self.FIELDS:
            setattr(self, name, torch.cat((getattr(self, name), values[name])))
        self.map_version += 1

    def filter(self, keep):
        keep = keep.to(device="cpu", dtype=torch.bool)
        assert keep.numel() == len(self)
        for name in self.FIELDS:
            setattr(self, name, getattr(self, name)[keep])
        self.map_version += 1

    def assert_aligned(self, count):
        assert all(getattr(self, name).numel() == count for name in self.FIELDS)
        assert torch.unique(self.gaussian_id).numel() == count

    def apply_feedback(self, *, ids, version, sequence, frames, weighted_hits,
                       last_seen, beta):
        if version != self.map_version:
            raise ValueError("Tracking feedback map version mismatch")
        if sequence != getattr(self, "_next_feedback_sequence", 0):
            raise ValueError("Tracking feedback sequence mismatch")
        ids = torch.as_tensor(ids, dtype=torch.int64)
        weighted_hits = torch.as_tensor(weighted_hits, dtype=torch.float32)
        last_seen = torch.as_tensor(last_seen, dtype=torch.int32)
        if ids.numel() != weighted_hits.numel() or ids.numel() != last_seen.numel():
            raise ValueError("Tracking feedback length mismatch")
        # Stable IDs stay ordered after append/filter. Duplicate feedback IDs are not
        # emitted by the frontend, but retain the old sequential behavior for them.
        if ids.numel() and torch.unique(ids).numel() != ids.numel():
            positions = {int(value): idx for idx, value in enumerate(self.gaussian_id.tolist())}
            for j, value in enumerate(ids.tolist()):
                i = positions.get(value)
                if i is None:
                    continue
                self.tracking_ema[i] = beta ** frames * self.tracking_ema[i] + weighted_hits[j]
                self.last_seen_frame[i] = max(self.last_seen_frame[i], last_seen[j])
                if last_seen[j] >= 0:
                    self.observed_since_creation[i] = True
        elif ids.numel() and len(self):
            positions = torch.searchsorted(self.gaussian_id, ids)
            valid = positions < len(self)
            valid[valid.clone()] &= self.gaussian_id[positions[valid]] == ids[valid]
            rows = positions[valid]
            self.tracking_ema[rows] = beta ** frames * self.tracking_ema[rows] + weighted_hits[valid]
            self.last_seen_frame[rows] = torch.maximum(self.last_seen_frame[rows], last_seen[valid])
            self.observed_since_creation[rows] |= last_seen[valid] >= 0
        self._next_feedback_sequence = sequence + 1
