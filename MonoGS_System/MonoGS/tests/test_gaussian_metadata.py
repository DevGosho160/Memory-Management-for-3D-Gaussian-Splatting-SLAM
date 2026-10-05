import pytest
import torch
from types import SimpleNamespace

from project_utils.gaussian_metadata import GaussianMetadata


def test_insert_clone_split_filter_and_never_reuse_ids():
    meta = GaussianMetadata()
    meta.append(3, frame=4)
    meta.tracking_ema[:] = torch.tensor([0.1, 0.2, 0.3])
    meta.last_seen_frame[:] = torch.tensor([1, 2, 3])
    meta.kf_event_index = 2
    meta.append(1, frame=8, parent_indices=torch.tensor([1]))
    meta.append(2, frame=8, parent_indices=torch.tensor([0, 2]))
    assert meta.gaussian_id.tolist() == list(range(6))
    assert meta.lineage_birth_frame.tolist() == [4] * 6
    assert meta.tracking_ema.tolist() == pytest.approx([.1, .2, .3, .2, .1, .3])
    assert meta.probation_until_kf_event.tolist() == [1] * 6
    assert not meta.observed_since_creation.any()
    meta.filter(torch.tensor([False, True, True, True, True, True]))
    meta.append(1, frame=9)
    assert meta.gaussian_id.tolist() == [1, 2, 3, 4, 5, 6]
    assert meta.last_seen_frame[-1] == -1
    assert meta.probation_until_kf_event[-1] == 3


def test_feedback_aggregate_and_version_checks():
    meta = GaussianMetadata()
    meta.append(2, frame=0)
    beta = 2 ** (-1 / 20)
    hits = [torch.tensor([1., 0.]), torch.tensor([0., 1.]), torch.tensor([1., 1.])]
    expected = torch.zeros(2)
    aggregate = torch.zeros(2)
    for hit in hits:
        expected = beta * expected + (1 - beta) * hit
        aggregate = beta * aggregate + (1 - beta) * hit
    meta.apply_feedback(ids=[0, 1], version=meta.map_version, sequence=0,
                        frames=3, weighted_hits=aggregate, last_seen=[2, 2], beta=beta)
    assert torch.allclose(meta.tracking_ema, expected)
    with pytest.raises(ValueError, match="sequence"):
        meta.apply_feedback(ids=[0], version=meta.map_version, sequence=0,
                            frames=1, weighted_hits=[0], last_seen=[-1], beta=beta)
    with pytest.raises(ValueError, match="version"):
        meta.apply_feedback(ids=[0], version=-1, sequence=1,
                            frames=1, weighted_hits=[0], last_seen=[-1], beta=beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA model required")
def test_real_model_optimizer_and_metadata_follow_clone_split_prune():
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    model = GaussianModel(0)
    model.init_lr(6.0)
    model.training_setup(SimpleNamespace(
        percent_dense=0.1, position_lr_init=0.001, position_lr_final=0.001,
        position_lr_delay_mult=1.0, position_lr_max_steps=100,
        feature_lr=0.001, opacity_lr=0.001, scaling_lr=0.001,
        rotation_lr=0.001))
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], device="cuda")
    features = torch.zeros((3, 3, 1), device="cuda")
    scales = torch.tensor([[-4.] * 3, [-4.] * 3, [0.] * 3], device="cuda")
    rots = torch.tensor([[1., 0., 0., 0.]] * 3, device="cuda")
    opacity = torch.zeros((3, 1), device="cuda")
    model.extend_from_pcd(xyz, features, scales, rots, opacity, kf_id=4)
    assert model.metadata.gaussian_id.tolist() == [0, 1, 2]
    model.get_xyz.sum().backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    before = model.optimizer.state[model._xyz]["exp_avg"].clone()
    model.current_frame = 8
    model.densify_and_clone(torch.tensor([[1.], [0.], [0.]], device="cuda"), .5, 1.)
    assert model.metadata.gaussian_id.tolist() == [0, 1, 2, 3]
    assert model.metadata.lineage_birth_frame[-1] == 4
    assert torch.all(model.optimizer.state[model._xyz]["exp_avg"][-1] == 0)
    model.densify_and_split(torch.tensor([[0.], [0.], [1.], [0.]], device="cuda"),
                            .5, 1.)
    assert model.metadata.gaussian_id.tolist() == [0, 1, 3, 4, 5]
    model.prune_points(torch.tensor([False, True, False, False, False], device="cuda"))
    assert model.metadata.gaussian_id.tolist() == [0, 3, 4, 5]
    assert torch.allclose(model.optimizer.state[model._xyz]["exp_avg"][0], before[0])
    assert model.optimizer.state[model._xyz]["step"] == 1
    model.metadata.assert_aligned(model.get_xyz.shape[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA model required")
def test_prepruned_frozen_split_ids_respect_gross_staging_limit():
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    model = GaussianModel(0)
    model.init_lr(6.0)
    model.training_setup(SimpleNamespace(
        percent_dense=.1, position_lr_init=.001, position_lr_final=.001,
        position_lr_delay_mult=1., position_lr_max_steps=100,
        feature_lr=.001, opacity_lr=.001, scaling_lr=.001,
        rotation_lr=.001))
    model.extend_from_pcd(
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], device="cuda"),
        torch.zeros((3, 3, 1), device="cuda"),
        torch.tensor([[-4.] * 3, [-4.] * 3, [0.] * 3], device="cuda"),
        torch.tensor([[1., 0., 0., 0.]] * 3, device="cuda"),
        torch.zeros((3, 1), device="cuda"), kf_id=0)
    model.xyz_gradient_accum[:] = 1
    model.denom[:] = 1
    model.row_limit = 5

    def admit(grads, clone, split):
        assert clone.tolist() == [True, True, False]
        assert split.tolist() == [False, False, True]
        model.prune_points(torch.tensor([False, True, False], device="cuda"))
        return [0], [2]

    model.densify_admission_callback = admit
    model.densify_and_prune(.5, 0., 1., None)
    assert model.metadata.gaussian_id.tolist() == [0, 3, 4, 5]
    assert model.max_live_rows == 5
    model.metadata.assert_aligned(4)
