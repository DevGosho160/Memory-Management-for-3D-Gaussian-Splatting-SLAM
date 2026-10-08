import torch
from gaussian_splatting.scene.gaussian_model import GaussianModel

def prune_by_opacity(gaussian_model: GaussianModel, min_opacity: float):
    """Prune low-opacity rows; return the pre-prune mask for visibility alignment."""
    count = gaussian_model.get_xyz.shape[0]
    if count == 0:
        return torch.zeros(0, dtype=torch.bool, device=gaussian_model.get_xyz.device)

    opacity = gaussian_model.get_opacity.reshape(-1)
    prune_mask = opacity < min_opacity
    to_prune = int(prune_mask.sum().item())
    if to_prune == count:
        # The renderer cannot operate on an empty map. Ties keep the first row.
        prune_mask[torch.argmax(opacity)] = False
        to_prune -= 1
    if to_prune:
        gaussian_model.prune_points(prune_mask)
    return prune_mask

def prune_by_volume(gaussian_model: GaussianModel, max_scale: float):
    count = gaussian_model.get_xyz.shape[0]
    if count == 0: return torch.zeros(0, dtype=torch.bool, device=gaussian_model.get_xyz.device)
    
    max_scales = gaussian_model.get_scaling.max(dim=1).values
    prune_mask = max_scales > max_scale
    if prune_mask.any():
        gaussian_model.prune_points(prune_mask)
    return prune_mask

def prune_by_visibility(gaussian_model: GaussianModel, min_observations: int):
    count = gaussian_model.get_xyz.shape[0]
    if count == 0: return torch.zeros(0, dtype=torch.bool, device=gaussian_model.get_xyz.device)
    
    prune_mask = (gaussian_model.n_obs < min_observations).cuda()
    if prune_mask.any():
        gaussian_model.prune_points(prune_mask)
    return prune_mask

def prune_by_voxel_grid(gaussian_model: GaussianModel, voxel_size: float):
    count = gaussian_model.get_xyz.shape[0]
    if count == 0: return torch.zeros(0, dtype=torch.bool, device=gaussian_model.get_xyz.device)
    
    xyz = gaussian_model.get_xyz
    voxel_coords = torch.floor(xyz / voxel_size).to(torch.int64)
    min_coords = voxel_coords.min(dim=0).values
    shifted_coords = voxel_coords - min_coords
    
    max_coords = shifted_coords.max(dim=0).values
    stride_y = max_coords[0] + 1
    stride_z = stride_y * (max_coords[1] + 1)
    
    linear_indices = shifted_coords[:, 0] + shifted_coords[:, 1] * stride_y + shifted_coords[:, 2] * stride_z
    sorted_indices, sorted_idx = torch.sort(linear_indices)
    
    is_first = torch.cat([torch.tensor([True], device=xyz.device), sorted_indices[1:] != sorted_indices[:-1]])
    keep_indices = sorted_idx[is_first]
    
    prune_mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    prune_mask[keep_indices] = False
    
    if prune_mask.any():
        gaussian_model.prune_points(prune_mask)
    return prune_mask

def prune_by_density(gaussian_model: GaussianModel, min_density: float):
    count = gaussian_model.get_xyz.shape[0]
    if count == 0: return torch.zeros(0, dtype=torch.bool, device=gaussian_model.get_xyz.device)
    
    from scipy.spatial import KDTree
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    tree = KDTree(xyz)
    distances, _ = tree.query(xyz, k=5)
    avg_dist = distances[:, 1:].mean(axis=1)
    density = 1.0 / (avg_dist + 1e-6)
    
    prune_mask = torch.from_numpy(density < min_density).cuda()
    if prune_mask.any():
        gaussian_model.prune_points(prune_mask)
    return prune_mask
