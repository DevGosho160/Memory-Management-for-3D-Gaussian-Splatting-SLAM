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
    if gaussian_model.get_xyz.shape[0] == 0: return
    
    # Find the maximum scaling factor for each gaussian
    max_scales = gaussian_model.get_scaling.max(dim=1).values
    
    # Prune if the gaussian's largest axis exceeds the maximum allowed scale
    prune_mask = max_scales > max_scale
    gaussian_model.prune_points(prune_mask)

def prune_by_visibility(gaussian_model: GaussianModel, min_observations: int):
    if gaussian_model.get_xyz.shape[0] == 0: return
    
    # Assuming gaussian_model.n_obs tracks how many times a gaussian was seen
    prune_mask = (gaussian_model.n_obs < min_observations).cuda()
    gaussian_model.prune_points(prune_mask)

def prune_by_voxel_grid(gaussian_model: GaussianModel, voxel_size: float):
    if gaussian_model.get_xyz.shape[0] == 0: return
    
    # 1. Discretize xyz coordinates into voxel indices
    # 2. Find duplicate voxel indices
    # 3. Create a prune_mask that is True for all redundant gaussians in the same voxel
    # 4. gaussian_model.prune_points(prune_mask)
    pass

def prune_by_density(gaussian_model: GaussianModel, min_density: float):
    """
    Removes gaussians based on density (inverse of average inter-gaussian distance).
    Calculated using k-nearest neighbors.
    """
    if gaussian_model.get_xyz.shape[0] == 0: return
    
    from scipy.spatial import KDTree
    
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()
    
    # Build KD-tree for fast neighbor lookup
    tree = KDTree(xyz)
    
    # Query for 5 nearest neighbors (including self)
    distances, _ = tree.query(xyz, k=5)
    
    # Average distance to neighbors (ignore self distance k=1)
    avg_dist = distances[:, 1:].mean(axis=1)
    
    # Density is inverse of distance
    density = 1.0 / (avg_dist + 1e-6) # Add epsilon to avoid division by zero
    
    prune_mask = (density < min_density).cuda()
    gaussian_model.prune_points(prune_mask)
