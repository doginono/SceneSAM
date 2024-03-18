import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt  # J: added


# ----------------added-------------------
def calculate_orientation_error(gt_pose, est_pose):
    # Implement based on your representation (quaternions or Euler angles)
    # Placeholder function:
    return np.abs(np.array(gt_pose[:3, :3]) - np.array(est_pose[:3, :3])).mean()


def calculate_position_error(gt_pose, est_pose):
    # Assuming gt_pose and est_pose are (x, y, z)
    return np.linalg.norm(np.array(gt_pose[:3, -1]) - np.array(est_pose[:3, -1]))


def log_tracking_error(gt_poses, est_poses, writer):
    s = np.ones((4, 4), int)
    s[[0, 0, 1, 1, 2], [1, 2, 0, 3, 3]] *= -1
    # Calculate errors
    est_poses = est_poses * s
    position_errors = [
        calculate_position_error(gt, est) for gt, est in zip(gt_poses, est_poses)
    ]
    orientation_errors = [
        calculate_orientation_error(gt, est) for gt, est in zip(gt_poses, est_poses)
    ]

    # Create the plot
    fig, axs = plt.subplots(2, 1, figsize=(7, 5))
    axs[0].plot(position_errors, label="Position Error")
    axs[0].set_xlabel("Frame")
    axs[0].set_ylabel("Error")
    axs[0].set_title("Position Tracking Error")
    axs[0].legend()

    axs[1].plot(orientation_errors, label="Orientation Error")
    axs[1].set_xlabel("Frame")
    axs[1].set_ylabel("Error")
    axs[1].set_title("Orientation Tracking Error")
    axs[1].legend()

    plt.tight_layout()

    writer.add_figure("Tracking Errors", fig)

    # Don't forget to close the plot
    plt.close(fig)


# function to turn a vector of instance ids into rgb colors
def get_rgb_from_instance_id(instance_id, num_instances, colormap="tab20"):
    # instance_id is a vector of shape (N, )
    # returns a vector of shape (N, 3) with rgb colors
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=0, vmax=num_instances)
    color_mapping = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Map each instance ID to a color
    colors = color_mapping.to_rgba(instance_id)

    return colors


# ------------------------------------------


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def sample_pdf(bins, weights, N_samples, det=False, device="cuda:0"):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted

        inds = searchsorted(cdf, u, side="right")
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[: min(l, k)])


def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(
        device
    )
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def select_uv(i, j, n, depth, color, semantic, device="cuda:0"):
    """
    changed arguments to include semantic

    Select n uv from dense uv.

    """
    i = i.reshape(-1)
    j = j.reshape(-1)
    indices = torch.randint(i.shape[0], (n,), device=device)
    indices = indices.clamp(0, i.shape[0])
    i = i[indices]  # (n)
    j = j[indices]  # (n)
    depth = depth.reshape(-1)
    color = color.reshape(-1, 3)

    depth = depth[indices]  # (n)
    color = color[indices]  # (n,3)
    if semantic is not None:
        semantic = semantic.reshape(
            -1, semantic.shape[-1]
        )  # TODO: check if this is 101 for room0
        semantic = semantic[indices]  # (n, #instances)
    return i, j, depth, color, semantic


def get_sample_uv(H0, H1, W0, W1, n, depth, color, semantic, device="cuda:0"):
    """
    changed arguments to include semantic

    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depth = depth[H0:H1, W0:W1]
    color = color[H0:H1, W0:W1]
    if semantic is not None:
        semantic = semantic[H0:H1, W0:W1]
    i, j = torch.meshgrid(
        torch.linspace(W0, W1 - 1, W1 - W0).to(device),
        torch.linspace(H0, H1 - 1, H1 - H0).to(device),
    )
    i = i.t()  # transpose
    j = j.t()
    i, j, depth, color, semantic = select_uv(
        i, j, n, depth, color, semantic, device=device
    )
    return i, j, depth, color, semantic


def get_samples(
    H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, c2w, depth, color, semantic, device
):
    """
    added semantic as argument, changed get_samples_uv to get_samples for semantic

    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.

    """
    i, j, sample_depth, sample_color, sample_semantic = get_sample_uv(
        H0, H1, W0, W1, n, depth, color, semantic, device=device
    )
    rays_o, rays_d = get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device)
    return rays_o, rays_d, sample_depth, sample_color, sample_semantic


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj**2 + qk**2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi**2 + qk**2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi**2 + qj**2)
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix

    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(
    stage,
    raw,
    z_vals,
    rays_d,
    occupancy=False,
    device="cuda:0",
    semantic_occupancy_multiplier=10,
):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1.0 - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).float().to(device).expand(dists[..., :1].shape)],
        -1,
    )  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]  # J: should be one-hot encodings in semantic stage
    # J: Calculating alpha from occupancy
    if occupancy:
        if stage != "semantic":
            raw[..., 3] = torch.sigmoid(10 * raw[..., -1])
        else:
            # semantic
            raw[..., -1] = torch.sigmoid(
                semantic_occupancy_multiplier * raw[..., -1]
            )  # TODO: finetune this, it makes the occupancy map sharper when increased
        alpha = raw[..., -1]
    else:
        # J: never enters here in our case
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = (
        alpha.float()
        * torch.cumprod(
            torch.cat(
                [
                    torch.ones((alpha.shape[0], 1)).to(device).float(),
                    (1.0 - alpha + 1e-10).float(),
                ],
                -1,
            ).float(),
            -1,
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3 or #instances)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)

    # Done: add semantic map like rgb_map, maybe change it later <- rgb_map is semantic map in stage semantic
    tmp = z_vals - depth_map.unsqueeze(-1)  # (N_rays, N_samples)
    depth_var = torch.sum(weights * tmp * tmp, dim=1)  # (N_rays)
    return (
        depth_map,
        depth_var,
        rgb_map,
        weights,
    )  # Done: add semantic map like rgb_map, maybe change it later <- rgb_map is semantic map in stage semantic


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1).to(
        device
    )
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0] - bound[0, 0]) / (bound[0, 1] - bound[0, 0])) * 2 - 1.0
    p[:, 1] = ((p[:, 1] - bound[1, 0]) / (bound[1, 1] - bound[1, 0])) * 2 - 1.0
    p[:, 2] = ((p[:, 2] - bound[2, 0]) / (bound[2, 1] - bound[2, 0])) * 2 - 1.0
    return p
