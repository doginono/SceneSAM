import numpy as np


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.zeros((4, 4))
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    T_inv[3, 3] = 1
    return T_inv


def backproject_one_uv(uv, Tf, Tg, K, depthf):
    """map uv to uv from the other frame
       for one pixelpair uv = (u,v) in frame f, we want to find the corresponding uv2 = (u2,v2) in frame g

    Args:
        uv (np.array): _description_
        Tf (np.array): includes both
        Tg (np.array): includes both
        K (np.array): _description_
        depthf (np.array): _description_
        depthg (np.array): _description_

    returns:
        (np.array): shape = (2,) uv coordinates of frame g
    """
    K_inv = np.array(
        [
            [1 / K[0, 0], 0.0, -K[0, 2] / K[0, 0]],
            [0.0, 1 / K[1, 1], -K[1, 2] / K[1, 1]],
            [0.0, 0.0, 1.0],
        ]
    )
    Tg_inv = T_inv(Tg)
    tmp = np.concatenate([uv, np.ones(1)])
    tmp = K_inv @ tmp
    tmp = tmp * depthf[uv[1], uv[0]].numpy()  # real world in camera coordinates
    tmp = np.concatenate([tmp, np.ones(1)])
    tmp = Tf @ tmp  # real world coordinates
    tmp = Tg_inv @ tmp
    tmp = tmp[:3]  # real world coordinates in camera coordinates of g
    zg = tmp[-1]  # zg has to align with the depthg
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:-1]  # uv coordinates of g

    return tmp, zg


def backproject(uv, Tf, Tg, K, depthf):
    """map uv to uv from the other frame
        get a numpy array of uv cordinates of frame f and return the corresponding uv coordinates of frame g

    Args:
        uv (np.array): of shape (2,n) with the pixel coordinates in frame f
        Tf (np.array): includes both
        Tg (np.array): includes both
        K (np.array): _description_
        depthf (np.array): _description_
        depthg (np.array): _description_

    returns:
        (np.array): shape = (2,) uv coordinates of frame g
    """
    K_inv = np.array(
        [
            [1 / K[0, 0], 0.0, -K[0, 2] / K[0, 0]],
            [0.0, 1 / K[1, 1], -K[1, 2] / K[1, 1]],
            [0.0, 0.0, 1.0],
        ]
    )
    Tg_inv = T_inv(Tg)
    tmp = np.concatenate([uv, np.ones((1, uv.shape[1]))])
    tmp = K_inv @ tmp
    tmp = tmp * depthf[uv[1], uv[0]].numpy()  # real world in camera coordinates
    tmp = np.concatenate([tmp, np.ones((1, tmp.shape[1]))])
    tmp = Tf @ tmp  # real world coordinates
    tmp = Tg_inv @ tmp
    tmp = tmp[:3, :]  # real world coordinates in camera coordinates of g
    zg = tmp[-1]  # zg has to align with the depthg
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:2, :]  # uv coordinates of g

    return tmp, zg


def sample_from_instances(masks, points_per_instance=1):
    """samples uv from the instances

    Args:
        masks (numpy.array): numpy array with shape (height, width) and values in [0, len(masks)-1]

    returns:
        uv (numpy.array): shape(2,points_per_instance, len(instances))

    """
    # the ids are a only the ids of the instances
    sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    ids = np.ones(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
            1,
        )
    )
    for i, ann in enumerate(sortedMasks):
        m = ann["segmentation"]
        idsForEachMask = np.concatenate([[i]])
        ids[m] = idsForEachMask

    torch_sampled_indices = torch.zeros(
        (
            2,  # 2D
            points_per_instance,  # number of points per instance
            len(sortedMasks),  # number of instances
        )
    )

    for i in range(len(sortedMasks)):
        labels = np.where(ids == i)
        indices = list(zip(labels[0], labels[1]))
        if len(indices) > 0:  # Check if there are any True pixels
            sampled_indices = np.random.choice(len(indices), points_per_instance)
            torch_sampled_indices[:, :, i] = torch.tensor(
                [indices[j][::-1] for j in sampled_indices]
            ).T
    return torch_sampled_indices.to(torch.int32)
