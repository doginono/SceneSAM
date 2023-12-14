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
        goal is to use broadcast to calculate uv2 for all samples in one go

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
    Tf_inv = T_inv(Tf)
    Tg_inv = T_inv(Tg)
    tmp = np.concatenate([uv, np.ones(1)])
    tmp = K_inv @ tmp
    tmp = tmp * depthf[uv[1], uv[0]].numpy()  # real world in camera coordinates
    tmp = np.concatenate([tmp, np.ones(1)])
    tmp = Tf_inv @ tmp  # real world coordinates
    tmp = Tg @ tmp
    tmp = tmp[:3]  # real world coordinates in camera coordinates of g
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:-1]  # uv coordinates of g

    return tmp


def sample_from_instances(masks, points_per_instance=1):
    """samples uv from the instances

    Args:
        masks (numpy.array): numpy array with shape (height, width) and values in [0, len(masks)-1]

    returns:
        uv (numpy.array): shape(2,points_per_instance, len(instances))

    """
    ans = {}
    sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=False)
    for i, ann in enumerate(sortedMasks):
        true_indices = np.where(ann["segmentation"])
        indices = list(zip(true_indices[0], true_indices[1]))
        if len(indices) > 0:  # Check if there are any True pixels
            sampled_indices = np.random.choice(len(indices), points_per_instance)
            ans[i] = [indices[j] for j in sampled_indices]
    return ans
