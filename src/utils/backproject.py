import numpy as np
import torch
import cv2

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
    tmp = (
        tmp * depthf[uv[1].long(), uv[0].long()].numpy()
    )  # real world in camera coordinates
    tmp = np.concatenate([tmp, np.ones((1, tmp.shape[1]))])
    tmp = Tf @ tmp  # real world coordinates

    tmp = Tg_inv @ tmp
    tmp = tmp[:3, :]  # real world coordinates in camera coordinates of g
    zg = tmp[-1]  # zg has to align with the depthg
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:2, :]  # uv coordinates of g

    return tmp.astype(int), zg  # 1D array for each element of tmp


# First part of the Backprojection
def realWorldProject(uv, Tf, K, depthf):
    K_inv = np.array(
        [
            [1 / K[0, 0], 0.0, -K[0, 2] / K[0, 0]],
            [0.0, 1 / K[1, 1], -K[1, 2] / K[1, 1]],
            [0.0, 0.0, 1.0],
        ]
    )

    tmp = np.concatenate([uv, np.ones((1, uv.shape[1]))])
    tmp = K_inv @ tmp
    if isinstance(uv, np.ndarray):
        tmp = tmp * depthf[uv[1].astype(np.int64), uv[0].astype(np.int64)].numpy()
    else:
        tmp = (
            tmp * depthf[uv[1].long(), uv[0].long()].numpy()
        )  # real world in camera coordinates
    tmp = np.concatenate([tmp, np.ones((1, tmp.shape[1]))])
    tmp = Tf @ tmp  # real world coordinates
    tmp = tmp[:3, :]  # real world coordinates

    return tmp


# Later Part of backprojection
def camProject(samples, Tg, K):
    Tg_inv = T_inv(Tg)
    tmp = samples[:3, :]
    tmp = np.concatenate([tmp, np.ones((1, tmp.shape[1]))])
    tmp = Tg_inv @ tmp
    tmp = tmp[:3, :]  # real world coordinates in camera coordinates of g
    zg = tmp[-1]  # zg has to align with the depthg
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:2, :]  # uv coordinates of g
    tmp = np.concatenate([tmp, samples[3:, :]])
    return tmp.astype(int), zg


def camProjectBoundingBoxes(bbox_1, Tg, K):
    Tg_inv = T_inv(Tg)
    tmp = bbox_1
    # print(tmp.shape)
    tmp = np.concatenate([tmp, np.ones((1, tmp.shape[1]))])
    tmp = Tg_inv @ tmp
    tmp = tmp[:3, :]  # real world coordinates in camera coordinates of g
    zg = tmp[-1]  # zg has to align with the depthg
    tmp = tmp / tmp[-1]
    tmp = K @ tmp
    tmp = tmp[:2, :]  # uv coordinates of g
    return tmp.astype(int), zg


def frontProject(uv, Tf, Tg, K, depthf):
    return backproject(uv, Tg, Tf, K, depthf)


def sample_from_instances(ids, numberOfMasks, points_per_instance=1):
    """samples uv from the instances

    Args:
        masks (numpy.array): numpy array with shape (height, width) and values in [0, len(masks)-1]

    returns:
        uv (numpy.array): shape(2,points_per_instance, len(instances))

    """
    torch_sampled_indices = torch.zeros(
        (
            2,  # 2D
            points_per_instance,  # number of points per instance
            numberOfMasks,  # number of instances
        )
    )

    for i in range(numberOfMasks):
        labels = np.where(ids == i)
        indices = list(zip(labels[0], labels[1]))
        if len(indices) > 0:  # Check if there are any True pixels
            sampled_indices = np.random.choice(len(indices), points_per_instance)

            torch_sampled_indices[:, :, i] = torch.tensor(
                [indices[j][::-1] for j in sampled_indices]
            ).T
    return torch_sampled_indices.to(torch.int32)


# id list better

# NEW Variable normalizePointNumber need to define it according to the dimension of the image
def sample_from_instances_with_ids_area(
    ids, normalizePointNumber=50
):
    tensors = []

    temp = np.unique(ids)[1:]
    for i, element in enumerate(list(temp.astype(int))):
        if element >= 0:
            '''mask= ids==element
            kernel = np.ones((samplePixelFarther, samplePixelFarther), np.uint8)
            mask= cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            labels = np.where(mask)'''
            labels = np.where(ids == element)
            indices = list(zip(labels[0], labels[1]))
            points_per_instance = len(indices)
            # points_per_instance=int(2*np.log2(points_per_instance))
            """points_per_instance = np.max(
                (points_per_instance // (5 * 5), min_points)
            )"""
            points_per_instance = points_per_instance // (normalizePointNumber * normalizePointNumber) # new parameter define
            if (
                len(indices) > points_per_instance
                and len(indices) > 1
                and points_per_instance > 1
            ):  # Check if there are any True pixels
                sampled_indices = np.linspace(
                    0, len(indices) - 1, points_per_instance, dtype=int
                )
                sampled_tensor = torch.tensor(
                    [indices[j][::-1] for j in sampled_indices]
                ).T
                element_tensor = torch.full((sampled_tensor.shape[1],), element)

                element_tensor = element_tensor.unsqueeze(0)

                tensors.append(torch.cat((sampled_tensor, element_tensor), axis=0))
    if len(tensors) == 0:
        return torch.zeros((2, 0, 0)).to(torch.int32)
    else:
        torch_sampled_indices = torch.cat(tensors, axis=1)
        return torch_sampled_indices.to(torch.int32)


def sample_from_instances_with_ids(ids, numberOfMasks, points_per_instance=1):
    """samples uv from the instances

    Args:
        masks (numpy.array): numpy array with shape (height, width) and values in [0, len(masks)-1]

    returns:
        uv (numpy.array): shape(2,points_per_instance, len(instances))

    """
    tensors = []

    temp = np.unique(ids)[1:]
    for i, element in enumerate(list(temp.astype(int))):
        if element >= 0:
            labels = np.where(ids == element)
            indices = list(zip(labels[0], labels[1]))
            if len(indices) > points_per_instance:  # Check if there are any True pixels
                sampled_indices = np.linspace(
                    0, len(indices) - 1, points_per_instance, dtype=int
                )
                sampled_tensor = torch.tensor(
                    [indices[j][::-1] for j in sampled_indices]
                ).T
                element_tensor = torch.full((sampled_tensor.shape[1],), element)

                element_tensor = element_tensor.unsqueeze(0)

                tensors.append(torch.cat((sampled_tensor, element_tensor), axis=0))

    torch_sampled_indices = torch.cat(tensors, axis=1)
    return torch_sampled_indices.to(torch.int32)


def generateIds(masks, min_area=1000):
    """sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    ids = np.ones(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
            1,
        )
    )
    # maybe more efficient
    # first frame has 85 instances so not too bad
    for i, ann in enumerate(sortedMasks):
        m = ann["segmentation"]
        idsForEachMask = np.concatenate([[i]])
        ids[m] = idsForEachMask
    return ids.squeeze().astype(np.int32)"""
    sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=False)
    if min_area > 0:
        sortedMasks = [mask for mask in sortedMasks if mask["area"] > min_area]
    ids = np.full(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
        ),
        -100,
    )
    for i, ann in enumerate(sortedMasks):
        m = ann["segmentation"]
        ids[m] = i
    unique_ids, counts = np.unique(ids, return_counts=True)
    for i in range(len(unique_ids)):
        if counts[i] < min_area:
            ids[ids == unique_ids[i]] = -100
    return ids


def generateIds_Auto(masks, depth, min_area=1000, samplePixelFarther=4):
    sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    if min_area > 0:
        sortedMasks = [mask for mask in sortedMasks if mask["area"] > min_area]
    ids = np.full(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
        ),
        -100,
    )
    copyOfIds = np.full(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
        ),
        -100,
    )
    for i, ann in enumerate(sortedMasks):
        m = ann["segmentation"]
        ids[m] = i
    print(np.unique(ids))
    unique_ids, counts = np.unique(ids, return_counts=True)
    
    for i in unique_ids:
        mask = ids == i
        #print(np.sum(mask))
        kernel = np.ones((samplePixelFarther, samplePixelFarther), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        #print(np.sum(mask))
        label= np.where(mask)
        copyOfIds[label] = i
        

    unique_ids, counts = np.unique(copyOfIds, return_counts=True)
    for i in range(len(unique_ids)):
        if counts[i] < min_area:
            copyOfIds[copyOfIds == unique_ids[i]] = -100

    return copyOfIds


def generateIdsNew(masks):
    sortedMasks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    ids = np.ones(
        (
            sortedMasks[0]["segmentation"].shape[0],
            sortedMasks[0]["segmentation"].shape[1],
            1,
        )
    )
    # maybe more efficient
    # first frame has 85 instances so not too bad
    bbox = {}
    for i, ann in enumerate(sortedMasks):
        m = ann["segmentation"]
        idsForEachMask = np.concatenate([[i]])
        ids[m] = idsForEachMask
        bbox[i] = ann["bbox"]
    return ids.squeeze().astype(np.int32), bbox
