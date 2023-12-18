from backproject import *
import numpy as np
import cv2
import torch

import backproject



def readDepth(filepath):
    depth=cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    depth_data = depth.astype(np.float32) / 6553.5
    depth_data = torch.from_numpy(depth_data)
    return depth_data

def get_ids(masks1, masks2, instance=10):
    ids1 = generateIds(masks1)
    ids2 = generateIds(masks2)
    samplesFromCurrentMask = sample_from_instances(masks2, ids2, points_per_instance=5)

    backprojectedSamples, _ = backproject.backproject(
        samplesFromCurrentMask[:, :, instance], Tf, Tg, K, Depthf
    )
    backprojectedSamples = backprojectedSamples.astype(int)

    backProjectedIds = []
    for i, backProjsample in enumerate(
        list(zip(backprojectedSamples[0], backprojectedSamples[1]))
    ):
        if (
            backProjsample[1] < 0
            or backProjsample[0] < 0
            or backProjsample[1] > 679
            or backProjsample[0] > 1199
        ):
            print("Outside of image bounds:", backProjsample)
            backProjectedIds.append(-1)
        else:
            print("inside of image bounds:", backProjsample)
            backProjectedIds.append(ids1[backProjsample[1]][backProjsample[0]])
    return (
        backprojectedSamples,
        samplesFromCurrentMask[:, :, instance],
        backProjectedIds,
    )

Depthg = readDepth(path_to_frames + "depth000000.png")
Depthf = readDepth(path_to_frames + "depth000020.png")
K = np.array([[600, 0.0, 599.5], [0.0, 600, 339.5], [0.0, 0.0, 1.0]])
Tg = T_wc[0]
Tf = T_wc[20]