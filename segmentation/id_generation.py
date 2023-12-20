from backproject import *
import numpy as np
import cv2
import torch

import backproject


def readDepth(filepath):
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    depth_data = depth.astype(np.float32) / 6553.5
    depth_data = torch.from_numpy(depth_data)
    return depth_data


# check one class mapping
def createMapping(
    ids1, ids2, backprojectedSamples, samplesFromCurrentMask, instance=10
):
    points_per_instance = 5
    backprojectedSamples = backprojectedSamples.astype(int)
    # efficient
    # filter out samples outside of image bounds
    condition = (
        (backprojectedSamples[1, :] < 0)
        | (backprojectedSamples[0, :] < 0)
        | (backprojectedSamples[1, :] > 679)
        | (backprojectedSamples[0, :] > 1199)
    )
    filteredBackProj = backprojectedSamples[:, ~condition]

    numOutofBounds = points_per_instance - len(filteredBackProj[0])
    print("numOutofBounds", numOutofBounds)
    mapping = {}
    elementsBackprojected = np.array(
        list(zip(filteredBackProj[0], filteredBackProj[1]))
    )
    elementesCurrentFrame = np.array(
        list(zip(samplesFromCurrentMask[0], samplesFromCurrentMask[1]))
    )
    print(elementesCurrentFrame.shape)
    print(elementesCurrentFrame)
    ids1_elements = ids1[elementsBackprojected[:, 1], elementsBackprojected[:, 0]]
    ids2_elements = ids2[elementesCurrentFrame[:, 1], elementesCurrentFrame[:, 0]]
    print("ids1_elements", ids1_elements)
    print("ids2_elements", ids2_elements)

    # output: instances number-> to one earlier frame
    # np unique mapping with OutofBounds
    return filteredBackProj, numOutofBounds
