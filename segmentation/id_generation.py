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
# behind or not check
def createMapping(
    ids1,
    ids2,
    backprojectedSamples,
    samplesFromCurrentMask,
    depth1=zg,  #
    depth2=,
    instance=10,
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

    # 10->12
    # 15->16


def update_current_frame(curr_mask, id2id):
    """update curr_mask according to sampleFromCurrentMask

    Args:
        curr_mask (np.array): (W,H) with ids
        id2id (np.array): shpae (2, #num ids in curr_mask)

    Return:
        np.array (W,H): updated mask
    """
    # for all map[0,:]: curr_mask[curr_mask == map[0,i]] = map[1,i]
    pass


def create_complete_mapping(ids_curr, frames):
    map = []
    for frame in frames:
        for instance in np.unique(np.flatten(ids_curr)):
            map.append(createMapping())
    # map = combineMaps(map)
    # update ids_curr according to map; update_current_frame


def combineMaps(map):
    pass
