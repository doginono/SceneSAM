import backproject
import numpy as np
import cv2


def get_ids(masks1, masks2):
    ids1 = generateIds(masks1)
    ids2 = generateIds(masks2)
    samplesFromCurrentMask = sample_from_instances(masks2, ids2, points_per_instance=10)
    backprojectedSamples, _ = backproject.backproject(
        samplesFromCurrentMask[:, :, 0], Tf, Tg, K, Depthf
    )
    backprojectedSamples = backprojectedSamples.astype(int)
    print(len(list(zip(backprojectedSamples[0], backprojectedSamples[1]))))
    for i, sample in enumerate(
        list(zip(backprojectedSamples[0], backprojectedSamples[1]))
    ):
        print(sample)
        if (
            backprojectedSamples[0][i] < 0
            or backprojectedSamples[1][i] < 0
            or backprojectedSamples[0][i] > 679
            or backprojectedSamples[1][i] > 1199
        ):
            ids2[samplesFromCurrentMask[1, i, 0]][samplesFromCurrentMask[0, i, 0]] = -1
            print("sample out of bounds")
            print(sample)
        else:
            print(sample)
            ids2[
                samplesFromCurrentMask[1, i, 0], samplesFromCurrentMask[0, i, 0]
            ] = ids1[sample[1]][sample[0]]
    return backprojectedSamples, samplesFromCurrentMask
