import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import os
import torch
import argparse
from tqdm import tqdm


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


def get_scene_bounds(depth_paths, poses):
    K = np.loadtxt(
        "/home/rozenberszki/project/wsnsl/Datasets/Scannet/scene0423_02_panoptic/intrinsic/intrinsic_color.txt"
    )
    png_depth_scale = 1000
    glob_min = np.array([np.inf, np.inf, np.inf])
    glob_max = np.array([-np.inf, -np.inf, -np.inf])
    for i, depth in tqdm(enumerate(depth_paths)):
        depth_data = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / png_depth_scale
        depth = np.array(depth_data)
        out = realWorldProject(
            np.array(np.where(depth > 0))[::-1], poses[i], K, torch.from_numpy(depth)
        )
        out_min = np.min(out, axis=1)
        out_max = np.max(out, axis=1)
        glob_min = np.minimum(glob_min, out_min)
        glob_max = np.maximum(glob_max, out_max)
    return glob_min, glob_max


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument(
        "--depth_directory",
        type=str,
        help="Path to depth folder.",
        default="/home/rozenberszki/project/wsnsl/Datasets/Scannet/scene0423_02_panoptic/depth/",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Path to the trajectory file.",
        default="/home/rozenberszki/project/wsnsl/Datasets/Scannet/scene0423_02_panoptic/traj.txt",
    )
    args = parser.parse_args()
    depths_directory = args.depth_directory
    poses = np.loadtxt(args.trajectory).reshape(-1, 4, 4)
    depths = sorted(
        glob.glob(depths_directory + "*.png"),
        key=lambda x: int(x.split("/")[-1].split(".")[0]),
    )
    min, max = get_scene_bounds(depths, poses)
    print(np.concatenate([min[:, None], max[:, None]], axis=1))
    print(min, max)


if __name__ == "__main__":
    main()
