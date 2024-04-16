import glob
import os
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scripts.gifMaker import color_gif_from_array, make_gif_from_array
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import threading
from tqdm import tqdm
from src.utils.datasets import get_dataset
import time

import torch.multiprocessing as mp
from src.utils import backproject, create_instance_seg, id_generation, vis
import argparse
from src.NICE_SLAM import NICE_SLAM
from src import config
import seaborn as sns
from tqdm import tqdm
from src.utils import vis


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file.")
    parser.add_argument("trajectory", type=str, help="Path to the trajectory file.")
    parser.add_argument(
        "--output",
        type=str,
        help="output folder, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        help="input folder, this have higher priority, can overwrite the one in config file",
    )
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument("--nice", dest="nice", action="store_true")
    nice_parser.add_argument("--imap", dest="nice", action="store_false")
    parser.set_defaults(nice=True)

    args = parser.parse_args()
    cfg = config.load_config(  # J:changed it to use our config file including semantics
        args.config, "configs/nice_slam_sem.yaml" if args.nice else "configs/imap.yaml"
    )
    if args.output:
        path = args.output
    os.makedirs(path, exist_ok=True)
    slam = NICE_SLAM(cfg, args)
    slam.set_log_dict(args.checkpoint)
    poses = np.loadtxt(args.trajectory)
    poses = poses.reshape(-1, 4, 4)
    poses = torch.from_numpy(poses).float()
    render_gif(slam, poses, path)


def render_gif(slam, poses, path=None):
    if path is None:
        path = "run_traj/"
    else:
        path = path + "/"
    visualizerForId = vis.visualizerForIds()
    depths, colors, semantics = [], [], []
    for i, c2w in tqdm(enumerate(poses)):
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        depth, _, color, semantic = slam.vis_renderer.render_img(
            slam.shared_c,
            slam.shared_decoders,
            c2w.to("cuda"),
            "cuda",
            stage="visualize",
            gt_depth=None,
        )
        depth_np = depth.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        color_np = np.clip(color_np, 0, 1)
        semantic_np = semantic.detach().cpu().numpy()
        semantic_argmax = np.argmax(semantic_np, axis=2)
        depths.append(depth_np)
        colors.append(color_np)
        semantics.append(semantic_argmax)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(color_np)
        ax[1].imshow(depth_np / np.max(depth_np))
        ax[2], im = visualizerForId.visualize(semantic_argmax, ax=ax[2])
        plt.savefig(f"{path}frame_{i*4}.png")

    depths = np.stack(depths)
    depths /= np.max(depths)

    color_gif_from_array(colors, "test_color.gif")
    color_gif_from_array(depths, "test_depth.gif")
    make_gif_from_array(semantics, "test_semantic.gif")


if __name__ == "__main__":
    main()
