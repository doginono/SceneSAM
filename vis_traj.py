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
    else:
        path = None
    slam = NICE_SLAM(cfg, args)
    slam.set_log_dict(args.checkpoint)
    poses = np.loadtxt(args.trajectory)
    poses = poses.reshape(-1, 4, 4)
    poses = torch.from_numpy(poses).float()
    render_gif(slam, poses, path)


def render_gif(slam, poses, path=None):
    if path is None:
        store_path = "run_traj_reversed/"
    else:
        store_path = path + "/"
    seg_path = store_path + "/pred_semantics"
    os.makedirs(seg_path, exist_ok=True)
    os.makedirs(store_path+'/all', exist_ok=True)
    
    basepath = "/home/rozenberszki/project/wsnsl/Datasets/Replica/room0_panoptic/test_results_org_pan"

    gt_depths = sorted(glob.glob(basepath + "/depth*"))
    gt_colors = sorted(glob.glob(basepath + "/frame*"))
    visualizerForId = vis.visualizerForIds()
    depths, colors, semantics = [], [], []
    for i, c2w in tqdm(enumerate(poses)):
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        # idx, gt_color, gt_depth, gt_pose, gt_semantic = slam.frame_reader[i]
        depth_data = cv2.imread(gt_depths[i], cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / slam.frame_reader.png_depth_scale
        depth_data = torch.from_numpy(depth_data)
        depth_data = depth_data.to("cuda")
        color_path = gt_colors[i]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        H, W = depth_data.shape
        color_data = cv2.resize(image, (W, H))
        color_data = torch.from_numpy(color_data).to("cuda")
        depth, _, color, semantic = slam.vis_renderer.render_img(
            slam.shared_c,
            slam.shared_decoders,
            c2w.to("cuda"),
            "cuda",
            stage="visualize",
            gt_depth=depth_data,
        )
        depth_np = depth.detach().cpu().numpy()
        color_np = color.detach().cpu().numpy()
        color_np = np.clip(color_np, 0, 1)
        semantic_np = semantic.detach().cpu().numpy()
        semantic_argmax = np.argmax(semantic_np, axis=2)
        depths.append(depth_np)
        colors.append(color_np)
        semantics.append(semantic_argmax)
        fig, ax = plt.subplots(1, 5, figsize=(15, 5))
        ax[0].imshow(color_np)
        ax[1].imshow(color_data.cpu().numpy())
        ax[2].imshow(depth_np / np.max(depth_np))
        ax[3].imshow(depth_data.cpu().numpy() / np.max(depth_data.cpu().numpy()))
        ax[4], _ = visualizerForId.visualize(semantic_argmax, ax=ax[4])
        plt.savefig(f"{store_path+'/all/'}frame_{i*4}.png")
        plt.close(fig)
        visualizerForId.visualize(semantic_argmax, path=f"{seg_path}/frame_{i*4}.png")

    depths = np.stack(depths)
    depths /= np.max(depths)

    color_gif_from_array(colors, "test_color.gif")
    color_gif_from_array(depths, "test_depth.gif")
    make_gif_from_array(semantics, "test_semantic.gif")


if __name__ == "__main__":
    main()
