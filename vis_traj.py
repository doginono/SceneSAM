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
from PIL import Image
import json
from src.utils.datasets import ScanNet_Panoptic
from matplotlib import cm


def main():
    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file.")
    #parser.add_argument("trajectory", type=str, help="Path to the trajectory file.")
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
    """poses = np.loadtxt(args.trajectory)
    poses = poses.reshape(-1, 4, 4)
    poses = torch.from_numpy(poses).float()"""
    poses = None
    #basepath = "/home/rozenberszki/project/wsnsl/Datasets/Scannet/scene0423_02_panoptic"
    #render_gif_dataset(cfg, args, slam, cfg['data']['output'])
    render_gif(slam, cfg, cfg['data']['output'])

def load_panoptic_room0():
    basepath = "/home/rozenberszki/project/wsnsl/Datasets/Replica/room0_panoptic/test_results_org_pan"
    poses = np.loadtxt('Datasets/Replica/room0_panoptic/traj_test.txt')
    poses = poses.reshape(-1, 4, 4)
    poses = torch.from_numpy(poses).float()
    gt_depths = sorted(glob.glob(basepath + "/depth*"))
    gt_colors = sorted(glob.glob(basepath + "/frame*"))
    return gt_colors, gt_depths, poses

def load_scannet(basepath, split):
    gt_colors = sorted([path for path in glob.glob(os.path.join(basepath,'color', "*.jpg")) if os.path.basename(path).split('.')[0] in split],
            key=lambda x: int(os.path.basename(x).split('.')[0]),
        )
    gt_depths = sorted([path for path in glob.glob(os.path.join(basepath,'depth', "*.png")) if os.path.basename(path).split('.')[0] in split],
            key=lambda x: int(os.path.basename(x).split('.')[0]),
        )
    pose_paths = sorted([traj for traj in  glob.glob(os.path.join(basepath,'pose', "*.txt")) if os.path.basename(traj).split('.')[0] in split],
            key=lambda x: int(os.path.basename(x).split('.')[0]),
        )
    poses = [np.loadtxt(p) for p in pose_paths]
    poses = torch.from_numpy(np.stack(poses)).float()
    return gt_colors, gt_depths, poses

def render_gif_dataset(cfg, args, slam, path):
    visualizerForId = vis.visualizerForIds()
    frame_reader = ScanNet_Panoptic(cfg, args, 1,slam=slam, split='test')
    frame_reader.__post_init__(slam)
    if path is None:
        store_path = "run_traj_reversed/"
    else:
        store_path = path + "/"
    seg_path = store_path + "/pred_semantics"
    os.makedirs(seg_path, exist_ok=True)
    os.makedirs(store_path + "/all", exist_ok=True)
    depths, colors, semantics = [], [], []
    for i, color_data, depth_data, c2w, _ in tqdm(frame_reader):
        if i % 10 != 0:
            continue
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
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(color_data.cpu().numpy())
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(color_np)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        #ax[1].imshow(color_data.cpu().numpy())
        ax[2].imshow(depth_np / np.max(depth_np), cmap='jet')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        #ax[3].imshow(depth_data.cpu().numpy() / np.max(depth_data.cpu().numpy()))
        ax[3], _ = visualizerForId.visualize(semantic_argmax, ax=ax[3], sep_boarder=True, samplePixelFarther=4)
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        plt.tight_layout()
        plt.savefig(f"{store_path+'/all/'}0_{(i):04d}.png", bbox_inches='tight',pad_inches=0)
        plt.close(fig)
        #Image.fromarray(semantic_argmax.astype(np.uint8)).save(
        #    f"{seg_path}/0_{(i):04d}.png"
        #)

    depths = np.stack(depths)
    depths /= np.max(depths)

    color_gif_from_array(colors, f"{seg_path}/test_color.gif")
    color_gif_from_array(depths, f"{seg_path}/test_depth.gif")
    make_gif_from_array(semantics, f"{seg_path}/test_semantic.gif")

    pass
def render_gif(slam, cfg, path=None):
    if path is None:
        store_path = "run_traj_reversed/"
    else:
        store_path = path + "/"
    seg_path = store_path + "/pred_semantics"
    os.makedirs(seg_path, exist_ok=True)
    os.makedirs(store_path + "/all", exist_ok=True)
    
   
    #split = json.load(open(os.path.join(basepath,"splits.json")))['test']

    #gt_depths = sorted(glob.glob(basepath + "/depth*"))
    #gt_colors = sorted(glob.glob(basepath + "/frame*"))
    #gt_colors, gt_depths, poses = load_scannet(basepath, split)
    gt_colors, gt_depths, poses = load_panoptic_room0()
    visualizerForId = vis.visualizerForIds()
    crop_size = cfg["cam"]["crop_size"] if "crop_size" in cfg["cam"] else None
    edge = cfg["cam"]["crop_edge"] if "crop_edge" in cfg["cam"] else 0
    depths, colors, semantics = [], [], []
    for i, c2w in tqdm(enumerate(poses)):
        if i != 6 and i != 150:
            continue
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1
        # idx, gt_color, gt_depth, gt_pose, gt_semantic = slam.frame_reader[i]
        depth_data = cv2.imread(gt_depths[i], cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / slam.frame_reader.png_depth_scale
        depth_data = torch.from_numpy(depth_data)
        color_path = gt_colors[i]
        color_data = cv2.imread(color_path)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        color_data = torch.from_numpy(color_data)
        if crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], crop_size, mode="bilinear", align_corners=True
            )[0]
            """sns.histplot(depth_data.numpy().reshape(-1))
            plt.title("Depth data before resize")
            plt.show()[384,512]
            print(depth_data.shape, " before resize")"""

            depth_data = F.interpolate(
                depth_data[None, None], crop_size, mode="nearest"
            )[0, 0]
            """print(depth_data.shape, " after resize")

            sns.histplot(depth_data.flatten().reshape(-1))
            plt.title("Depth data after resize")
            plt.show()"""
            color_data = color_data.permute(1, 2, 0).contiguous()
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        depth_data = depth_data.to("cuda")
        color_data = color_data.to("cuda")
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
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        ax[0].imshow(color_data.cpu().numpy())
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(color_np)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        #ax[1].imshow(color_data.cpu().numpy())
        ax[2].imshow(depth_np / np.max(depth_np), cmap='jet')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        #ax[3].imshow(depth_data.cpu().numpy() / np.max(depth_data.cpu().numpy()))
        ax[3], _ = visualizerForId.visualize(semantic_argmax, ax=ax[3], sep_boarder=True, samplePixelFarther=4)
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        plt.tight_layout()
        print(f"{store_path+'/all/'}0_{(i):04d}.png")
        plt.savefig(f"{store_path+'/all/'}0_{(i):04d}.png", bbox_inches='tight',pad_inches=0)
        plt.close(fig)

    depths = np.stack(depths)
    depths /= np.max(depths)

    color_gif_from_array(colors, "test_color.gif")
    color_gif_from_array(depths, "test_depth.gif")
    make_gif_from_array(semantics, "test_semantic.gif")


if __name__ == "__main__":
    main()
