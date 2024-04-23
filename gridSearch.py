import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM
from src.Segmenter import Segmenter

import os  # J:added
from torch.utils.tensorboard import SummaryWriter  # J: added
import yaml  # J: added
from scripts import gifMaker  # J: added
import time
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="input folder, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output folder, this have higher priority, can overwrite the one in config file",
    )
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument("--nice", dest="nice", action="store_true")
    nice_parser.add_argument("--imap", dest="nice", action="store_false")
    parser.set_defaults(nice=True)
    args = parser.parse_args()
    print(args)
    cfg = config.load_config(  # J:changed it to use our config file including semantics
        args.config, "configs/nice_slam_sem.yaml" if args.nice else "configs/imap.yaml"
    )

    if os.path.exists("grid_search"):
        print("'grid_search' directory already exists.")
    else:
        os.makedirs("grid_search/")

    with open("New_Gridsearch.txt", "a") as file:
        file.write(
            "=============================================================================\n New Grid Search\n=============================================================================\n"
        )
    # crop_scales = [0.7, 0.5, 0.4, 0.3]  # [0.3, 0.4, 0.5, 0.6]
    every_frames = [2]
    tracking_it = [50]
    pixels = [5000]
    #mapping iters
    train_iters = [50]
    samplePixelFarther=[i*3 for i in range(4)]
    depthCondition=[0.3, 0.1]
    border=[30]
    normalizePointNumber=[5,10]


    ''' 
    samplePixelFarther: 8
        normalizePointNumber: 5
        depthCondition: 0.1 #0.005 #0.0 # 0.3
        border: 30
        every_frame: 10
        smallestMaskSize: 3000 
    '''
    timings = []
    for it, pixel, every_frame, train_iter, samplePixelFar,condition,bord,normalizePointNum in tqdm(
        zip(tracking_it, pixels, every_frames, train_iters, samplePixelFarther, depthCondition, border, normalizePointNumber),
    ):
        d = {
            "track_iter": it,
            "pixels": pixel,
            "every_frame": every_frame,
            "train_iter": train_iter,
            "samplePixelFarther": samplePixelFar,
            "depthCondition": condition,
            "border": bord,
            "normalizePointNumber": normalizePointNum
        }
        #filename = os.path.basename(args.input_folder)
        #print(filename)
        path = f"grid_search/{args.config.split('/')[-1][:-5]}/logs/run_crop_{it}_{pixel}_{every_frame}_{train_iter}_{samplePixelFar}_{condition}_{bord}_{normalizePointNum}/segmentation"
        cfg["data"]["logs"] = path
        out_path = f"grid_search/{args.config.split('/')[-1][:-5]}/output/crop_{it}_{pixel}_{every_frame}_{train_iter}_{samplePixelFar}_{condition}_{bord}_{normalizePointNum}/segmentation"
        cfg["data"]["output"] = out_path
        store_seg_path = f"grid_search/{args.config.split('/')[-1][:-5]}/output/crop_{it}_{pixel}_{every_frame}_{train_iter}_{samplePixelFar}_{condition}_{bord}_{normalizePointNum}/segmentation"
        cfg["Segmenter"]["store_seg_path"] = store_seg_path
        
        os.makedirs(store_seg_path, exist_ok=True)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        cfg["mapping"]["every_frame"] = every_frame
        cfg["mapping"]["iters"] = train_iter
        cfg["tracking"]["iters"] = it
        cfg["tracking"]["pixels"] = pixel
        cfg["Segmenter"]["samplePixelFarther"] = samplePixelFar
        cfg["Segmenter"]["depthCondition"] = condition
        cfg["Segmenter"]["border"] = bord
        cfg["Segmenter"]["normalizePointNumber"] = normalizePointNum

        """H, W = cfg["cam"]["H"], cfg["cam"]["W"]
        d["crop_scale"] = crop_scale
        cfg["tracking"]["ignore_edge_W"] = int(100 * crop_scale)
        cfg["tracking"]["ignore_edge_H"] = int(100 * crop_scale)
        cfg["cam"]["crop_size"] = (int(H * crop_scale), int(W * crop_scale))
        d["crop_size"] = cfg["cam"]["crop_size"]

        cfg["Segmenter"]["smallestMaskSize"] = np.max(
            (int(1000 * crop_scale * crop_scale), 5)
        )
        cfg["mapping"]["first_min_area"] = np.max(
            (int(1000 * crop_scale * crop_scale), 5)
        )"""
        start = time.time()
        slam = NICE_SLAM(cfg, args)
        slam.run()
        end = time.time()
        d["time"] = end - start
        timings.append(d)
        with open("New_Gridsearch.txt", "a") as file:
            for key in d.keys():
                file.write(f"   {key}: {d[key]}\n")
            file.write("\n")

    print(timings)
    with open("grid_search/timings_crop.yaml", "w") as f:
        yaml.dump(timings, f)


if __name__ == "__main__":
    main()
