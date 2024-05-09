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
#python psnr.py configs/Own/room0_panoptic.yaml output/Own/room0_panoptic/ckpts/ef2_ruunAuto_00674.tar
#python psnr.py configs/ScanNet/scene0423_02_panoptic.yaml output/scannet/track_scene0423_02_panoptic/ckpts/plot_paper_00683.tar

def main():
    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file.")
    #parser.add_argument("trajectory", type=str, help="Path to the trajectory file.")
    parser.add_argument("mesh", type = str, help = "Path to the out mesh file")
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
    keyframe_dict = slam.set_log_dict(args.checkpoint)
    slam.mesher.confidence_threshold = 0.9
    slam.mesher.get_mesh(
                            args.mesh + ".ply",
                            slam.shared_c,
                            slam.shared_decoders,
                            keyframe_dict,
                            slam.estimate_c2w_list,  
                            1999,
                            'cuda',
                            show_forecast=cfg["meshing"]["mesh_coarse_level"],
                            clean_mesh=cfg["meshing"]["clean_mesh"],
                            get_mask_use_all_frames=False,
                            color=True,
                            semantic=False,
                        ) 
 
if __name__ == "__main__":
    main() 