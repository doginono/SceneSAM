import glob
import os
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scripts.gifMaker import make_gif_from_array
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
from src.Segmenter import Segmenter

#hyperparameters to check: smallestmasksize (also first smallest mask), samplePixelsFarther, NormalizePointNumber, depthCondition, border?,
#scenes: 
#also for Replica?, maybe with only smallest mask size

paths = ['scene0423_02_panoptic', 'scene0300_01', 'scene0616_00', 'scene0354_00', 'scene0389_00', 'scene0494_00', 'scene0645_02', 'scene0693_00']
basepath = '/home/rozenberszki/project/wsnsl/configs/ScanNet/'
smallestMaskSizes = [1000, 2000, 5000]
samplePixelFarthers = [2, 5, 8]
normalizePointNumbers = [7]
border = [10]
depthConditions = [0.05, 0.1]
hypers = []
for sms in smallestMaskSizes:
    for spf in samplePixelFarthers:
        for npn in normalizePointNumbers:
            for b in border:
                for dc in depthConditions:
                    hypers.append((sms, spf, npn, b, dc))

for p in paths:
    path = basepath + p + '.yaml'
    args = argparse.Namespace()
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
    args = parser.parse_args(args=[path])
    #args = parser.parse_args(args=['/home/rozenberszki/project/wsnsl/configs/Own/room0.yaml'])
    cfg = config.load_config(  # J:changed it to use our config file including semantics
            args.config, "configs/nice_slam_sem.yaml" if args.nice else "configs/imap.yaml"
        )
    
    for sms, spf, npn, b, dc in hypers:
        parameter_string = f'sms_{sms}_spf_{spf}_npn_{npn}_b_{b}_dc_{dc}'
        cfg['mapping']['first_min_area'] = sms
        cfg['Segmenter']['smallestMaskSize'] = sms
        cfg['Segmenter']['samplePixelFarther'] = spf
        cfg['Segmenter']['normalizePointNumber'] = npn
        cfg['Segmenter']['border'] = b
        cfg['Segmenter']['depthCondition'] = dc
        cfg['Segmenter']['every_frame'] = 2
        cfg['tracking']['gt_camera'] = True
        print('Using GT Camera Pose for tracking.')
        slam = NICE_SLAM(cfg, args)
        frame_reader = get_dataset(cfg, args, cfg["scale"], slam = slam)
        frame_reader.__post_init__(slam)
        zero_pos = frame_reader.poses[0]

        out_path = cfg['data']['output']
        modified_out_path = out_path.replace('output', 'output_segmented')
        modified_out_path = os.path.join(modified_out_path, parameter_string, 'segmentation')
        segmenter = Segmenter(slam, cfg, args, zero_pos, modified_out_path)
        segmenter.estimate_c2w_list = torch.from_numpy(np.concatenate([p[None] for p in segmenter.frame_reader.poses], axis=0)).float()
        #we could also use poses from a checkpoint here.
        segmenter.runAuto()