import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM
from src.Segmenter import Segmenter
from scripts.gifMaker import make_gif_from_array

import os  # J:added
from torch.utils.tensorboard import SummaryWriter  # J: added
import yaml  # J: added


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

    cfg = config.load_config(  # J:changed it to use our config file including semantics
        args.config, "configs/nice_slam_sem.yaml" if args.nice else "configs/imap.yaml"
    )

    # ----------------------------added for tensorboard writer---------------------------
    num_of_runs = (
        len(os.listdir(cfg["data"]["logs"]))
        if os.path.exists(cfg["data"]["logs"])
        else 0
    )
    path = os.path.join(cfg["data"]["logs"], f"run_{num_of_runs + 1}")
    cfg["data"]["logs"] = path
    os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(path)
    hparams_path = cfg["inherit_from"]
    with open(hparams_path, "r") as file:
        hparams_dict = yaml.safe_load(file)
    yaml_string = yaml.dump(hparams_dict, default_flow_style=False)
    writer.add_text("hparams", yaml_string)
    writer.close()
    print("read in hparams")
    slam = NICE_SLAM(cfg, args)
    # -----------------------------------------------------------------------------------
    segmenter = Segmenter(
        slam,
        cfg,
        args,
        slam.frame_reader.get_zero_pose(),
        store_directory=os.path.join(cfg["data"]["output"], "segmentation"),
    )
    # TODO
    semanticFrames, _ = segmenter.runAuto()
    make_gif_from_array(
        semanticFrames,
        store=os.path.join(cfg["data"]["output"], "segmentation", "gif.gif"),
        duration=300,
    )


if __name__ == "__main__":
    main()
