import argparse
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM

import os #J:added


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config( #J:changed it to use our config file including semantics
        args.config, 'configs/nice_slam_sem.yaml' if args.nice else 'configs/imap.yaml')
    
    #----------------------------added for tensorboard writer---------------------------
    num_of_runs = len(os.listdir(cfg["writer_path"])) if os.path.exists(cfg["writer_path"]) else 0
    path = os.path.join(cfg["writer_path"], f'run_{num_of_runs + 1}')
    cfg['writer_path'] = path
    #-----------------------------------------------------------------------------------

    slam = NICE_SLAM(cfg, args)

    slam.run()


if __name__ == '__main__':
    main()
