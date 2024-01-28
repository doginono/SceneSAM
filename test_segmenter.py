import argparse
import random

import numpy as np
import torch
from scripts.gifMaker import make_gif_from_array
from src import config
from src.NICE_SLAM import NICE_SLAM
from src.Segmenter import Segmenter

import os #J:added
from torch.utils.tensorboard import SummaryWriter #J: added
import yaml #J: added

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
    num_of_runs = len(os.listdir(cfg["data"]['logs'])) if os.path.exists(cfg["data"]['logs']) else 0
    path = os.path.join(cfg["data"]['logs'], f'run_{num_of_runs + 1}')
    cfg["data"]['logs'] = path
    os.makedirs(path, exist_ok=True)
    
    writer = SummaryWriter(path)
    hparams_path = cfg['inherit_from']
    with open(hparams_path, 'r') as file:
        hparams_dict = yaml.safe_load(file)
    yaml_string = yaml.dump(hparams_dict, default_flow_style=False)
    writer.add_text('hparams', yaml_string)
    writer.close()
    print('read in hparams')
    #-----------------------------------------------------------------------------------
    
    overlaps = [0.7, 0.9]
    relevants = [0.3, 0.7]
    hit_percents = [0.3, 0.5, 0.7]
    merging_paramenters = [3, 5, 7]
    for overlap in overlaps: 
        for relevant in relevants:
            for hit_percent in hit_percents:
                for merging_parameter in merging_paramenters:
            
                    '''cfg['Segmenter']['border'] = border
                    cfg['Segmenter']['num_clusters'] = num_cluster'''
                    cfg['Segmenter']['hit_percent'] = hit_percent
                    cfg['Segmenter']['overlap'] = overlap
                    cfg['Segmenter']['relevant'] = relevant
                    cfg['Segmenter']['merging_parameter'] = merging_parameter
                    
                    segmenter = Segmenter(cfg, args, store_directory=os.path.join(cfg['data']['input_folder'], 'segmentation'))
                    frames,_ = segmenter.run()
                    make_gif_from_array(frames, store = f'gif/standard_%{hit_percent}_m{merging_parameter}_o{overlap}_r{relevant}.gif')
    


if __name__ == '__main__':
    main()
