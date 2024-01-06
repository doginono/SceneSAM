import glob
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import threading

import torch.multiprocessing as mp
from src.utils import backproject, create_instance_seg, id_generation

class Segmenter(object):

    def __init__(self,cfg, args, slam):
        self.min_area = cfg['mapping']['min_area']
        self.idx = slam.idx_segmenter
        self.T_wc = slam.T_wc
        self.slam = slam
        self.semantic_frames = slam.semantic_frames
        self.id_counter = slam.id_counter
        self.idx_mapper = slam.idx_mapper
        self.idx_coarse_mapper = slam.idx_coarse_mapper
        self.n_img = slam.n_img
        self.every_frame = cfg['mapping']['every_frame']
        self.points_per_instance = cfg['mapping']['points_per_instance']
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/results/depth*.png')) 


    def update(self, semantic_data, id_counter, index):
    
        map , id_counter= id_generation.create_complete_mapping_of_current_frame(
            semantic_data,
            index,
            np.arange(index)[0:(index-1):self.every_frame],  # Corrected slice notation
            self.T_wc,
            self.K, 
            self.depth_paths,
            self.semantic_frames,
            id_counter,
            points_per_instance=self.points_per_instance,  # Corrected parameter name
            verbose=False
        )
        semantic_data = id_generation.update_current_frame(semantic_data, map)
        print(f"update id_counter from {self.id_counter[0]} to {id_counter}")
        self.id_counter[0] = id_counter
        return semantic_data

    def segment(self):
        idx = self.idx[0].clone()
        print("called segment on idx ", idx)
        color_path = self.color_paths[idx]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        print("initialize sam")
        sam = create_instance_seg.create_sam('cuda')
        print("start mask generation")
        masks = sam.generate(image)
        """with open(self.mask_paths[idx], 'rb') as f:
            masks = pickle.load(f)"""
        print("end sam")
        del sam
        semantic_data = id_generation.generateIds(masks, min_area=self.min_area)
        if idx ==0:
            self.id_counter[0] = semantic_data.max() +1 
        else:
            id_counter = self.id_counter[0].clone()
            semantic_data = self.update(semantic_data, id_counter, idx)
        

        self.semantic_frames[idx//self.every_frame]=torch.from_numpy(semantic_data)
        #TODO visualize semantic_data and store to file
        print(f'idx = {idx} with segmentation {np.unique(semantic_data)}')
        del masks
        torch.cuda.empty_cache()
        print("segmentation done and cleared memory")
        self.idx[0] = idx + self.every_frame
        
        

    def run(self):
        while(True):
            if self.idx.item() + self.every_frame > self.n_img-1:
                return
            
            while(self.idx.item()>self.idx_mapper.item() or self.idx.item()>self.idx_coarse_mapper.item()):
                time.sleep(0.1)
            print("start segmenting")
            self.slam.to_cpu()
            torch.cuda.empty_cache()
            self.segment()
            

