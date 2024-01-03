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
        self.idx = slam.idx_segmenter
        self.semantic_frames = slam.semantic_frames
        self.id_counter = slam.id_counter
        self.idx_mapper = slam.idx_mapper
        self.idx_coarse_mapper = slam.idx_coarse_mapper
        self.n_img = slam.n_img
        self.every_frame = cfg['mapping']['every_frame']
        self.points_per_instance = cfg['mapping']['points_per_instance']
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.mask_paths = sorted(glob.glob(f'{self.input_folder}/results/mask*.pkl')) #only needed while not running normal sam


    def segment(self):
        idx = self.idx[0].clone()
        color_path = self.color_paths[idx]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        """sam = create_instance_seg.create_sam('cuda')
        print("start sam by ")
        masks = sam.generate(image)"""
        with open(self.mask_paths[idx], 'rb') as f:
            masks = pickle.load(f)
        print("end sam")
            
        semantic_data = id_generation.generateIds(masks)
        self.id_counter[0] = semantic_data.max() +1 
        print("id_counter: ", self.id_counter.item())
        #self.semantic_frames[index] = semantic_data
        self.semantic_frames[idx//self.every_frame]=torch.from_numpy(semantic_data)
        #TODO visualize semantic_data and store to file
        #del sam
        print(f'idx = {idx} with segmentation {np.unique(semantic_data)}')
        del masks
        print("segmentation done")
        self.idx[0] = idx + self.every_frame

    def run(self):
        while(True):
            if self.idx.item() + self.every_frame > self.n_img-1:
                return
            
            while(self.idx.item()>self.idx_mapper.item() or self.idx.item()>self.idx_coarse_mapper.item()):
                time.sleep(0.1)
            print("start segmenting")
            self.segment()
            

