import glob
import os
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import threading
from tqdm import tqdm

import torch.multiprocessing as mp
from src.utils import backproject, create_instance_seg, id_generation, vis

class Segmenter(object):

    def __init__(self,cfg, args, store_directory):
        self.store_directory = store_directory
        os.makedirs(f'{store_directory}', exist_ok=True)

        self.store_vis = cfg['Segmenter']['store_vis']
        self.use_stored = cfg['Segmenter']['use_stored']

        self.mask_generator = cfg['Segmenter']['mask_generator']
        self.first_min_area = cfg['mapping']['first_min_area']
        #self.idx = slam.idx_segmenter
        path_to_traj = cfg['data']['input_folder']+'/traj.txt'
        self.T_wc = np.loadtxt(path_to_traj).reshape(-1, 4, 4)
        self.T_wc[:,1:3] *= -1

        self.every_frame = cfg['mapping']['every_frame']
        #self.slam = slam
        #self.semantic_frames = slam.semantic_frames

        #self.id_counter = slam.id_counter
        #self.idx_mapper = slam.idx_mapper
        #self.idx_coarse_mapper = slam.idx_coarse_mapper
        
        self.every_frame = cfg['mapping']['every_frame']
        self.points_per_instance = cfg['mapping']['points_per_instance']
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if args is None or args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.input_folder}/results/depth*.png')) 

        self.n_img = len(self.color_paths)
        self.semantic_frames = torch.from_numpy(np.zeros((self.n_img//self.every_frame, self.H, self.W))).int()
        
        #self.new_id = 0
        self.visualizer = vis.visualizerForIds()
        self.frame_numbers = []
        self.samples = None
        self.deleted = {}
        self.border = cfg['Segmenter']['border']
        self.num_clusters = cfg['Segmenter']['num_clusters']
        self.overlap = cfg['Segmenter']['overlap']
        self.relevant = cfg['Segmenter']['relevant']
        self.max_id = 0
        self.update = {}

    '''def update(self, semantic_data, id_counter, index):
    
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
            #verbose=False
        )
        semantic_data = id_generation.update_current_frame(semantic_data, map)
        print(f"update id_counter from {self.id_counter[0]} to {id_counter}")
        self.id_counter[0] = id_counter
        return semantic_data'''

    '''def segment(self):
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
        semantic_data = backproject.generateIds(masks, min_area=self.first_min_area)
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
    '''
    def segment_reverse(self,idx):
        img  = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masksCreated,self.samples = id_generation.createReverseReverseMappingCombined(idx, self.T_wc, self.K, self.depth_paths, predictor=self.predictor, current_frame=img,samples=self.samples,num_of_clusters=4)
        self.semantic_frames[idx//self.every_frame]=torch.from_numpy(masksCreated)
        
    def segment_idx(self,idx):
        img  = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masksCreated, s, max_id, update = id_generation.createReverseMappingCombined_area_sort(idx, self.T_wc, self.K, self.depth_paths, 
                                                                     predictor=self.predictor,
                                                                     max_id=self.max_id,
                                                                     update=self.update,
                                                                     points_per_instance=self.points_per_instance, 
                                                                     current_frame=img, samples=self.samples, 
                                                                     kernel_size=40,smallesMaskSize=40*40, 
                                                                     deleted = self.deleted,
                                                                     num_of_clusters=self.num_clusters,
                                                                     border=self.border,
                                                                     overlap_threshold=self.overlap,
                                                                     relevant_threshhold=self.relevant)
        self.samples = s
        self.max_id = max_id
        
        self.semantic_frames[idx//self.every_frame]=torch.from_numpy(masksCreated)
        

    def segment_first(self):
        color_path = self.color_paths[0]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        sam = create_instance_seg.create_sam('cuda')
        masks = sam.generate(image)
        del sam
        torch.cuda.empty_cache()
       


        ids = id_generation.generateIds(masks, min_area=self.first_min_area)
        visualizerForId = vis.visualizerForIds()
        visualizerForId.visualizer(ids)
        self.semantic_frames[0]=torch.from_numpy(ids)
        self.frame_numbers.append(0)
        self.max_id = ids.max() +1

        samplesFromCurrent = backproject.sample_from_instances_with_ids(
            ids,
            self.max_id,
            points_per_instance=100
        )
        realWorldSamples = backproject.realWorldProject(samplesFromCurrent[:2,:], self.T_wc[0], self.K, id_generation.readDepth(self.depth_paths[0]) )
        realWorldSamples = np.concatenate((realWorldSamples, samplesFromCurrent[2:,:]), axis = 0)
        return realWorldSamples

    def process_keys(self, deleted):
        for target in deleted.values():
            if target in deleted.keys():
                update_keys = [key for key, value in deleted.items() if value == target]
                for uk in update_keys:
                    deleted[uk] = deleted[target]
        return deleted

    def run(self, max = -1):
        if self.use_stored:
            index_frames = np.arange(0, self.n_img, self.every_frame)
            for index in tqdm(index_frames, desc='Loading stored segmentations'):
                path = os.path.join(self.store_directory, f'seg_{index}.npy')
                self.semantic_frames[index//self.every_frame] = torch.from_numpy(np.load(path).astype(np.int32))
            return

        #----------end zero frame------------------
        if self.mask_generator:
            while(True):
                if self.idx.item() + self.every_frame > self.n_img-1:
                    return
                
                while(self.idx.item()>self.idx_mapper.item() or self.idx.item()>self.idx_coarse_mapper.item()):
                    time.sleep(0.1)
                print("start segmenting")
                #self.slam.to_cpu()
                torch.cuda.empty_cache()
                self.segment()
        else:
            print('segment first frame')
            s = self.segment_first()
            self.samples = s
            self.predictor = create_instance_seg.create_predictor('cuda')
            if max == -1:
                index_frames = np.arange(self.every_frame, self.n_img, self.every_frame)
            else:
                index_frames = np.arange(self.every_frame, max, self.every_frame)
            for idx in tqdm(index_frames, desc='Segmenting frames'):
                self.segment_idx(idx)
                #print(f'outside samples: {np.unique(self.samples[-1])}')
                
            """reverse_index_frames = np.arange(self.n_img-1, -1, -self.every_frame)
            for idx in tqdm(reverse_index_frames, desc='Segmenting frames in reverse'):
                self.segment_reverse(idx)"""
            del self.predictor
            torch.cuda.empty_cache()

            #print('unprocessed map: ', self.deleted)
            #self.deleted = self.process_keys(self.deleted)
            #print('preocessed map: ', self.deleted)
            for old_instance in self.deleted.keys():
                self.semantic_frames[self.semantic_frames == old_instance] = self.deleted[old_instance]

            visualizerForId = vis.visualizerForIds()
            #for i in range(len(self.semantic_frames)):
            """for i in range(len(self.semantic_frames)):
                visualizerForId.visualizer(self.semantic_frames[i])"""

            #store the segmentations, such that the dataset class (frame_reader) can read them
            for index in tqdm([0]+list(index_frames), desc = 'Storing segmentations'):
                path = os.path.join(self.store_directory, f'seg_{index}.npy')
                np.save(path, self.semantic_frames[index//self.every_frame].numpy())


            if self.store_vis:
                for index in tqdm([0]+list(index_frames), desc = 'Storing visualizations'):
                    path = os.path.join(self.store_directory, f'seg_{index}.png')
                    self.visualizer.visualize(self.semantic_frames[index//self.every_frame].numpy(), path = path)

        return self.semantic_frames, torch.max(self.semantic_frames)
                    
        
            

