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


class Segmenter(object):

    def __init__(self, slam, cfg, args, zero_pos, store_directory):
        self.store_directory = store_directory
        self.zero_pos = zero_pos
        os.makedirs(f"{store_directory}", exist_ok=True)

        self.is_full_slam = cfg["Segmenter"]["full_slam"]
        self.store_vis = cfg["Segmenter"]["store_vis"]
        self.use_stored = cfg["Segmenter"]["use_stored"]
        self.first_min_area = cfg["mapping"]["first_min_area"]

        """path_to_traj = cfg["data"]["input_folder"] + "/traj.txt"
        self.T_wc = np.loadtxt(path_to_traj).reshape(-1, 4, 4)
        self.T_wc[:, 1:3] *= -1"""

        self.every_frame = cfg["mapping"]["every_frame"]
        # self.slam = slam
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.id_counter = slam.id_counter
        self.idx_mapper = slam.mapping_idx
        self.estimate_c2w_list = slam.estimate_c2w_list
        s = np.ones((4, 4), int)
        s[[0, 0, 1, 1, 2], [1, 2, 0, 3, 3]] *= -1
        self.shift = 1  # s
        self.id_counter = slam.id_counter
        self.idx_mapper = slam.mapping_idx
        # self.idx_coarse_mapper = slam.idx_coarse_mapper

        self.every_frame_seg = cfg["Segmenter"]["every_frame"]
        self.points_per_instance = cfg["mapping"]["points_per_instance"]
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if args is None or args.input_folder is None:
            self.input_folder = cfg["data"]["input_folder"]
        else:
            self.input_folder = args.input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        # self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.frame_reader = get_dataset(
            cfg,
            args,
            cfg["scale"],
            device=cfg["mapping"]["device"],
            tracker=True,
            slam=slam,
        )
        self.n_img = self.frame_reader.n_img
        self.semantic_frames = slam.semantic_frames
        self.idx_segmenter = slam.idx_segmenter
        if not self.is_full_slam:
            self.idx = torch.tensor([self.n_img])
        else:
            self.idx = slam.idx  # Tracking index
            # Segmenter index
        # self.new_id = 0
        self.visualizer = vis.visualizerForIds()
        self.frame_numbers = []
        self.samples = None
        self.deleted = {}
        self.border = (
            cfg["cam"]["crop_edge"]
            if "crop_edge" in cfg["cam"]
            else cfg["Segmenter"]["border"]
        )
        self.num_clusters = cfg["Segmenter"]["num_clusters"]
        self.overlap = cfg["Segmenter"]["overlap"]
        self.relevant = cfg["Segmenter"]["relevant"]
        self.max_id = 0
        self.update = {}
        self.verbose = cfg["Segmenter"]["verbose"]
        self.merging_parameter = cfg["Segmenter"]["merging_parameter"]
        self.hit_percent = cfg["Segmenter"]["hit_percent"]

    def segment_reverse(self, idx):
        assert False
        img = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masksCreated, self.samples = id_generation.createReverseReverseMappingCombined(
            idx,
            self.T_wc,
            self.K,
            self.depth_paths,
            predictor=self.predictor,
            current_frame=img,
            samples=self.samples,
            num_of_clusters=4,
        )
        self.semantic_frames[idx // self.every_frame] = torch.from_numpy(masksCreated)

    def segment_idx(self, idx):
        """img = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"""
        img, depth = self.frame_reader.get_colorAndDepth(idx)
        img = (img.cpu().numpy() * 255).astype(np.uint8)

        masksCreated, s, max_id, update = (
            id_generation.createReverseMappingCombined_area_sort(
                idx,
                self.estimate_c2w_list.cpu() * self.shift,
                self.K,
                depth.cpu(),
                predictor=self.predictor,
                max_id=self.max_id,
                update=self.update,
                points_per_instance=self.points_per_instance,
                current_frame=img,
                samples=self.samples,
                kernel_size=30,  # from 40*40 to 1000
                smallesMaskSize=1000,
                deleted=self.deleted,
                num_of_clusters=self.num_clusters,
                border=self.border,
                overlap_threshold=self.overlap,
                relevant_threshhold=self.relevant,
                every_frame=self.every_frame_seg,
                merging_parameter=self.merging_parameter,
                hit_percent=self.hit_percent,
            )
        )
        self.samples = s
        self.max_id = max_id
        frame = torch.from_numpy(masksCreated)
        self.semantic_frames[idx // self.every_frame_seg] = frame
        return frame

    def segment_idx_forAuto(self, idx):
        """img = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)"""
        img, depth = self.frame_reader.get_colorAndDepth(idx)
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        masksCreated, s, max_id = id_generation.createFrontMappingAutosort(
            idx,
            self.estimate_c2w_list.cpu() * self.shift,
            self.K,
            depth.cpu(),
            self.predictor,
            max_id=self.max_id,
            current_frame=img,
            samples=self.samples,
            smallesMaskSize=1000,
            border=self.border,
        )

        self.samples = s
        self.max_id = max_id

        frame = torch.from_numpy(masksCreated)
        self.semantic_frames[idx // self.every_frame_seg] = frame
        return frame

    def predict_idx(self, idx):
        assert False
        img = cv2.imread(self.color_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masksCreated = id_generation.createReverseMappingCombined_area_sort_predict(
            idx,
            self.T_wc,
            self.K,
            self.depth_paths,
            predictor=self.predictor,
            max_id=self.max_id,
            update=self.update,
            points_per_instance=self.points_per_instance,
            current_frame=img,
            samples=self.samples,
            kernel_size=40,
            every_frame=self.every_frame_seg,
            smallesMaskSize=40 * 40,
            deleted=self.deleted,
            num_of_clusters=self.num_clusters,
            border=self.border,
            overlap_threshold=self.overlap,
            relevant_threshhold=self.relevant,
        )

        self.semantic_frames[idx // self.every_frame] = torch.from_numpy(masksCreated)

    def segment_first(self):
        """color_path = self.color_paths[0]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)"""
        image, depth = self.frame_reader.get_colorAndDepth(0)
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        sam = create_instance_seg.create_sam("cuda")
        masks = sam.generate(image)
        del sam
        torch.cuda.empty_cache()

        ids = id_generation.generateIds(masks, min_area=self.first_min_area)
        print(np.sum(ids == -100) / (ids.shape[0] * ids.shape[1]))
        # visualizerForId = vis.visualizerForIds()
        # visualizerForId.visualize(ids, f'{self.store_directory}/first_segmentation.png')
        self.semantic_frames[0] = torch.from_numpy(ids)
        print(np.sum(ids == -100) / (ids.shape[0] * ids.shape[1]))
        # visualizerForId = vis.visualizerForIds()
        # visualizerForId.visualize(ids, f'{self.store_directory}/first_segmentation.png')
        self.semantic_frames[0] = torch.from_numpy(ids)
        self.frame_numbers.append(0)
        self.max_id = ids.max() + 1

        samplesFromCurrent = backproject.sample_from_instances_with_ids(
            ids, self.max_id, points_per_instance=100
        )
        realWorldSamples = backproject.realWorldProject(
            samplesFromCurrent[:2, :],
            self.zero_pos * self.shift,
            self.K,
            depth.cpu(),
        )
        realWorldSamples = np.concatenate(
            (realWorldSamples, samplesFromCurrent[2:, :]), axis=0
        )
        return realWorldSamples

    def segment_first_ForAuto(self):
        """color_path = self.color_paths[0]
        color_data = cv2.imread(color_path)
        image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)"""
        image, depth = self.frame_reader.get_colorAndDepth(0)
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        sam = create_instance_seg.create_sam_forauto("cuda")
        masks = sam.generate(image)
        del sam
        torch.cuda.empty_cache()

        ids = backproject.generateIds_Auto(masks, min_area=self.first_min_area)
        # visualizerForId = vis.visualizerForIds()
        # visualizerForId.visualize(ids, f'{self.store_directory}/first_segmentation.png')
        self.semantic_frames[0] = torch.from_numpy(ids)
        self.frame_numbers.append(0)
        self.max_id = ids.max() + 1

        samplesFromCurrent = backproject.sample_from_instances_with_ids_area(
            ids, self.max_id, points_per_instance=100
        )
        realWorldSamples = backproject.realWorldProject(
            samplesFromCurrent[:2, :],
            self.zero_pos * self.shift,
            self.K,
            depth.cpu(),
        )
        realWorldSamples = np.concatenate(
            (realWorldSamples, samplesFromCurrent[2:, :]), axis=0
        )
        return realWorldSamples

    def process_keys(self, deleted):
        assert False
        for target in deleted.values():
            if target in deleted.keys():
                update_keys = [key for key, value in deleted.items() if value == target]
                for uk in update_keys:
                    deleted[uk] = deleted[target]
        return deleted

    def process_frames(self, semantic_frames):
        """process the semantic ids such that we have the minimum max(id), number"""
        ids = np.unique(semantic_frames)
        result = semantic_frames.clone()
        for i in range(len(ids)):
            result[semantic_frames == ids[i]] = i
        result[semantic_frames == -100] = -100
        semantic_frames[:, :, :] = result
        return result, len(ids) - 1

    def run(self, max=-1):
        if self.use_stored:
            index_frames = np.arange(0, self.n_img, self.every_frame_seg)
            for index in tqdm(index_frames, desc="Loading stored segmentations"):
                path = os.path.join(self.store_directory, f"seg_{index}.npy")
                self.semantic_frames[index // self.every_frame_seg] = torch.from_numpy(
                    np.load(path).astype(np.int32)
                )
            if self.n_img - 1 % self.every_frame_seg != 0:
                path = os.path.join(self.store_directory, f"seg_{self.n_img - 1}.npy")
                self.semantic_frames[-1] = torch.from_numpy(
                    np.load(path).astype(np.int32)
                )
            self.idx_segmenter[0] = self.n_img
            return self.semantic_frames, self.semantic_frames.max() + 1

        print("segment first frame")
        s = self.segment_first()
        if self.is_full_slam:
            path = os.path.join(self.store_directory, f"seg_{0}.npy")
            # np.save(path, self.semantic_frames[0].numpy())
            self.idx_segmenter[0] = 0
        self.samples = s
        self.predictor = create_instance_seg.create_predictor("cuda")
        if max == -1:
            index_frames = np.arange(
                self.every_frame_seg, self.n_img, self.every_frame_seg
            )
            index_frames = np.concatenate((index_frames, [self.n_img - 1]))
            index_frames_predict = np.setdiff1d(
                np.arange(self.every_frame, self.n_img, self.every_frame), index_frames
            )
        else:
            index_frames = np.arange(self.every_frame_seg, max, self.every_frame_seg)
            index_frames_predict = np.setdiff1d(
                np.arange(self.every_frame, max, self.every_frame), index_frames
            )
        visualizerForId = vis.visualizerForIds()  # for testign
        for idx in tqdm(index_frames, desc="Segmenting frames"):

            # wait for tracker to estimate pose first
            while self.idx[0] < idx:
                # print("segmenter stuck")
                time.sleep(0.1)
            _ = self.segment_idx(idx)
            visualizerForId.visualize(
                self.semantic_frames[idx // self.every_frame_seg],
                path=f"{self.store_directory}/seg_{idx}.png",
            )
            if self.is_full_slam:
                self.idx_segmenter[0] = idx
            # self.plot()
            # print(f'outside samples: {np.unique(self.samples[-1])}')
        if self.n_img - 1 % self.every_frame_seg != 0:
            while self.idx[0] < self.n_img - 1:
                # print("segmenter stuck")
                time.sleep(0.1)
            _ = self.segment_idx(self.n_img - 1)
            self.idx_segmenter[0] = self.n_img - 1

        if not self.is_full_slam:
            for old_instance in self.deleted.keys():
                self.semantic_frames[self.semantic_frames == old_instance] = (
                    self.deleted[old_instance]
                )
            _, self.max_id = self.process_frames(self.semantic_frames)
        # if self.verbose:
        # for i in range(len(self.semantic_frames)):Fself.estim
        make_gif_from_array(
            self.semantic_frames[index_frames // self.every_frame_seg],
            os.path.join(self.store_directory, "segmentation.gif"),
        )

        """for idx in tqdm(index_frames_predict, desc='Predicting frames'):
            print(f'predicting frame {idx}')
            self.predict_idx(idx)"""

        """reverse_index_frames = np.arange(self.n_img-1, -1, -self.every_frame)
        for idx in tqdm(reverse_index_frames, desc='Segmenting frames in reverse'):
            self.segment_reverse(idx)"""
        del self.predictor
        torch.cuda.empty_cache()
        """for idx in tqdm(index_frames_predict, desc='Predicting frames'):
            print(f'predicting frame {idx}')
            self.predict_idx(idx)"""

        """reverse_index_frames = np.arange(self.n_img-1, -1, -self.every_frame)
        for idx in tqdm(reverse_index_frames, desc='Segmenting frames in reverse'):
            self.segment_reverse(idx)"""
        del self.predictor
        torch.cuda.empty_cache()

        # print('unprocessed map: ', self.deleted)
        # self.deleted = self.process_keys(self.deleted)
        # print('preocessed map: ', self.deleted)
        # if self.verbose:
        #    make_gif_from_array(self.semantic_frames, os.path.join(self.store_directory, f'segmentation_full.gif'))
        # print('unprocessed map: ', self.deleted)
        # self.deleted = self.process_keys(self.deleted)
        # print('preocessed map: ', self.deleted)
        # if self.verbose:
        #    make_gif_from_array(self.semantic_frames, os.path.join(self.store_directory, f'segmentation_full.gif'))

        visualizerForId = vis.visualizerForIds()
        # for i in range(len(self.semantic_frames)):
        """for i in range(len(self.semantic_frames)):
            visualizerForId.visualizer(self.semantic_frames[i])"""

        # store the segmentations, such that the dataset class (frame_reader) can read them
        for index in tqdm(
            [0] + list(index_frames),
            desc="Storing segmentations",
        ):
            path = os.path.join(self.store_directory, f"seg_{index}.npy")
            np.save(path, self.semantic_frames[index // self.every_frame_seg].numpy())
        if self.n_img - 1 % self.every_frame_seg != 0:
            path = os.path.join(self.store_directory, f"seg_{self.n_img - 1}.npy")
            np.save(path, self.semantic_frames[-1].numpy())
        if self.store_vis:
            for index in tqdm([0] + list(index_frames), desc="Storing visualizations"):
                path = os.path.join(self.store_directory, f"seg_{index}.png")
                self.visualizer.visualize(
                    self.semantic_frames[index // self.every_frame_seg].numpy(),
                    path=path,
                )
        # EDIT THIS

        return self.semantic_frames, self.max_id

    def runAuto(self, max=-1):
        if self.use_stored:
            index_frames = np.arange(0, self.n_img, self.every_frame_seg)
            for index in tqdm(index_frames, desc="Loading stored segmentations"):
                path = os.path.join(self.store_directory, f"seg_{index}.npy")
                self.semantic_frames[index // self.every_frame_seg] = torch.from_numpy(
                    np.load(path).astype(np.int32)
                )
            if self.n_img - 1 % self.every_frame_seg != 0:
                path = os.path.join(self.store_directory, f"seg_{self.n_img - 1}.npy")
                self.semantic_frames[-1] = torch.from_numpy(
                    np.load(path).astype(np.int32)
                )
            self.idx_segmenter[0] = self.n_img
            return self.semantic_frames, self.semantic_frames.max() + 1

        print("segment first frame")
        s = self.segment_first_ForAuto()
        print("finished segmenting first frame")
        if self.is_full_slam:
            path = os.path.join(self.store_directory, f"seg_{0}.npy")
            # np.save(path, self.semantic_frames[0].numpy())
            self.idx_segmenter[0] = 0
        self.samples = s
        self.predictor = create_instance_seg.create_sam_forauto("cuda")
        # create sam
        if max == -1:
            index_frames = np.arange(
                self.every_frame_seg, self.n_img, self.every_frame_seg
            )
            index_frames_predict = np.setdiff1d(
                np.arange(self.every_frame, self.n_img, self.every_frame), index_frames
            )
        else:
            index_frames = np.arange(self.every_frame_seg, max, self.every_frame_seg)
            index_frames_predict = np.setdiff1d(
                np.arange(self.every_frame, max, self.every_frame), index_frames
            )
        visualizerForId = vis.visualizerForIds()
        for idx in tqdm(index_frames, desc="Segmenting frames"):
            while self.idx[0] < idx:
                # print("segmenter stuck")
                time.sleep(0.1)
            print("start segmenting frame: ", idx)
            self.segment_idx_forAuto(idx)
            print("finished segmenting frame: ", idx)
            visualizerForId.visualize(
                self.semantic_frames[idx // self.every_frame_seg],
                path=f"{self.store_directory}/seg_{idx}.png",
            )
            if self.is_full_slam:
                self.idx_segmenter[0] = idx
            # self.plot()
            # print(f'outside samples: {np.unique(self.samples[-1])}')
        if self.n_img - 1 % self.every_frame_seg != 0:
            while self.idx[0] < self.n_img - 1:
                # print("segmenter stuck")
                time.sleep(0.1)
            _ = self.segment_idx_forAuto(self.n_img - 1)
            self.idx_segmenter[0] = self.n_img - 1

        del self.predictor
        torch.cuda.empty_cache()

        if not self.is_full_slam:
            self.semantic_frames, max_id = self.process_frames(self.semantic_frames)

        # store the segmentations, such that the dataset class (frame_reader) can read them -> outdated
        # maybe the stored segmentations can be used for loading segmentations
        for index in tqdm([0] + list(index_frames), desc="Storing segmentations"):
            path = os.path.join(self.store_directory, f"seg_{index}.npy")
            np.save(path, self.semantic_frames[index // self.every_frame_seg].numpy())

        if self.store_vis:
            for index in tqdm([0] + list(index_frames), desc="Storing visualizations"):
                path = os.path.join(self.store_directory, f"seg_{index}.png")
                self.visualizer.visualize(
                    self.semantic_frames[index // self.every_frame_seg].numpy(),
                    path=path,
                )
        # EDIT THIS

        return self.semantic_frames, self.max_id + 1

    def plot(self):
        data = self.samples.copy()
        data = data[:, data[1] > -2]
        data = self.samples.copy()
        data = data[:, data[1] > -2]
        x = data[0]
        y = data[1]
        z = data[2] * -1
        z = data[2] * -1
        labels = data[3]

        # Create a scatter plot
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        ax = fig.add_subplot(111, projection="3d")
        ax = fig.add_subplot(111, projection="3d")

        # Plot each point with a color corresponding to its label
        unique_labels = np.unique(labels)
        for label in unique_labels:
            indices = np.where(labels == label)
            ax.scatter(x[indices], y[indices], z[indices], s=3)

        # Set axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_ylim((-2, 2))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_ylim((-2, 2))
        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()
