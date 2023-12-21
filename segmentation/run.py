import backproject
import id_generation
import numpy as np
import glob
import cv2
import torch
import matplotlib.pyplot as plt
import vis


def run():
    visulizer = vis.visualizerForIds()
    path_to_traj = "/home/julius/Project/nice-slam/Datasets/Replica/room0/traj.txt"
    T_wc = np.loadtxt(path_to_traj).reshape(-1, 4, 4)

    # color_paths = sorted(glob.glob('/home/julius/Project/nice-slam/Datasets/Replica/room0/results/frame*.jpg'))
    depth_paths = sorted(
        glob.glob(
            "/home/julius/Project/nice-slam/Datasets/Replica/room0/results/depth*.png"
        )
    )
    seg_paths = sorted(
        glob.glob("/home/julius/Project/nice-slam/segmentation/data/room0/seg*.npy")
    )

    K = np.array([[600, 0.0, 599.5], [0.0, 600, 339.5], [0.0, 0.0, 1.0]])
    seg_path = "/home/julius/Project/nice-slam/segmentation/data/room0"
    first = np.load(
        "/home/julius/Project/nice-slam/segmentation/data/room0/seg000000.npy"
    )
    id_counter = len(np.unique(first))
    store_path = "/home/julius/Project/nice-slam/segmentation/our_gen/room0"
    np.save(store_path + "/seg000000.npy", first)

    frame_numbers = [0]
    every_frame = 5
    ppi = 100
    segmentations = []
    segmentations.append(first)
    id_counter = len(np.unique(first))
    every_frame = 5

    for i in range(0, 20, every_frame):
        print(f"Mapping frame {i + every_frame}")
        ids_curr = np.load(seg_paths[int(i / every_frame)])
        map, id_counter = id_generation.create_complete_mapping_of_current_frame(
            ids_curr,
            5,
            frame_numbers,
            T_wc,
            K,
            depth_paths,
            segmentations,
            id_counter,
            points_per_instance=ppi,
        )
        ids = id_generation.update_current_frame(ids_curr, map)
        segmentations.append(ids)
        # save output
        frame_numbers.append(i)

    for i, segment in enumerate(segmentations):
        visulizer.visualizer(segment, f"Frame {i*every_frame}")
