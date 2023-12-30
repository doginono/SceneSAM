import cv2
import numpy as np
import torch
from src.utils import create_instance_seg as cis
import glob
import pickle
from tqdm import tqdm

path = "/home/koerner/Project/nice-slam/Datasets/generated/room0/results/frame*.jpg"

paths = sorted(glob.glob(path))
print(paths[:10])

sam = cis.create_sam("cuda")

for path in tqdm(paths):
    color_data = cv2.imread(path)
    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
    mask = sam.generate(color_data)
    store_path = path.replace('frame', 'mask').replace('jpg', 'pkl')
    with open(store_path, 'wb') as f:
        pickle.dump(mask, f)