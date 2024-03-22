import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from src.utils import backproject


def instance_encoding2file(encoding, path):
    """_summary_

    Args:
        encoding (np.array): with shape (height, width) and values in [0, len(masks)-1]
        path (str): path to save the encoding
    """
    np.save(path, encoding)


def create_predictor(device="cuda"):
    sam_checkpoint = "/home/rozenberszki/project/wsnsl/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


# TODO implement in nice-slam later
def create_sam(device):
    """_summary_

    Returns:
        SamAutomaticMaskGenerator: An SamAutomaticMaskGenerator object
    """
    # sam_checkpoint = "/home/koerner/Project/nice-slam/sam/sam_vit_b_01ec64.pth"
    sam_checkpoint = "/home/rozenberszki/project/wsnsl/sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
    )

    return mask_generator

def create_sam_forauto(device):
    """_summary_

    Returns:
        SamAutomaticMaskGenerator: An SamAutomaticMaskGenerator object
    """
    #sam_checkpoint = "/home/koerner/Project/nice-slam/sam/sam_vit_b_01ec64.pth"
    sam_checkpoint = '/home/koerner/Project/nice-slam/sam/sam_vit_h_4b8939.pth'
    model_type = "vit_h"


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    
        
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=16, pred_iou_thresh=0.9, stability_score_thresh=0.9, crop_nms_thresh=0.2,box_nms_thresh=0.4,crop_n_layers=0, crop_n_points_downscale_factor=2, min_mask_region_area=10000)


    return mask_generator

def create_instance_seg(img_path, store_directory, mask_generator):
    """combines the functions in this file to create an instance segmentation of an image and save it to a file
        and return it

    Args:
        img_path (str): path to image file
        store_directory (str): path to directory where the instance segmentation should be saved,
        mask_generator (SamAutomaticMaskGenerator): An SamAutomaticMaskGenerator object,

    """
    instances = create_id(img_path, mask_generator)

    save_path = os.path.join(
        store_directory,
        img_path.split("/")[-1].replace("frame", "seg").replace("jpg", "npy"),
    )
    print(save_path)
    np.save(save_path, instances)


def create_id(image, sam):
    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = sam.generate(image)
    return backproject.generateIds(masks)
