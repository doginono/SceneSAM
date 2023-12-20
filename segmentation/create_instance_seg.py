import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import backproject


def instance_encoding2file(encoding, path):
    """_summary_

    Args:
        encoding (np.array): with shape (height, width) and values in [0, len(masks)-1]
        path (str): path to save the encoding
    """
    np.save(path, encoding)
    
#TODO implement in nice-slam later
def create_sam():
    """_summary_

    Returns:
        SamAutomaticMaskGenerator: An SamAutomaticMaskGenerator object
    """
    sam_checkpoint = "/home/julius/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
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
    
    save_path = os.path.join(store_directory, img_path.split("/")[-1].replace("frame","seg").replace("jpg", "npy"))
    print(save_path)
    np.save(save_path, instances)
        
    

def create_id(image, sam):
    image = cv2.imread(image)
    print(image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = sam.generate(image)
    return backproject.generateIds(masks)