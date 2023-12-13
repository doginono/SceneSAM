import numpy as np
import cv2
import os




def masks2encoding(masks):
    """_summary_

    Args:
        masks (list): return of the SamAutomaticMaskGenerator.generate() function

    Returns:
        np.array: with shape (height, width) and values in [0, len(masks)-1]
    """
    onehot = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], len(masks)))
    for i, e in enumerate(masks):
        encoding = e['segmentation']
        onehot[...,i] = encoding
    return np.argmax(onehot, axis=-1)

def path2instances(path, mask_generator):
    """_summary_

    Args:
        path (str): path to image file
        mask_generator (SamAutomaticMaskGenerator): An SamAutomaticMaskGenerator object

    Returns:
        np.array: with shape (height, width) and values in [0, len(masks)-1]
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks = mask_generator.generate(image)
    instances = masks2encoding(masks)
    return instances

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
        store_directory (str): path to directory where the instance segmentation should be saved, if None, it is not saved
        mask_generator (SamAutomaticMaskGenerator): An SamAutomaticMaskGenerator object,

    Returns:
        np.array: with shape (height, width) and values in [0, len(masks)-1]
    """
    instances = path2instances(img_path, mask_generator)
    if store_directory is not None:
        save_path = os.path.join(store_directory, img_path.split("/")[-1].replace("frame","seg").replace("jpg", "npy"))
        print(save_path)
        instance_encoding2file(instances, save_path)
    return instances