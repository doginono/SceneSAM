import numpy as np
import cv2


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