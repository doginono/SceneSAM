import glob
from PIL import Image
import os
from src.utils import vis
import numpy as np
import matplotlib.pyplot as plt
def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    print(frames)
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder, 'gif.gif'), format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    

def make_gif_from_array(semantic_frames, store):
    frames = []
    visualizerForId = vis.visualizerForIds()
    
    for frame in semantic_frames:
        # Create image from array using plt.plot
        colors = visualizerForId.get_colors(frame.numpy().astype(np.uint8))
        im = Image.fromarray(colors)
        frames.append(im)
        
        # Convert plot to image
        
    frame_one = frames[0]
    frame_one.save(store, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
        
    # Rest of the code...
    
if __name__ == "__main__":
    make_gif("/home/koerner/Project/nice-slam/Datasets/Replica/room0/segmentation")
