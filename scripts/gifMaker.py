import glob
from PIL import Image
import os
def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    print(frames)
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder, 'gif.gif'), format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif("/home/koerner/Project/nice-slam/Datasets/Replica/room0/segmentation")
