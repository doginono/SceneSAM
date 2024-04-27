import glob
from PIL import Image
import os
from src.utils import vis
import numpy as np
import matplotlib.pyplot as plt
import torch


def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    print(frames)
    frame_one = frames[0]
    frame_one.save(
        os.path.join(frame_folder, "gif.gif"),
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=100,
        loop=0,
    )


def color_gif_from_array(color_frames, store, duration=100):
    frames = []
    for frame in color_frames:
        # Create image from array using plt.plot
        im = Image.fromarray((frame * 255).astype(np.uint8))
        frames.append(im)

    frame_one = frames[0]
    frame_one.save(
        store,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )


def make_gif_from_array(semantic_frames, store, max_frame=-1, duration=100):
    frames = []
    visualizerForId = vis.visualizerForIds()

    for frame in semantic_frames[:max_frame]:
        # Create image from array using plt.plot
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
        colors = visualizerForId.get_colors(frame) * 255
        im = Image.fromarray(colors.astype(np.uint8))
        frames.append(im)

        # Convert plot to image

    frame_one = frames[0]
    frame_one.save(
        store,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )

    # Rest of the code...


if __name__ == "__main__":
    make_gif("/home/rozenberszki/D_Project/wsnsl/Dataset/56a0ec536c/segmentation")
