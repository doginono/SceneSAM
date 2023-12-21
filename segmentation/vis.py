import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.colors as mcolors


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def vis(path_to_image, masks):
    """visualizes the predicted masks on the image

    Args:
        path_to_image (str): path to image file
        masks (list): return of the SamAutomaticMaskGenerator.generate() function"""
    image = cv2.imread(path_to_image)
    print(image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis("off")
    plt.show()


def vis(path_to_image, masks, uv=None):
    """visualizes the predicted masks on the image

    Args:
        path_to_image (str): path to image file
        masks (list): return of the SamAutomaticMaskGenerator.generate() function"""
    image = cv2.imread(path_to_image)
    print(image.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    if uv is not None:
        plt.annotate(
            "25, 50",
            xy=uv,
            xycoords="data",
            textcoords="figure fraction",
            arrowprops=dict(arrowstyle="->"),
        )
        plt.scatter(25, 50, s=500, c="red", marker="o")
    plt.axis("off")
    plt.show()


class VisualizerForIds:
    # Generate random colors once during class initialization
    colors = [np.random.random(3) for _ in range(10000)]
    shared_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors) - 1)

    def __init__(self):
        # Assign the shared colormap and normalization to the instance
        self.cmap = self.shared_cmap
        self.norm = self.norm

    def visualizer(self, anns, title=""):
        plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
        plt.title(title)
        plt.imshow(anns, cmap=self.cmap, norm=self.norm)
        plt.show()
