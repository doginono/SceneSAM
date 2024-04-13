from PIL import Image
import os


def convert_png_to_jpg(source_dir, target_dir):
    # Check if target directory exists, create if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".png"):
            # Construct full file path
            img_path = os.path.join(source_dir, filename)
            # Open the image file
            img = Image.open(img_path)
            # Define the new filename and new path
            new_filename = filename.replace(".png", ".jpg")
            new_path = os.path.join(target_dir, new_filename)
            # Convert the image to RGB mode and save it as a JPG
            rgb_img = img.convert("RGB")
            rgb_img.save(
                new_path, format="JPEG", quality=95
            )  # quality is from 1 (worst) to 95 (best)

            print(f"Converted and saved {new_filename} to {target_dir}")


# Define source and target directories
source_directory = "/home/rozenberszki/project/wsnsl/Datasets/Replica/room0_panoptic/color"  # Path to the directory containing PNG files
target_directory = "/home/rozenberszki/project/wsnsl/Datasets/Replica/room0_panoptic/results"  # Path where JPG files will be stored

convert_png_to_jpg(source_directory, target_directory)
