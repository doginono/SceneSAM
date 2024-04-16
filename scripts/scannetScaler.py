import numpy as np
import glob
import cv2
import torch
from sklearn.cluster import KMeans
import os
import json

def read_poses_from_json(file_path):
    """Reads the pose data from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_matrices_to_txt(data, output_file_path):
    """Writes the 4x4 pose matrices to a text file."""
    with open(output_file_path, 'w') as file:
        for frame_id, frame_data in data.items():
            pose = frame_data['pose']
            #file.write(f"{frame_id}:\n")
            for row in pose:
                formatted_row = ' '.join(f"{value:.18e}" for value in row)
                file.write(formatted_row + "\n")

def write_intirinsic_to_txt(data, output_file_path):
    """Writes the 4x4 pose matrices to a text file."""
    with open(output_file_path, 'w') as file:
        for frame_id, frame_data in data.items():
            pose = frame_data['intrinsic']
            #file.write(f"{frame_id}:\n")
            for row in pose:
                formatted_row = ' '.join(f"{value:.18e}" for value in row)
                file.write(formatted_row + "\n")


import torch.nn.functional as F
def readDepth(filepath):
    depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    depth_data = depth.astype(np.float32) / 6553.5
    depth_data = torch.from_numpy(depth_data)
    #print(depth_data.shape)
    #print(depth_data[0,0])
    return depth_data
    

import glob
from tqdm import tqdm  # Import tqdm

def readColorImage(filepath):
    # Read color image and convert to RGB (OpenCV uses BGR by default)
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to float32 and normalize to [0, 1] if needed
    return torch.from_numpy(image.astype(np.float32) / 255.0)

def save_depth_image(input_path, output_filepath,amount=0,upscale=7.5):
    """Saves the depth image to a file."""
    depth_paths = sorted(
            glob.glob(os.path.join(input_path, "*.png"))
        )
    #print(depth_paths)
    os.makedirs(output_filepath, exist_ok=True)  # Ensure output path exists
    depth_paths=depth_paths[:amount]
    for i, depth_path in enumerate(tqdm(depth_paths, desc='Upsampling depth images')):
        # this changes the vals
        depth_data=readDepth(depth_path)*upscale
        # this image shape
        upsampled_depth_bilinear = F.interpolate(depth_data.unsqueeze(0).unsqueeze(0), 
                                                scale_factor=upscale, 
                                                mode='bilinear', 
                                               ).squeeze()
        #print(upsampled_depth_bilinear.shape)
        upsampled_depth_bilinear = upsampled_depth_bilinear
        upsampled_depth = upsampled_depth_bilinear * 6553.5
        upsampled_depth = upsampled_depth.numpy().astype(np.uint16)
        #print((upsampled_depth.shape))

        cv2.imwrite(os.path.join(output_filepath, f"frame_{str(i).zfill(6)}.png"), upsampled_depth)
        

def save_color_image(input_path, output_filepath, amount=0, upscale=1):
    """Saves the color images to a file after upscaling."""
    color_paths = sorted(glob.glob(os.path.join(input_path, "*.jpg")))
    color_paths = color_paths[:amount] if amount > 0 else color_paths
    
    os.makedirs(output_filepath, exist_ok=True)  # Ensure output path exists
    
    for i, color_path in enumerate(tqdm(color_paths, desc='Processing color images')):
        color_data = readColorImage(color_path)
        color_data = color_data.permute(2, 0, 1).unsqueeze(0)  # CxHxW and add batch dimension
        
        upsampled_color_bilinear = F.interpolate(color_data, 
                                                scale_factor=upscale, 
                                                mode='bilinear', 
                                                align_corners=True).squeeze(0)
        
        # Convert back to uint8 and BGR for saving
        upsampled_color = (upsampled_color_bilinear.permute(1, 2, 0) * 255).byte().numpy()
        upsampled_color = cv2.cvtColor(upsampled_color, cv2.COLOR_RGB2BGR)
        #print(upsampled_color.shape)
        cv2.imwrite(os.path.join(output_filepath, f"frame_{str(i).zfill(6)}.jpg"), upsampled_color)



if __name__ == "__main__":

    # Paths to your JSON input and text output files
    input_json_path = '/home/rozenberszki/project/wsnsl/Datasets/Scannet++/data/56a0ec536c/iphone/pose_intrinsic_imu.json'  # Change this to the path of your JSON file
    output_txt_path = '/home/rozenberszki/D_Project/wsnsl/Dataset/traj.txt'  # Change this to your desired output file path

    # Read the data from JSON
    pose_data = read_poses_from_json(input_json_path)

    # Write the matrices to a text file
    write_matrices_to_txt(pose_data, output_txt_path)


    pose_data = read_poses_from_json(input_json_path)
    output_txt_path = '/home/rozenberszki/D_Project/wsnsl/Dataset/intrinsic.txt'
    # Write the matrices to a text file
    write_intirinsic_to_txt(pose_data, output_txt_path)


    color_paths = sorted(glob.glob(f"/home/rozenberszki/project/wsnsl/Datasets/Scannet++/data/56a0ec536c/iphone/rgb/*.jpg"))
    print(color_paths)

    color_data = cv2.imread(color_paths[0])
    image = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

    depth_paths = sorted(glob.glob(f"/home/rozenberszki/project/wsnsl/Datasets/Scannet++/data/56a0ec536c/iphone/processedData/depth/*.png"))
    print("image",image.shape)
    print("DEPT",readDepth(depth_paths[0]).shape)

    #save_color_image('/home/rozenberszki/project/wsnsl/Datasets/Scannet++/data/56a0ec536c/iphone/rgb/', '/home/rozenberszki/D_Project/wsnsl/Dataset/color_path/',amount=500)
    save_depth_image('/home/rozenberszki/project/wsnsl/Datasets/Scannet++/data/56a0ec536c/iphone/depth', '/home/rozenberszki/D_Project/wsnsl/Dataset/dept_path/',amount=500)
