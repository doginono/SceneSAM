import numpy as np
import Quaternion
import open3d as o3d
from tqdm import tqdm

colmap = '/home/rozenberszki/D_Project/wsnsl/Dataset/07f5b601ee/pose.txt'#
arkit = '/home/rozenberszki/D_Project/wsnsl/Dataset/07f5b601ee/traj.txt'

colmap = np.loadtxt(colmap).reshape(-1,4,4)
arkit = np.loadtxt(arkit).reshape(-1,4,4)

def create_coordinate_frame(transform, size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    return frame

# Convert COLMAP data to transformation matrices
colmap_poses = []
arkit_poses = []
for i in tqdm(range(len(colmap))):
    colmap_poses.append(create_coordinate_frame(colmap[i]))
    #arkit_poses.append(create_coordinate_frame(arkit[i*10]))

# Visualize the poses using Open3D
o3d.visualization.draw_geometries(colmap_poses)