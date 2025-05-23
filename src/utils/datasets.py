import glob
import os
import pickle
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
import threading
import seaborn as sns
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if "Y" not in header["channels"] else channelData["Y"]

    return Y


def get_dataset(cfg, args, scale, device="cuda:0", tracker=False, slam=None):
    return dataset_dict[cfg["dataset"]](
        cfg, args, scale, device=device, tracker=tracker, slam=slam
    )


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, slam, tracker, device="cuda:0"):
        super(BaseDataset, self).__init__()

        s = torch.ones((4, 4)).int()
        if cfg["dataset"] == "tumrgbd":
            s[[0, 0, 1, 2], [0, 1, 2, 2]] *= -1
        elif cfg["dataset"] == "replica":
            s[[0, 0, 1, 1, 2], [1, 2, 0, 3, 3]] *= -1
        self.shift = s  # s
        self.name = cfg["dataset"]
        if args.input_folder is None:
            self.input_folder = cfg["data"]["input_folder"]
        else:
            self.input_folder = args.input_folder

        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            cfg["cam"]["H"],
            cfg["cam"]["W"],
            cfg["cam"]["fx"],
            cfg["cam"]["fy"],
            cfg["cam"]["cx"],
            cfg["cam"]["cy"],
        )

        self.distortion = (
            np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        )
        self.crop_size = cfg["cam"]["crop_size"] if "crop_size" in cfg["cam"] else None

        self.crop_edge = cfg["cam"]["crop_edge"]
        # -------------------added-----------------------------------------------
        # if self.name == "replica":
        self.output_dimension_semantic = cfg["output_dimension_semantic"]
        self.every_frame_seg = cfg["Segmenter"]["every_frame"]
        self.seg_folder = f"{self.input_folder}/segmentation"
        self.round = slam.round
        self.mask_paths = sorted(glob.glob(f"{self.input_folder}/results/mask*.pkl"))
        self.output_dimension_semantic = slam.output_dimension_semantic
        self.every_frame = cfg["mapping"]["every_frame"]
        self.every_frame_seg = cfg["Segmenter"]["every_frame"]
        self.istracker = tracker
        self.points_per_instance = cfg["mapping"]["points_per_instance"]
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        self.id_counter = slam.id_counter
        # ------------------end-added-----------------------------------------------

    def __len__(self):
        return self.n_img

    def __post_init__(self, slam):
        self.semantic_frames = slam.semantic_frames

    def get_zero_pose(self):
        return self.poses[0]  # * self.shift

    def get_segmentation(self, index):
        # assert index % self.every_frame_seg == 0, "index should be multiple of every_frame"
        """semantic_path = os.path.join(self.seg_folder, f"seg_{index}.npy")
        semantic_data = np.load(semantic_path)"""
        # Create one-hot encoding using numpy.eye
        if index % self.every_frame_seg == 0:
            semantic_data = (
                self.semantic_frames[index // self.every_frame_seg].clone().int()
            )
        else:
            semantic_data = self.semantic_frames[-1].clone().int()
        negative = np.where(semantic_data < 0)
        semantic_data[negative] = 0
        semantic_data = np.eye(self.output_dimension_semantic[0])[semantic_data].astype(
            bool
        )
        semantic_data[negative] = False
        semantic_data = torch.from_numpy(semantic_data)
        edge = self.crop_edge
        if edge > 0:
            semantic_data = semantic_data[edge:-edge, edge:-edge]
        return semantic_data.to(self.device)

    def get_colorAndDepth(self, index, edge=True):
        if edge:
            edge = self.crop_edge
        else:
            edge = 0
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".exr" in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale
        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            print(f"depth shape: {depth_data.shape}, color shape: {color_data.shape}")
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode="bilinear", align_corners=True
            )[0]
            """sns.histplot(depth_data.numpy().reshape(-1))
            plt.title("Depth data before resize")
            plt.show()[384,512]
            print(depth_data.shape, " before resize")"""

            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode="nearest"
            )[0, 0]
            """print(depth_data.shape, " after resize")

            sns.histplot(depth_data.flatten().reshape(-1))
            plt.title("Depth data after resize")
            plt.show()"""
            color_data = color_data.permute(1, 2, 0).contiguous()

        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        return color_data.to(self.device), depth_data.to(self.device)

    def __getitem__(self, index):
        """color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".exr" in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode="bilinear", align_corners=True
            )[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode="nearest"
            )[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]"""
        color_data, depth_data = self.get_colorAndDepth(index)
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        semantic_data = self.get_segmentation(index)
        # pose = pose * self.shift.float()
        return (
            index,
            color_data.to(self.device),
            depth_data.to(self.device),
            pose.to(self.device),
            semantic_data.to(self.device),
        )


class Replica(BaseDataset):

    def __init__(self, cfg, args, scale, device="cuda:0", tracker=False, slam=None):
        super(Replica, self).__init__(cfg, args, scale, slam, tracker, device)

        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))

        """# -------------------added-----------------------------------------------
        # self.semantic_paths = sorted(glob.glob(f'{self.input_folder}/results/semantic*.npy'))
        self.round = slam.round
        self.mask_paths = sorted(glob.glob(f"{self.input_folder}/results/mask*.pkl"))
        self.output_dimension_semantic = slam.output_dimension_semantic
        self.every_frame = cfg["mapping"]["every_frame"]
        self.every_frame_seg = cfg["Segmenter"]["every_frame"]
        self.istracker = tracker
        self.points_per_instance = cfg["mapping"]["points_per_instance"]
        # self.T_wc = slam.T_wc
        self.K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        self.id_counter = slam.id_counter
        # -------------------end added-----------------------------------------------"""
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)

        # -------------------added-----------------------------------------------

        # -----------------end--added-----------------------------------------------
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".exr" in depth_path:
            depth_data = readEXR_onlydepth(depth_path)
        if (
            self.distortion is not None
        ):  # TODO should distorion be applied to semantics?
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        image = cv2.cvtColor(
            color_data, cv2.COLOR_BGR2RGB
        )  # J: convertion BGR -> RGB, image is passed to sam
        color_data = image / 255.0
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        # -------------------added-----------------------------------------------
        """semantic_path = os.path.join(self.seg_folder, f"seg_{index}.npy")
        semantic_data = semantic_data.resize(
            (H, W, self.output_dimension_semantic)
        )  # TODO check if this works"""
        # ------------------end-added-----------------------------------------------
        color_data = cv2.resize(
            color_data, (W, H)
        )  # shape after (680, 1200, 3) = (H, W, 3)
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data) * self.scale
        # -------------------added-----------------------------------------------
        semantic_data = self.get_segmentation(index)
        """if index % self.every_frame_seg == 0:
            
            semantic_path = os.path.join(self.seg_folder, f"seg_{index}.npy")
            # semantic_data = semantic_data.resize((H, W, self.output_dimension_semantic)) #TODO check if this works
            # semantic_data = self.semantic_frames[index//self.every_frame].clone().int()
            if self.round[0] == 0:
                semantic_data = np.ones((H, W), int)
            else:
                semantic_data = np.load(semantic_path)

            # Create one-hot encoding using numpy.eye
            negative = np.where(semantic_data < 0)
            semantic_data[negative] = 0
            semantic_data = np.eye(self.output_dimension_semantic[0])[
                semantic_data
            ].astype(bool)
            semantic_data[negative] = False
            # TODO set sematic data corresponding to negative ids, set one-hot encoding to zero
            assert (
                self.output_dimension_semantic[0] >= semantic_data.shape[-1]
            ), "Number of classes is smaller than the number of unique values in the semantic data"
            semantic_data = torch.from_numpy(semantic_data)"""
        """else:
            semantic_data = (
                torch.ones((H, W, self.output_dimension_semantic[0])).to(bool) * -1
            )"""

        # ----------------------
        if (
            self.crop_size is not None
        ):  # TODO check if we ever use this, if yes add to semantic (maybe use assert(...))
            # follow the pre-processing step in lietorch, actually is resize
            # assert False, "crop_size is not None -> need to crop semantic data"
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode="bilinear", align_corners=True
            )[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode="nearest"
            )[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
            # -------------------added-----------------------------------------------
            # semantic_data = semantic_data[edge:-edge, edge:-edge]
            # ------------------end-added-----------------------------------------------
        pose = self.poses[index]
        pose[:3, 3] *= self.scale
        return (
            index,
            color_data.to(self.device),
            depth_data.to(self.device),
            pose.to(self.device),
            semantic_data.to(self.device),
        )  # Done: add return semantics

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class Azure(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(Azure, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "color", "*.jpg"))
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth", "*.png"))
        )
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, "scene", "trajectory.log"))

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(" ")))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(
                            map(
                                float, ("".join(content[i + 1 : i + 5])).strip().split()
                            )
                        )
                    ).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(ScanNet, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder, "frames")
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "rgb", "*.jpg")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "processedData/depth", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.load_poses(os.path.join(self.input_folder, "processedData"))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, "*.txt")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(" ")))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class CoFusion(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0"):
        super(CoFusion, self).__init__(cfg, args, scale, device)
        self.input_folder = os.path.join(self.input_folder)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "colour", "*.png"))
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, "depth_noise", "*.exr"))
        )
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(self.input_folder, "trajectories"))

    def load_poses(self, path):
        # We tried, but cannot align the coordinate frame of cofusion to ours.
        # So here we provide identity matrix as proxy.
        # But it will not affect the calculation of ATE since camera trajectories can be aligned.
        self.poses = []
        for i in range(self.n_img):
            c2w = np.eye(4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device="cuda:0", tracker=False, slam=None):
        super(TUM_RGBD, self).__init__(cfg, args, scale, slam, tracker, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32
        )
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """read list data"""
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """pair images, depths, and poses"""
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """read video data in tum-rgbd format"""
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            """if inv_pose is None:
                inv_pose = np.linalg.inv(c2w) #
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w"""
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """convert 4x4 pose matrix to (t, q)"""
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "cofusion": CoFusion,
    "azure": Azure,
    "tumrgbd": TUM_RGBD,
}
