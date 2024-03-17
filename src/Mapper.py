import os
import time

import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from tqdm import tqdm

from src.common import (
    get_camera_from_tensor,
    get_samples,
    get_tensor_from_camera,
    random_select,
)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer
from src.utils import backproject

from torch.utils.tensorboard import SummaryWriter  # J: added


class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper

        # -------added------------------
        self.every_frame_seg = cfg["Segmenter"]["every_frame"]
        self.round = slam.round
        # self.wait_segmenter = cfg['Segmenter']['mask_generator']
        self.seg_freq = cfg["Segmenter"]["every_frame"]
        # self.T_wc = slam.T_wc
        self.output_dimension_semantic = slam.output_dimension_semantic
        self.semantic_iter_ratio = cfg["mapping"]["semantic_iter_ratio"]
        self.w_semantic_loss = cfg["mapping"]["w_semantic_loss"]
        self.writer_path = cfg["data"]["logs"]  # J:added
        self.use_vis = cfg["mapping"]["use_vis"]
        self.use_mesh = cfg["mapping"]["use_mesh"]
        self.iters_first = cfg["mapping"]["iters_first"]
        self.idx_writer = 0
        self.vis_freq = cfg["mapping"]["vis_freq"]
        self.vis_offset = cfg["mapping"]["vis_offset"]
        self.no_vis_on_first_frame = cfg["mapping"]["no_vis_on_first_frame"]
        self.freq = cfg["mapping"]["vis_freq"]
        self.inside_freq = cfg["mapping"]["vis_inside_freq"]
        """if ~self.coarse_mapper:
            self.writer = SummaryWriter(os.path.join(cfg['writer_path'], 'coarse')) #J: added
        else:
            self.writer = SummaryWriter(os.path.join(cfg['writer_path'], 'regular'))"""

        # self.writer = SummaryWriter(os.path.join(cfg['writer_path'])) #J: added
        # self.idx_mapper = slam.idx_mapper
        # self.idx_coarse_mapper = slam.idx_coarse_mapper
        self.idx_segmenter = slam.idx_segmenter
        self.is_full_slam = cfg["Segmenter"]["full_slam"]
        # ------end-added------------------

        self.idx = slam.idx  # tracking index
        self.nice = slam.nice
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.vis_renderer = (
            slam.vis_renderer
        )  # J: added to use smaller batch size for visualization
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list  # for tracker
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg["scale"]
        self.coarse = cfg["coarse"]
        self.occupancy = cfg["occupancy"]
        self.sync_method = cfg["sync_method"]

        self.device = cfg["mapping"]["device"]
        self.fix_fine = cfg["mapping"]["fix_fine"]
        self.eval_rec = cfg["meshing"]["eval_rec"]
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg["mapping"]["BA_cam_lr"]
        self.mesh_freq = cfg["mapping"]["mesh_freq"]
        self.ckpt_freq = cfg["mapping"]["ckpt_freq"]
        self.fix_color = cfg["mapping"]["fix_color"]
        self.mapping_pixels = cfg["mapping"]["pixels"]
        self.num_joint_iters = cfg["mapping"]["iters"]
        self.clean_mesh = cfg["meshing"]["clean_mesh"]
        self.every_frame = cfg["mapping"]["every_frame"]
        # TODO maybe need to add semantics<- check what color_refine and w_color_loss is doing
        self.color_refine = cfg["mapping"]["color_refine"]
        self.w_color_loss = cfg["mapping"]["w_color_loss"]
        self.keyframe_every = cfg["mapping"]["keyframe_every"]
        self.fine_iter_ratio = cfg["mapping"]["fine_iter_ratio"]
        self.middle_iter_ratio = cfg["mapping"]["middle_iter_ratio"]
        self.mesh_coarse_level = cfg["meshing"]["mesh_coarse_level"]
        self.mapping_window_size = cfg["mapping"]["mapping_window_size"]
        self.no_vis_on_first_frame = cfg["mapping"]["no_vis_on_first_frame"]
        self.no_log_on_first_frame = cfg["mapping"]["no_log_on_first_frame"]
        self.no_mesh_on_first_frame = cfg["mapping"]["no_mesh_on_first_frame"]
        self.frustum_feature_selection = cfg["mapping"]["frustum_feature_selection"]
        self.keyframe_selection_method = cfg["mapping"]["keyframe_selection_method"]
        self.save_selected_keyframes_info = cfg["mapping"][
            "save_selected_keyframes_info"
        ]
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = "global"

        self.keyframe_dict = []
        self.keyframe_list = []
        """if coarse_mapper:
            self.frame_reader = get_dataset(
                cfg, args, self.scale, device=self.device, slam = slam, tracker=True)
        else:"""
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device, slam=slam, tracker=False
        )
        # self.frame_reader.__post_init__(slam)
        self.n_img = len(self.frame_reader)
        if "Demo" not in self.output:  # disable this visualization in demo
            self.visualizer = Visualizer(
                freq=cfg["mapping"]["vis_freq"],
                inside_freq=cfg["mapping"]["vis_inside_freq"],
                vis_dir=os.path.join(self.output, "mapping_vis"),
                renderer=self.vis_renderer,
                verbose=self.verbose,
                device=self.device,
                iters_first=cfg["mapping"]["iters_first"],
                num_iter=cfg["mapping"]["iters"],
                input_dimension_semantic=cfg["output_dimension_semantic"],
            )
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = (
            slam.H,
            slam.W,
            slam.fx,
            slam.fy,
            slam.cx,
            slam.cy,
        )

    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        (
            H,
            W,
            fx,
            fy,
            cx,
            cy,
        ) = (
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        X, Y, Z = torch.meshgrid(
            torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
            torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
            torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]),
        )

        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == "grid_coarse":
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate([points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c @ homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K @ cam_cord
        z = uv[:, -1:] + 1e-5
        uv = uv[:, :2] / z
        uv = uv.astype(np.float32)

        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [
                cv2.remap(
                    depth_np,
                    uv[i : i + remap_chunk, 0],
                    uv[i : i + remap_chunk, 1],
                    interpolation=cv2.INTER_LINEAR,
                )[:, 0].reshape(-1, 1)
            ]
        depths = np.concatenate(depths, axis=0)

        edge = 0
        mask = (
            (uv[:, 0] < W - edge)
            * (uv[:, 0] > edge)
            * (uv[:, 1] < H - edge)
            * (uv[:, 1] > edge)
        )

        # For ray with depth==0, fill it with maximum depth
        zero_mask = depths == 0
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths + 0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak - ray_o
        dist = torch.sum(dist * dist, axis=1)
        mask2 = dist < 0.5 * 0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    def keyframe_selection_overlap(
        self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100
    ):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color, tmp = get_samples(
            0,
            H,
            0,
            W,
            pixels,
            H,
            W,
            fx,
            fy,
            cx,
            cy,
            c2w,
            gt_depth,
            gt_color,
            None,
            self.device,
        )

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0.0, 1.0, steps=N_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe["est_c2w"].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate([vertices, ones], axis=1).reshape(
                -1, 4, 1
            )  # (N, 4)
            cam_cord_homo = w2c @ homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K @ cam_cord
            z = uv[:, -1:] + 1e-5
            uv = uv[:, :2] / z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (
                (uv[:, 0] < W - edge)
                * (uv[:, 0] > edge)
                * (uv[:, 1] < H - edge)
                * (uv[:, 1] > edge)
            )
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum() / uv.shape[0]
            list_keyframe.append({"id": keyframeid, "percent_inside": percent_inside})

        list_keyframe = sorted(
            list_keyframe, key=lambda i: i["percent_inside"], reverse=True
        )
        selected_keyframe_list = [
            dic["id"] for dic in list_keyframe if dic["percent_inside"] > 0.00
        ]
        selected_keyframe_list = list(
            np.random.permutation(np.array(selected_keyframe_list))[:k]
        )
        return selected_keyframe_list

    def optimize_map(
        self,
        num_joint_iters,
        lr_factor,
        idx,
        cur_gt_color,
        cur_gt_depth,
        gt_cur_c2w,
        keyframe_dict,
        keyframe_list,
        cur_c2w,
        cur_gt_semantic=None,
        writer=None,
        round=0,
        mapping_first_frame=None,
    ):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).

        Args:
            cur_gt_semantic(tensor): gt_semantic image of the current camera.

            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        cfg = self.cfg
        device = self.device
        bottom = (
            torch.from_numpy(np.array([0, 0, 0, 1.0]).reshape([1, 4]))
            .type(torch.float32)
            .to(device)
        )

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == "global":  # J: usually it is global
                num = self.mapping_window_size - 2
                optimize_frame = random_select(len(self.keyframe_dict) - 1, num)
            elif self.keyframe_selection_method == "overlap":
                num = self.mapping_window_size - 2
                # TODO: if full_slam: this is fine, else all but current frame
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num
                )

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]  # TODO not working if post segmentation is used

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]["gt_c2w"]
                    tmp_est_c2w = keyframe_dict[frame]["est_c2w"]
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {"idx": frame_idx, "gt_c2w": tmp_gt_c2w, "est_c2w": tmp_est_c2w}
                )
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels // len(optimize_frame)

        decoders_para_list = []
        coarse_grid_para = []
        middle_grid_para = []
        fine_grid_para = []
        color_grid_para = []
        # -----------------added-------------------
        semantic_grid_para = []
        # -----------------end-added-------------------

        gt_depth_np = cur_gt_depth.cpu().numpy()
        if self.nice:
            if self.frustum_feature_selection:
                masked_c_grad = {}
                mask_c2w = cur_c2w
            for key, val in c.items():
                if not self.frustum_feature_selection:
                    val = Variable(val.to(device), requires_grad=True)
                    c[key] = val
                    if key == "grid_coarse":
                        coarse_grid_para.append(val)
                    elif key == "grid_middle":
                        middle_grid_para.append(val)
                    elif key == "grid_fine":
                        fine_grid_para.append(val)
                    elif key == "grid_color":
                        color_grid_para.append(val)
                    # -----------------added-------------------
                    elif key == "grid_semantic":
                        semantic_grid_para.append(val)
                    # -----------------end-added-------------------

                else:
                    mask = self.get_mask_from_c2w(
                        mask_c2w, key, val.shape[2:], gt_depth_np
                    )
                    mask = (
                        torch.from_numpy(mask)
                        .permute(2, 1, 0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .repeat(1, val.shape[1], 1, 1, 1)
                    )
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key + "mask"] = mask
                    if key == "grid_coarse":
                        coarse_grid_para.append(val_grad)
                    elif key == "grid_middle":
                        middle_grid_para.append(val_grad)
                    elif key == "grid_fine":
                        fine_grid_para.append(val_grad)
                    elif key == "grid_color":
                        color_grid_para.append(val_grad)
                    # -----------------added-------------------
                    elif key == "grid_semantic":
                        semantic_grid_para.append(val_grad)
                    # -----------------end-added-------------------

        if self.nice:
            if (
                not self.fix_fine
            ):  # TODO check purpose -> maybe add similar for semantics
                decoders_para_list += list(self.decoders.fine_decoder.parameters())
            if not self.fix_color:
                decoders_para_list += list(self.decoders.color_decoder.parameters())

        else:
            # imap*, single MLP
            # J:will not be entered because nice = True always in our case
            decoders_para_list += list(self.decoders.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:
                    if frame != -1:
                        c2w = keyframe_dict[frame]["est_c2w"]
                        gt_c2w = keyframe_dict[frame]["gt_c2w"]
                    else:
                        c2w = cur_c2w
                        gt_c2w = gt_cur_c2w
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True
                    )
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        if self.nice:
            if self.BA:
                # Done: add the parameter list for semantics
                # The corresponding lr will be set according to which stage the optimization is in
                optimizer = torch.optim.Adam(
                    [
                        {"params": decoders_para_list, "lr": 0},
                        {"params": coarse_grid_para, "lr": 0},
                        {"params": middle_grid_para, "lr": 0},
                        {"params": fine_grid_para, "lr": 0},
                        {"params": color_grid_para, "lr": 0},
                        {"params": semantic_grid_para, "lr": 0},
                        {"params": camera_tensor_list, "lr": 0},
                    ]
                )
            else:
                optimizer = torch.optim.Adam(
                    [
                        {"params": decoders_para_list, "lr": 0},
                        {"params": coarse_grid_para, "lr": 0},
                        {"params": middle_grid_para, "lr": 0},
                        {"params": fine_grid_para, "lr": 0},
                        {"params": color_grid_para, "lr": 0},
                        {"params": semantic_grid_para, "lr": 0},
                    ]
                )
        else:
            assert False
            # imap*, single MLP
            # J:will not be entered because nice = True always in our case
            if self.BA:
                optimizer = torch.optim.Adam(
                    [
                        {"params": decoders_para_list, "lr": 0},
                        {"params": camera_tensor_list, "lr": 0},
                    ]
                )
            else:
                optimizer = torch.optim.Adam([{"params": decoders_para_list, "lr": 0}])
            from torch.optim.lr_scheduler import StepLR

            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)

        # J: added semantic optimizing part: for now it is 0.4 which is the same as the number of iterations spend on color. Carefull the semantic_iter_rati is added tothe num_joint_iters
        """if ((~self.coarse_mapper and idx % self.seg_freq == 0) or idx == self.n_img - 1):
            inc = int(self.semantic_iter_ratio * num_joint_iters)

        else:
            inc = 0"""
        if round == 1:
            start = num_joint_iters
            inc = max(int(self.semantic_iter_ratio * num_joint_iters), 1)
        else:
            start = 0
            if self.is_full_slam and idx % self.seg_freq == 0:
                inc = int(self.semantic_iter_ratio * num_joint_iters)
            else:
                inc = 0

        for joint_iter in tqdm(
            range(start, num_joint_iters + inc), desc=f"Training on Frame {idx.item()}"
        ):
            if self.nice:
                if self.frustum_feature_selection:
                    for key, val in c.items():
                        if (self.coarse_mapper and "coarse" in key) or (
                            (not self.coarse_mapper) and ("coarse" not in key)
                        ):
                            val_grad = masked_c_grad[key]
                            mask = masked_c_grad[key + "mask"]
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val

                if self.coarse_mapper:
                    self.stage = "coarse"
                elif joint_iter <= int(
                    num_joint_iters * self.middle_iter_ratio
                ):  # middle_iter_ratio is 0.4
                    self.stage = "middle"
                elif joint_iter <= int(
                    num_joint_iters * self.fine_iter_ratio
                ):  # fine_iter_ratio is 0.6
                    self.stage = "fine"
                elif joint_iter <= num_joint_iters:
                    self.stage = "color"
                else:
                    mapping_first_frame[0] = 1
                    while idx > self.idx_segmenter[0]:
                        time.sleep(0.1)
                    self.stage = "semantic"
                    # DONE: add semantics, we should probably increase the
                    # num_joint_iters and decrease the ratios to train on semantics
                    # as long as on colors and keeping the rest the same

                optimizer.param_groups[0]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["decoders_lr"] * lr_factor
                )
                optimizer.param_groups[1]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["coarse_lr"] * lr_factor
                )
                optimizer.param_groups[2]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["middle_lr"] * lr_factor
                )
                optimizer.param_groups[3]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["fine_lr"] * lr_factor
                )
                optimizer.param_groups[4]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["color_lr"] * lr_factor
                )
                # -----------------added-------------------
                optimizer.param_groups[5]["lr"] = (
                    cfg["mapping"]["stage"][self.stage]["semantic_lr"] * lr_factor
                )
                # -----------------end-added-------------------

                if self.BA:
                    if self.stage == "color":
                        optimizer.param_groups[6]["lr"] = self.BA_cam_lr
            else:  # J: this else will not be entered because nice = True always in our case
                self.stage = "color"
                optimizer.param_groups[0]["lr"] = cfg["mapping"]["imap_decoders_lr"]
                if self.BA:
                    optimizer.param_groups[1]["lr"] = self.BA_cam_lr

            # J: Part of the if is double checked (also in vis(), but we dont want to load data unnecessarily)
            if (
                (not (idx == 0 and self.no_vis_on_first_frame))
                and ("Demo" not in self.output)
                and self.use_vis
                and idx - self.vis_offset >= 0
                and (idx % self.freq == 0)
                and (joint_iter % self.inside_freq == 0)
                and self.stage != "coarse"
            ):
                _, gt_vis_color, gt_vis_depth, gt_c2w, gt_vis_semantic = (
                    self.frame_reader[idx - self.vis_offset]
                )
                self.visualizer.vis(
                    idx,
                    joint_iter,
                    gt_vis_depth,
                    gt_vis_color,
                    cur_c2w,
                    self.c,
                    self.decoders,
                    gt_vis_semantic,
                    only_semantic=False,
                    stage=self.stage,
                    writer=writer,
                    offset=self.vis_offset,
                )
                torch.cuda.empty_cache()

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []
            # -----------------added-------------------
            batch_gt_semantic_list = []
            # -----------------end-added-------------------

            camera_tensor_id = 0

            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]["depth"].to(device)
                    gt_color = keyframe_dict[frame]["color"].to(device)
                    # -----------------added-------------------
                    # jkl%
                    gt_semantic = (
                        torch.eye(self.output_dimension_semantic[0])[
                            keyframe_dict[frame]["semantic"]
                        ]
                        .to(bool)
                        .to(device)
                    )
                    ignore_pixel = keyframe_dict[frame]["ignore_pixel"].to(device)
                    gt_semantic[ignore_pixel] = 0
                    # gt_semantic = keyframe_dict[frame]['semantic'].to(device)
                    # -----------------end-added-------------------
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]["est_c2w"]

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    # -----------------added-------------------
                    """if self.stage != 'coarse':
                        gt_semantic = cur_gt_semantic.to(device)
                    else:
                        gt_semantic = None"""
                    gt_semantic = cur_gt_semantic.to(device)
                    # -----------------end-added-------------------
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w

                # -----------------added-------------------
                (
                    batch_rays_o,
                    batch_rays_d,
                    batch_gt_depth,
                    batch_gt_color,
                    batch_gt_semantic,
                ) = get_samples(
                    0,
                    H,
                    0,
                    W,
                    pixs_per_image,
                    H,
                    W,
                    fx,
                    fy,
                    cx,
                    cy,
                    c2w,
                    gt_depth,
                    gt_color,
                    gt_semantic,
                    self.device,
                )
                # -----------------end-added-------------------
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())
                # -----------------added-------------------
                batch_gt_semantic_list.append(batch_gt_semantic.float())
                # -----------------end-added-------------------

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)
            # -----------------added-------------------
            batch_gt_semantic = torch.cat(batch_gt_semantic_list)
            # -----------------end-added-------------------

            if self.nice:
                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = (
                        batch_rays_o.clone().detach().unsqueeze(-1)
                    )  # (N, 3, 1)
                    det_rays_d = (
                        batch_rays_d.clone().detach().unsqueeze(-1)
                    )  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(device) - det_rays_o) / det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
                # -----------------added-------------------
                batch_gt_semantic = batch_gt_semantic[inside_mask]
                # -----------------end-added-------------------

                # Done: add semantics in Render output
            ret = self.renderer.render_batch_ray(
                c,
                self.decoders,
                batch_rays_d,
                batch_rays_o,
                device,
                self.stage,
                gt_depth=None if self.coarse_mapper else batch_gt_depth,
            )
            depth, uncertainty, color_semantics = (
                ret  # J: color will contain semantics in semantic stage
            )

            depth_mask = batch_gt_depth > 0
            loss = torch.abs(  # J: we backpropagate only through depth in stage middle and fine
                batch_gt_depth[depth_mask] - depth[depth_mask]
            ).sum()
            if writer is not None:
                """if joint_iter == num_joint_iters +inc -1:
                depth_loss_writer = loss.item()/torch.sum(depth_mask)"""
                writer.add_scalar(f"Loss/depth", loss.item(), self.idx_writer)
            if (
                self.stage == "color"
            ):  # J: changed it from condition not self.nice or self.stage == 'color'
                color_loss = torch.abs(batch_gt_color - color_semantics).sum()
                """if joint_iter == num_joint_iters +inc -1:
                    print('Entered')
                    color_loss_writer = color_loss.item()/color_semantics.shape[0]"""
                if writer is not None:
                    writer.add_scalar(f"Loss/color", color_loss.item(), self.idx_writer)
                weighted_color_loss = self.w_color_loss * color_loss
                loss += weighted_color_loss
            # -----------------added-------------------
            elif self.stage == "semantic":
                loss_function = torch.nn.CrossEntropyLoss()

                """mask = (batch_gt_semantic >= 0)
                color_semantics = color_semantics[mask].reshape(-1, self.output_dimension_semantic)
                batch_gt_semantic = batch_gt_semantic[mask].reshape(-1, self.output_dimension_semantic)
                assert torch.all(torch.sum(batch_gt_semantic == 1, dim=1) == 1), "batch_gt_semantic should have exactly one '1' per row"""
                semantic_loss = loss_function(color_semantics, batch_gt_semantic)

                # semantic_loss = loss_function(color_semantics, batch_gt_semantic)
                """if joint_iter == num_joint_iters +inc -1:
                    semantic_loss_writer = semantic_loss.item()/color_semantics.shape[0]"""
                writer.add_scalar(
                    f"Loss/semantic", semantic_loss.item(), self.idx_writer
                )
                weighted_semantic_loss = self.w_semantic_loss * semantic_loss
                loss += weighted_semantic_loss
            # -----------------end-added-------------------
            if writer is not None:
                writer.add_scalar(f"Loss/Loss_overall", loss.item(), self.idx_writer)

            self.idx_writer += 1

            # for imap*, it uses volume density
            regulation = not self.occupancy
            if regulation:  # Done: check if we enter here <- we never enter
                point_sigma = self.renderer.regulation(
                    c,
                    self.decoders,
                    batch_rays_d,
                    batch_rays_o,
                    batch_gt_depth,
                    device,
                    self.stage,
                )
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.0005 * regulation_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            if not self.nice:  # J: never enter
                # for imap*
                assert False
                scheduler.step()
            optimizer.zero_grad()

            # put selected and updated features back to the grid
            if self.nice and self.frustum_feature_selection:
                for key, val in c.items():
                    if (self.coarse_mapper and "coarse" in key) or (
                        (not self.coarse_mapper) and ("coarse" not in key)
                    ):
                        val_grad = masked_c_grad[key]
                        mask = masked_c_grad[key + "mask"]
                        val = val.detach()
                        val[mask] = val_grad.clone().detach()
                        c[key] = val

        """writer.add_scalar(f'Loss/depth', depth_loss_writer, idx)
        writer.add_scalar(f'Loss/color', color_loss_writer, idx)
        writer.add_scalar(f'Loss/semantic', semantic_loss_writer, idx)"""

        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(
                            camera_tensor_list[camera_tensor_id].detach()
                        )
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]["est_c2w"] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self):
        writer = SummaryWriter(self.writer_path)
        round = self.round[0]

        cfg = self.cfg
        """if self.coarse_mapper:
            idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0] #Done add semantics to output, runs into index error 
            gt_semantic = None
        #as long as the sematic files are not added like .../Results/sematic*.npy
        else: """

        # print(f"start mapping, is coarse mapper: {self.coarse_mapper}")
        # TODO: adapt framereader to only give correct output
        idx, gt_color, gt_depth, gt_c2w, gt_semantic = self.frame_reader[0]
        if round == 0:
            self.estimate_c2w_list[0] = gt_c2w.cpu()

        if round == 1:
            idx = torch.tensor(-self.every_frame_seg)
        init = True
        prev_idx = -1

        while 1:

            # the idea here is that the segmenter segments the current frame and after it has finished the two mappers train on that frame
            # this ensures that the current frame has always been segmented before the mappers start training on it
            # TODO: will need to update this according to postsegmentatoin or full-SLAM
            while round == 0:
                """print(
                    "in while loop: ",
                    self.coarse_mapper,
                    " ",
                    self.idx[0].clone(),
                    " ",
                    prev_idx,
                )"""
                idx = self.idx[0].clone()  # this should be the current tracking index
                if idx == self.n_img - 1:
                    break
                if self.sync_method == "strict":
                    if idx % self.every_frame == 0 and idx != prev_idx:
                        break

                elif self.sync_method == "loose":
                    if idx == 0 or idx >= prev_idx + self.every_frame // 2:
                        break
                elif self.sync_method == "free":
                    break
                time.sleep(0.1)

            if round == 1:
                idx += self.every_frame_seg
                print("round 2 idx: ", idx)
                if idx >= self.n_img - 1:
                    break

            prev_idx = idx
            """if self.wait_segmenter:
                if init:
                    pass
                elif self.coarse_mapper:

                    while True:
                        if self.idx_segmenter[0] > self.idx_coarse_mapper[0]:
                            break
                        time.sleep(0.1)
                else:  # normal mapper
                    while True:
                        if self.idx_segmenter[0] > self.mapping_idx[0]:
                            break
                        time.sleep(0.1)
            else:  # such that coarse mapper and normal mapper stay roughly in sync
                pass"""
            """     if self.coarse_mapper:
                    while True:
                        if self.idx_mapper[0] +3 > self.idx_coarse_mapper[0]:
                            break
                        time.sleep(0.1)
                else: #normal mapper
                    while True:
                        if self.idx_coarse_mapper[0] +3 > self.idx_mapper[0]:
                            break
                        time.sleep(0.1)"""
            """if init:
                self.idx[0] = idx
            else:
                if self.coarse_mapper:
                    self.idx[0] = idx+self.every_frame
                else:
                    while True: #this is used to ensure that the tracker is faster than the mapper, now we have to make sure that the mappers are roughly in sync
                        idx = self.idx[0].clone()
                        if idx == self.n_img-1:
                            break
                        if self.sync_method == 'strict':
                            if idx % self.every_frame == 0 and idx != prev_idx:
                                break

                        elif self.sync_method == 'loose':
                            if idx == 0 or idx >= prev_idx+self.every_frame//2:
                                break
                        elif self.sync_method == 'free':
                            break
                        time.sleep(0.1)
                    prev_idx = idx"""
            """if self.coarse_mapper:
                if not init:
                    self.idx_coarse_mapper[0] = idx+self.every_frame
                while True:
                    idx = self.idx_coarse_mapper[0].clone()
                    if self.idx_mapper[0] >= idx:
                        break
                    time.sleep(0.1)
            else:
                if not init:
                    self.idx_mapper[0] = idx+self.every_frame
                while True:
                    idx = self.idx_mapper[0].clone()
                    if self.idx_coarse_mapper[0] >= idx:
                        break
                    time.sleep(0.1)
            if self.coarse_mapper:
                assert False, "no coarse mapper anymore"
                idx = self.idx_coarse_mapper[0].clone()
            else:
                idx = self.idx_mapper[0].clone()"""

            if self.verbose:
                print(Fore.GREEN)
                prefix = "Coarse " if self.coarse_mapper else ""
                print(prefix + "Mapping Frame ", idx.item())  # idx.item()
                print(Style.RESET_ALL)

            """if self.coarse_mapper:
                _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx] #Done add semantics to output
                gt_semantic = None
            else:"""
            _, gt_color, gt_depth, gt_c2w, gt_semantic = self.frame_reader[idx]
            # runs into index error as long as the sematic files are not added like .../Results/sematic*.npy

            if not init:
                lr_factor = cfg["mapping"]["lr_factor"]
                num_joint_iters = cfg["mapping"]["iters"]

                # here provides a color refinement postprocess
                # TODO maybe add the same for semantics; this is a postprocessing step which makes the color outputs better
                # -> check when semantics are available
                if idx == self.n_img - 1 and self.color_refine:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    self.semantic_iter_ratio = (
                        1.0  # J: added to make it consistent to the previous code
                    )
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3

            else:
                outer_joint_iters = 1
                lr_factor = cfg["mapping"]["lr_first_factor"]
                num_joint_iters = cfg["mapping"]["iters_first"]

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)
            # cur_c2w = torch.from_numpy(self.T_wc[idx]).to(self.device)
            # cur_c2w = gt_c2w  # torch.from_numpy(backproject.T_inv(self.T_wc[idx])).to(self.device)
            # cur_c2w = gt_c2w.clone().to(self.device)
            # J: using a unseen frame during training for visualization
            # print(f'pose of frame {idx} is {gt_c2w}')
            num_joint_iters = num_joint_iters // outer_joint_iters
            for outer_joint_iter in range(outer_joint_iters):

                self.BA = (
                    (len(self.keyframe_list) > 4)
                    and cfg["mapping"]["BA"]
                    and (not self.coarse_mapper)
                )

                # Done: add semantics to optimize_map
                _ = self.optimize_map(
                    num_joint_iters,
                    lr_factor,
                    idx,
                    gt_color,
                    gt_depth,
                    gt_c2w,
                    self.keyframe_dict,
                    self.keyframe_list,
                    cur_c2w=cur_c2w,
                    cur_gt_semantic=gt_semantic,
                    writer=writer,
                    round=round,
                    mapping_first_frame=self.mapping_first_frame,
                )  # Done add semantics to arguments
                if self.BA:
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters - 1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img - 2)) and (
                        idx not in self.keyframe_list
                    ):
                        self.keyframe_list.append(idx)
                        ignore_pixel = torch.sum(gt_semantic, dim=-1) == 0
                        self.keyframe_dict.append(
                            {
                                "gt_c2w": gt_c2w.cpu(),
                                "idx": idx,
                                "color": gt_color.cpu(),
                                "depth": gt_depth.cpu(),
                                "est_c2w": cur_c2w.clone(),
                                "ignore_pixel": ignore_pixel.cpu(),
                                "semantic": torch.argmax(
                                    gt_semantic.to(int), dim=-1
                                ).cpu(),
                            }
                        )  # Done: add semantics ground truth

            if self.low_gpu_mem:
                if self.verbose:
                    print("Clearing GPU cache")
                torch.cuda.empty_cache()

            init = False
            self.mapping_first_frame[0] = 1

            if not self.coarse_mapper:
                if (
                    (not (idx == 0 and self.no_log_on_first_frame))
                    and idx % self.ckpt_freq == 0
                ) or idx == self.n_img - 1:
                    self.logger.log(
                        idx,
                        self.keyframe_dict,
                        self.keyframe_list,
                        selected_keyframes=(
                            self.selected_keyframes
                            if self.save_selected_keyframes_info
                            else None
                        ),
                    )

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                if (
                    self.use_mesh
                    and (idx % self.mesh_freq == 0)
                    and (not (idx == 0 and self.no_mesh_on_first_frame))
                ):
                    mesh_out_file = f"{self.output}/mesh/{idx:05d}_mesh"
                    if round == 0:
                        self.mesher.get_mesh(
                            mesh_out_file + "_color.ply",
                            self.c,
                            self.decoders,
                            self.keyframe_dict,
                            self.estimate_c2w_list,  # instead of estimatee_c2w
                            idx,
                            self.device,
                            show_forecast=self.mesh_coarse_level,
                            clean_mesh=self.clean_mesh,
                            get_mask_use_all_frames=False,
                        )  # mesh on color
                    else:
                        self.mesher.get_mesh(
                            mesh_out_file + "_seg.ply",
                            self.c,
                            self.decoders,
                            self.keyframe_dict,
                            self.estimate_c2w_list,  # instead of estimatee_c2w
                            idx,
                            self.device,
                            show_forecast=self.mesh_coarse_level,
                            color=False,
                            semantic=True,
                            clean_mesh=self.clean_mesh,
                            get_mask_use_all_frames=False,
                        )  # mesh on segmentation

                if idx == self.n_img - 1:
                    print("end, at round: ", round)
                    mesh_out_file_color = f"{self.output}/mesh/final_mesh_color.ply"
                    mesh_out_file_seg = f"{self.output}/mesh/final_mesh_seg.ply"
                    if round == 0:
                        self.mesher.get_mesh(
                            mesh_out_file_seg,
                            self.c,
                            self.decoders,
                            self.keyframe_dict,
                            self.estimate_c2w_list,  # instead of estimatee_c2w
                            idx,
                            self.device,
                            show_forecast=self.mesh_coarse_level,
                            clean_mesh=self.clean_mesh,
                            get_mask_use_all_frames=False,
                        )
                        os.system(
                            f"cp {mesh_out_file_color} {self.output}/mesh/{idx:05d}_mesh_color.ply"
                        )
                    else:
                        self.mesher.get_mesh(
                            mesh_out_file_color,
                            self.c,
                            self.decoders,
                            self.keyframe_dict,
                            self.estimate_c2w_list,  # instead of estimatee_c2w
                            idx,
                            self.device,
                            show_forecast=self.mesh_coarse_level,
                            color=True,
                            semantic=False,
                            clean_mesh=self.clean_mesh,
                            get_mask_use_all_frames=False,
                        )
                        os.system(
                            f"cp {mesh_out_file_seg} {self.output}/mesh/{idx:05d}_mesh_seg.ply"
                        )
                    """if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.T_wc, idx, self.device, show_forecast=False, #instead of estimatee_c2w
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break"""

            if idx >= self.n_img - 1:
                writer.close()
                break
            """
            # TODO: push the decoders to the cpu
            # print(f'Mapping frame done, is coarse: {self.coarse_mapper}')
            if self.coarse_mapper:
                assert False, "no coarse mapper anymore"
                self.idx_coarse_mapper[0] = idx + self.every_frame
            else:
                if idx == 10:
                    try:
                        # torch.cuda.memory._dump_snapshot(f"/home/koerner/Project/nice-slam/logs/memory_usage.pickle")
                        # torch.cuda.memory._record_memory_history(enabled=None)
                        pass
                    except Exception as e:
                        print(f"Failed to capture memory snapshot {e}")
                self.idx_mapper[0] = idx + self.every_frame"""
