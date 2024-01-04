import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor, get_rgb_from_instance_id
from src.utils.vis import visualizerForIds


class Visualizer(object):
    """
    Visualize intermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0', iters_first=1500, num_iter= 60, input_dimension_semantic = 101):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.input_dimension_semantic = input_dimension_semantic #added
        os.makedirs(f'{vis_dir}', exist_ok=True)
        
        #------------------added------------------
        self.iters_first = iters_first
        self.num_iter = num_iter
        self.visualizerForIds = visualizerForIds()

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders, gt_semantic=None, only_semantic=False, stage = "", writer = None, offset = 0):
        """
        Visualization of depth, color images and save to file.

        Args:
            gt_semantic (tensor): ground truth semantic image of the current frame.

            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """

        #torch.cuda.empty_cache()
        #TODO: render_img is super messy, clean it up
        if (idx % self.freq == 0) and (iter % self.inside_freq == 0) and stage != 'coarse':
            idx = idx - offset
            with torch.no_grad():
                if only_semantic == False:
                    if gt_semantic is not None:
                        #Carefull, this is the only up to date version of this function
                        gt_depth_np = gt_depth.cpu().numpy()
                        gt_color_np = gt_color.cpu().numpy() #TODO add semantics
                        gt_semantic_np = gt_semantic.cpu().numpy()
                        gt_semantic_np = np.argmax(gt_semantic_np, axis=2)
                        if len(c2w_or_camera_tensor.shape) == 1:
                            bottom = torch.from_numpy(
                                np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                                    torch.float32).to(self.device)
                            c2w = get_camera_from_tensor(
                                c2w_or_camera_tensor.clone().detach())
                            c2w = torch.cat([c2w, bottom], dim=0)
                        else:
                            c2w = c2w_or_camera_tensor

                        depth, uncertainty, color, semantic = self.renderer.render_img( #Done add semantics
                            c,
                            decoders,
                            c2w,
                            self.device,
                            stage='visualize',
                            gt_depth=gt_depth)
                        depth_np = depth.detach().cpu().numpy()
                        color_np = color.detach().cpu().numpy()
                        semantic_np = semantic.detach().cpu().numpy() #added
                        """print("SEMANTIC",semantic_np[0,:])
                        print("SEMANTIC",semantic_np[1,:])
                        print("SEMANTIC TYPE",type(semantic_np))
                        print("COLOR",color_np[0,0,:])"""
                        depth_residual = np.abs(gt_depth_np - depth_np)
                        depth_residual[gt_depth_np == 0.0] = 0.0
                        color_residual = np.abs(gt_color_np - color_np)
                        color_residual[gt_depth_np == 0.0] = 0.0
                        #------------------added------------------
                        semantic_argmax = np.argmax(semantic_np, axis=2)
                        print("semantic prediction: ", semantic_argmax)
                        print("pred. ids: ", np.unique(semantic_argmax))
                        #semantic_pred = np.abs(~(gt_semantic_np == semantic_argmax)) #added
                        #semantic_pred[gt_depth_np == 0.0] = -1 #not sure what is right here
                        predicted_semantic_probs = semantic_np[np.arange(semantic_np.shape[0])[:,None], np.arange(semantic_np.shape[1]), gt_semantic_np] #should contain the predicted probability of the correct instance
                        #-----------------end-added------------------
                        fig, axs = plt.subplots(3, 3) #previously 2,3
                        fig.suptitle(f'Frame: {idx:05d}, Iter: {iter:04d}')
                        fig.tight_layout()
                        max_depth = np.max(gt_depth_np)
                        axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 0].set_title('Input Depth')
                        axs[0, 0].set_xticks([])
                        axs[0, 0].set_yticks([])
                        axs[0, 1].imshow(depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 1].set_title('Generated Depth')
                        axs[0, 1].set_xticks([])
                        axs[0, 1].set_yticks([])
                        axs[0, 2].imshow(depth_residual, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 2].set_title('Depth Residual')
                        axs[0, 2].set_xticks([])
                        axs[0, 2].set_yticks([])
                        gt_color_np = np.clip(gt_color_np, 0, 1)
                        color_np = np.clip(color_np, 0, 1)
                        color_residual = np.clip(color_residual, 0, 1)
                        axs[1, 0].imshow(gt_color_np, cmap="plasma")
                        axs[1, 0].set_title('Input RGB')
                        axs[1, 0].set_xticks([])
                        axs[1, 0].set_yticks([])
                        axs[1, 1].imshow(color_np, cmap="plasma")
                        axs[1, 1].set_title('Generated RGB')
                        axs[1, 1].set_xticks([])
                        axs[1, 1].set_yticks([])
                        axs[1, 2].imshow(color_residual, cmap="plasma")
                        axs[1, 2].set_title('RGB Residual')
                        axs[1, 2].set_xticks([])
                        axs[1, 2].set_yticks([])
                        #------------------added------------------
                        axs[2,0], im = self.visualizerForIds.visualize(gt_semantic_np, ax = axs[2,0], title='Input Segmentation')
                        #axs[2, 0].imshow(gt_semantic_np, cmap="plasma", interpolation='nearest')
                        #axs[2, 0].im
                        axs[2, 0].set_title('Input Instance')
                        axs[2, 0].set_xticks([])
                        axs[2, 0].set_yticks([])
                        axs[2,1], im = self.visualizerForIds.visualize(semantic_argmax, ax=axs[2, 1], title='Generated Segmentation')
                        #axs[2, 1].imshow(semantic_argmax, cmap="plasma", interpolation='nearest')
                        #axs[2, 1].fig
                        axs[2, 1].set_title('Generated Instance')
                        axs[2, 1].set_xticks([])
                        axs[2, 1].set_yticks([])
                        img = axs[2, 2].imshow(predicted_semantic_probs, cmap='bwr', vmin=0, vmax=1)
                        fig.colorbar(img, ax = axs[2,2], label='Class Probability')
                        axs[2, 2].set_title('Residual Instance')
                        axs[2, 2].set_xticks([])
                        axs[2, 2].set_yticks([])
                        
                        """axs[2, 2].imshow(semantic_pred, cmap="plasma", interpolation='nearest')
                        axs[2, 2].set_title('Correctness of Semantic prediction')
                        axs[2, 2].set_xticks([])
                        axs[2, 2].set_yticks([])"""
                        
                        #-----------------end-added------------------
                        plt.subplots_adjust(wspace=0, hspace=0)
                        #plt.title(f'first_iter: {self.iters_first}, num_iter: {self.num_iter}')
                        writer.add_figure(f'figure/{idx:05d}_{iter:04d}', fig, idx)
                        """plt.clf()
                        fig, ax = plt.subplots()
                        img = ax.imshow(predicted_semantic_probs, cmap='bwr', vmin=0, vmax=1)
                        fig.colorbar(img, ax = ax, label='Class Probability')
                        ax.set_title('Predicted Probability of Correct Instance')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        writer.add_figure(f'figure/{idx:05d}_{iter:04d}_probs', fig, idx)"""
                        #plt.savefig(
                           # f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                        
                        plt.clf()

                        if self.verbose:
                            print(
                                f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
                    else: #normal execution without semantics 
                        assert False, "not up to date"
                        gt_depth_np = gt_depth.cpu().numpy()
                        gt_color_np = gt_color.cpu().numpy()
                        if len(c2w_or_camera_tensor.shape) == 1:
                            bottom = torch.from_numpy(
                                np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                                    torch.float32).to(self.device)
                            c2w = get_camera_from_tensor(
                                c2w_or_camera_tensor.clone().detach())
                            c2w = torch.cat([c2w, bottom], dim=0)
                        else:
                            c2w = c2w_or_camera_tensor

                        depth, uncertainty, color = self.renderer.render_img(
                            c,
                            decoders,
                            c2w,
                            self.device,
                            stage='color',
                            gt_depth=gt_depth)
                        depth_np = depth.detach().cpu().numpy()
                        color_np = color.detach().cpu().numpy()
                        depth_residual = np.abs(gt_depth_np - depth_np)
                        depth_residual[gt_depth_np == 0.0] = 0.0
                        color_residual = np.abs(gt_color_np - color_np)
                        color_residual[gt_depth_np == 0.0] = 0.0

                        fig, axs = plt.subplots(2, 3)
                        fig.tight_layout()
                        max_depth = np.max(gt_depth_np)
                        axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 0].set_title('Input Depth')
                        axs[0, 0].set_xticks([])
                        axs[0, 0].set_yticks([])
                        axs[0, 1].imshow(depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 1].set_title('Generated Depth')
                        axs[0, 1].set_xticks([])
                        axs[0, 1].set_yticks([])
                        axs[0, 2].imshow(depth_residual, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 2].set_title('Depth Residual')
                        axs[0, 2].set_xticks([])
                        axs[0, 2].set_yticks([])
                        gt_color_np = np.clip(gt_color_np, 0, 1)
                        color_np = np.clip(color_np, 0, 1)
                        color_residual = np.clip(color_residual, 0, 1)
                        axs[1, 0].imshow(gt_color_np, cmap="plasma")
                        axs[1, 0].set_title('Input RGB')
                        axs[1, 0].set_xticks([])
                        axs[1, 0].set_yticks([])
                        axs[1, 1].imshow(color_np, cmap="plasma")
                        axs[1, 1].set_title('Generated RGB')
                        axs[1, 1].set_xticks([])
                        axs[1, 1].set_yticks([])
                        axs[1, 2].imshow(color_residual, cmap="plasma")
                        axs[1, 2].set_title('RGB Residual')
                        axs[1, 2].set_xticks([])
                        axs[1, 2].set_yticks([])
                        plt.subplots_adjust(wspace=0, hspace=0)
                        #plt.title(f'first_iter: {self.iters_first}, num_iter: {self.num_iter}')
                        writer.add_figure(f'{idx:05d}_{iter:04d}', fig, idx)
                        plt.savefig(
                            f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                        
                        plt.clf()

                        if self.verbose:
                            print(
                                f'2Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')

                else: #include semantics ignore color
                    assert False, "not up to date"
                    if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                        gt_depth_np = gt_depth.cpu().numpy()
                        gt_color_np = gt_color.cpu().numpy() #Done add semantics
                        gt_semantic_np = gt_semantic.cpu().numpy()
                        gt_semantic_np = np.argmax(gt_semantic_np, axis=2)
                        if len(c2w_or_camera_tensor.shape) == 1:
                            bottom = torch.from_numpy(
                                np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                                    torch.float32).to(self.device)
                            c2w = get_camera_from_tensor(
                                c2w_or_camera_tensor.clone().detach())
                            c2w = torch.cat([c2w, bottom], dim=0)
                        else:
                            c2w = c2w_or_camera_tensor

                        depth, uncertainty, color, semantic = self.renderer.render_img( #TODO add semantics
                            c,
                            decoders,
                            c2w,
                            self.device,
                            stage='visualize_semantic',
                            gt_depth=gt_depth)
                        depth_np = depth.detach().cpu().numpy()
                        #color_np = color.detach().cpu().numpy()
                        semantic_np = semantic.detach().cpu().numpy() #added
                        depth_residual = np.abs(gt_depth_np - depth_np)
                        depth_residual[gt_depth_np == 0.0] = 0.0
                        #color_residual = np.abs(gt_color_np - color_np)
                        #color_residual[gt_depth_np == 0.0] = 0.0
                        semantic_residual = np.abs(gt_semantic_np - semantic_np) #added
                        semantic_residual[gt_depth_np == 0.0] = 0.0 #added

                        fig, axs = plt.subplots(2, 3) #previously 2,3
                        fig.tight_layout()
                        max_depth = np.max(gt_depth_np)
                        axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 0].set_title('Input Depth')
                        axs[0, 0].set_xticks([])
                        axs[0, 0].set_yticks([])
                        axs[0, 1].imshow(depth_np, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 1].set_title('Generated Depth')
                        axs[0, 1].set_xticks([])
                        axs[0, 1].set_yticks([])
                        axs[0, 2].imshow(depth_residual, cmap="plasma",
                                        vmin=0, vmax=max_depth)
                        axs[0, 2].set_title('Depth Residual')
                        axs[0, 2].set_xticks([])
                        axs[0, 2].set_yticks([])
                        """
                        gt_color_np = np.clip(gt_color_np, 0, 1)
                        color_np = np.clip(color_np, 0, 1)
                        color_residual = np.clip(color_residual, 0, 1)
                        axs[1, 0].imshow(gt_color_np, cmap="plasma")
                        axs[1, 0].set_title('Input RGB')
                        axs[1, 0].set_xticks([])
                        axs[1, 0].set_yticks([])
                        axs[1, 1].imshow(color_np, cmap="plasma")
                        axs[1, 1].set_title('Generated RGB')
                        axs[1, 1].set_xticks([])
                        axs[1, 1].set_yticks([])
                        axs[1, 2].imshow(color_residual, cmap="plasma")
                        axs[1, 2].set_title('RGB Residual')
                        axs[1, 2].set_xticks([])
                        axs[1, 2].set_yticks([])"""
                        #------------------added------------------
                        axs[1, 0].imshow(gt_semantic_np, cmap="plasma", interpolation='nearest')
                        axs[1, 0].set_title('Input Semantic')
                        axs[1, 0].set_xticks([])
                        axs[1, 0].set_yticks([])
                        axs[1, 1].imshow(semantic_np, cmap="plasma", interpolation='nearest')
                        axs[1, 1].set_title('Generated Semantic')
                        axs[1, 1].set_xticks([])
                        axs[1, 1].set_yticks([])
                        axs[1, 2].imshow(semantic_residual, cmap="plasma", interpolation='nearest')
                        axs[1, 2].set_title('Semantic Residual')
                        axs[1, 2].set_xticks([])
                        axs[1, 2].set_yticks([])
                        #-----------------end-added------------------
                        plt.subplots_adjust(wspace=0, hspace=0)
                        #plt.title(f'first_iter: {self.iters_first}, num_iter: {self.num_iter}')
                        writer.add_figure(f'{idx:05d}_{iter:04d}', fig, idx)
                        plt.savefig(
                            f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                        plt.clf()

                        if self.verbose:
                            print(
                                f'3Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
