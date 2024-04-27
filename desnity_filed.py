
import torch
import numpy as np

with torch.no_grad():
        # create a new model for background
        chunk_size = int(2 ** 16)
        alpha, dense_xyz = renderer.get_dense_alpha(model)
        xyz_sampled = renderer.normalize_coordinates(dense_xyz)
        outputs_semantic = []
        outputs_instance = []
        output_colors = []
        for chunk in tqdm(torch.split(xyz_sampled.view(-1, 3), chunk_size), desc='grid_conv'):
            outputs_semantic.append(model.render_semantic_mlp(None, model.compute_semantic_feature(chunk)).cpu())
            outputs_instance.append(model.compute_instance_feature(chunk).cpu())
            dir_0 = torch.zeros_like(chunk)
            dir_a, dir_b, dir_c = dir_0.clone(), dir_0.clone(), dir_0.clone()
            dir_a[:, 0] = 1
            dir_b[:, 1] = 1
            dir_c[:, 2] = 1
            total = model.render_appearance_mlp(dir_a, model.compute_appearance_feature(chunk)).cpu() + model.render_appearance_mlp(dir_b, model.compute_appearance_feature(chunk)).cpu() + model.render_appearance_mlp(dir_c, model.compute_appearance_feature(chunk)).cpu()
            total = total / 3
            output_colors.append(total)
        sem_labels = torch.cat(outputs_semantic, dim=0).reshape([xyz_sampled.shape[0], xyz_sampled.shape[1], xyz_sampled.shape[2], -1]).argmax(dim=-1).int().transpose(0, 2).contiguous()
        colors = torch.cat(output_colors, dim=0).reshape([xyz_sampled.shape[0], xyz_sampled.shape[1], xyz_sampled.shape[2], -1]).float().transpose(0, 2).contiguous()
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()
        alpha[alpha >= alpha_thres] = 1
        alpha[alpha < alpha_thres] = 0
        mask = alpha > 0.5
        valid_xyz = dense_xyz[mask]
        distinct_colors = DistinctColors()
        semantic_bg = torch.from_numpy(np.isin(sem_labels.cpu().numpy(), [i for i, x in enumerate(get_thing_semantics("extended")) if not x])).to(sem_labels.device)
        thing_semantics = sem_labels.clone()
        thing_semantics[semantic_bg] = 0
        colored_semantics = distinct_colors.get_color_fast_numpy(sem_labels.cpu().numpy().reshape(-1)).reshape(list(sem_labels.shape) + [3])
        normalized_mc(alpha.transpose(0, 2), torch.from_numpy(colored_semantics).transpose(0, 2), renderer.bbox_aabb).export('semantics.obj')