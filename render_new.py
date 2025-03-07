#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import DeformModel

import numpy as np
from utils.opengs_utlis import get_SAM_mask_and_feat, load_code_book
import cv2
# Randomly initialize 300 colors for visualizing the SAM mask. [OpenGaussian]
np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0]) # Ignore the mask ID of -1 and set it to black.
colors_defined = torch.from_numpy(colors_defined)

def get_feature(x, y, view, gaussians, pipeline, background, scaling_modifier, override_color, d_xyz, d_rotation, d_scaling, patch=None):
    with torch.no_grad():
        render_feature_dino_pkg = render(view, gaussians, pipeline, background, iteration=35000, d_xyz=d_xyz, d_scaling=d_scaling, d_rotation=d_rotation)
        image_feature_dino = render_feature_dino_pkg["ins_feat"]
    if patch is None:
        return image_feature_dino[:, y, x]
    else:
        a = image_feature_dino[:, y:y+patch[1], x:x+patch[0]]
        return a.mean(dim=(1,2))
    
def calculate_selection_score(features, query_feature, score_threshold=0.7):
    features /= features.norm(dim=-1, keepdim=True)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)
    scores = features.half() @ query_feature.half()
    # print(scores.shape)
    # scores = scores[:, 0]
    mask = (scores >= score_threshold).float()
    return mask
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    render_ins_feat_path1 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat1")
    render_ins_feat_path2 = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat2")
    # gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")
    # segment_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment_objects")
    # pred_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred_masks")
    
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_ins_feat_path1, exist_ok=True)
    makedirs(render_ins_feat_path2, exist_ok=True)
    # makedirs(segment_objects_path, exist_ok=True)
    # makedirs(pred_masks_path, exist_ok=True)

    # load codebook
    # root_code_book, root_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
    #     f'iteration_{iteration}', "root_code_book"))
    # leaf_code_book, leaf_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
    #     f'iteration_{iteration}', "leaf_code_book"))
    # root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    # leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    
    # mapping_file = os.path.join(model_path, "cluster_lang.npz")
    # saved_data = np.load(mapping_file)
    # leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=k1*k2, 512] cluster lang feat
    # leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=k1*k2] cluster score
    # leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=k1*k2] 
    # leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] fine id
    # leaf_lang_feat[leaf_occu_count < 5] *= 0.0      # Filter out clusters that occur too infrequently.
    # leaf_cluster_indices = leaf_ind
    
    # root_num = root_cluster_indices.max() + 1
    # leaf_num = leaf_lang_feat.shape[0] / root_num
    
    # print(leaf_cluster_indices.shape)
    # print(leaf_cluster_indices)
    # print(leaf_lang_feat.shape)
    # print(leaf_lang_feat)
    # makedirs(gt_sam_mask_path, exist_ok=True)

    # if args.points is not None:
    #     points = [eval(point) for point in args.points] if args.points is not None else None
    # else:
    #     points = None
    # if points is not None:
    #     thetas = [0.8 for i in range(len(points))]
        
    # selected_indices = None
    # if points is not None:
    #     selected_indices = []
    #     view = views[0]
    #     fid = view.fid
    #     xyz = gaussians.get_xyz
    #     time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
    #     d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
    #     ins_feat = gaussians.get_ins_feat()
    #     for i in range(len(points)):
    #         query_feature = get_feature(points[i][0], points[i][1], view, gaussians, pipeline, background, 1.0, override_color=None,
    #                                      d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, patch = None)
    #         mask = calculate_selection_score(ins_feat, query_feature, score_threshold = thetas[i])
    #         indices_above_threshold = np.where(mask.cpu().numpy() >= thetas[i])[0]
    #         selected_indices.append(indices_above_threshold)
    #     selected_indices = np.concatenate(selected_indices, axis=0)
    #     selected_indices = np.unique(selected_indices)
    # root_num = root_cluster_indices.max() + 1
    # leaf_num = (leaf_cluster_indices.max() + 1) / root_num
    # render
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if not view.data_on_gpu:
            view.to_gpu()
        fid = view.fid
        # print(fid)
        
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # if idx == 0:
        #     cur_xyz = xyz + d_xyz
        #     torch.save(cur_xyz, os.path.join(model_path, "point_cloud", \
        #     f'iteration_{iteration}', 'xyz_0.pt'))
        render_pkg = render(view, gaussians, pipeline, background, iteration, rescale=False, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        # render target object
        # render_pkg = render(view, gaussians, pipeline, background, iteration,
        #                     rescale=False,                #)  # wherther to re-scale the gaussian scale
        #                     cluster_idx=root_cluster_indices,     # root id
        #                     leaf_cluster_idx=leaf_cluster_indices,  # leaf id
        #                     # selected_leaf_id=text_leaf_indices.cuda(),  # text query 所选择的 leaf id
        #                     # render_feat_map=True, 
        #                     render_cluster=True,
        #                     # better_vis=True,
        #                     seg_rgb=True,
        #                     # post_process=True,
        #                     # root_num=root_num, leaf_num=leaf_num, 
        #                     d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        # RGB
        rendering = render_pkg["render"]
        
        try:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        except:
            print("no GT...")
        # print(render_pkg['leaf_clusters_imgs'][50].shape)
        # for i in range(int(root_num * leaf_num)):
        # print(root_num * leaf_num)
        # opencvImage = cv2.cvtColor(render_pkg['leaf_clusters_imgs'][27][:3, :, :].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Training View', opencvImage)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        # ins_feat
        rendered_ins_feat = render_pkg["ins_feat"]
        # gt_sam_mask = view.original_sam_mask.cuda()    # [4, H, W]

        # Rendered RGB
        # torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        # GT RGB
        # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # ins_feat
        torchvision.utils.save_image(rendered_ins_feat[:3,:,:], os.path.join(render_ins_feat_path1, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_ins_feat[3:6,:,:], os.path.join(render_ins_feat_path2, '{0:05d}'.format(idx) + ".png"))

        # if selected_indices is not None:
        #     # print("hi")
        #     # dummy_indices = np.arange(xyz.shape[0])
        #     mask = np.zeros(xyz.shape[0])
        #     # print(mask)
        #     mask[selected_indices] = 1
        #     mask = mask.astype('bool')
        #     # print(mask.sum())
        #     buffer_image = render(view, gaussians, pipeline, background, iteration, rescale=False, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, mask=mask, override_color=torch.ones(xyz.shape[0], 3).cuda().float())['render']
        #     buffer_image[buffer_image < 0.5] = 0
        #     buffer_image[buffer_image != 0] = 1
        #     inlier_mask = buffer_image.mean(axis=0).bool()
            
        #     torchvision.utils.save_image(buffer_image.cpu(), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
            
        #     buffer_image = render(view, gaussians, pipeline, background, iteration, rescale=False, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, mask=mask)['render']
        #     buffer_image[:, ~inlier_mask] = 0
            
            
        #     # import matplotlib.pyplot as plt
        #     # plt.imshow(to8b(buffer_image).transpose(1,2,0))
        #     # plt.show()
        if view.data_on_gpu:
            view.to_cpu()
            
        #     torchvision.utils.save_image(buffer_image.cpu(), os.path.join(segment_objects_path, '{0:05d}'.format(idx) + ".png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform = DeformModel()
        deform.load_weights(dataset.model_path, iteration=iteration)
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument('--points', nargs='+', default=None)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)