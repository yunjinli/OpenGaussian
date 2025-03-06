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
import numpy as np
from PIL import Image
import json
from utils.opengs_utlis import mask_feature_mean, get_SAM_mask_and_feat, load_code_book
import pytorch3d.ops
from scene import DeformModel


np.random.seed(42)
colors_defined = np.random.randint(100, 256, size=(300, 3))
colors_defined[0] = np.array([0, 0, 0])
colors_defined = torch.from_numpy(colors_defined)

def get_pixel_values(image_path, position, radius=10):
    # 打开图像并确保图像是 RGB 模式
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        
        # 获取指定位置的像素值范围
        left = max(position[0] - radius, 0)
        right = min(position[0] + radius + 1, width)
        top = max(position[1] - radius, 0)
        bottom = min(position[1] + radius + 1, height)
        
        # 提取范围内的像素值
        pixels = []
        for x in range(left, right):
            for y in range(top, bottom):
                pixels.append(img.getpixel((x, y)))
        
        # 计算像素值的均值
        pixels_array = np.array(pixels)
        mean_pixel = pixels_array.mean(axis=0)
    
    return tuple(mean_pixel)

def compute_click_values(model_path, image_name, pix_xy, radius=5):
    def compute_level_click_val(iter, model_path, image_name, pix_xy, radius):
        img_path1 = f"{model_path}/test/ours_{iter}/renders_ins_feat1/{image_name}.png"
        img_path2 = f"{model_path}/test/ours_{iter}/renders_ins_feat2/{image_name}.png"
        val1 = get_pixel_values(img_path1, pix_xy, radius)
        val2 = get_pixel_values(img_path2, pix_xy, radius)
        click_val = (torch.tensor(list(val1) + list(val2)) / 255) * 2 - 1
        return click_val
    
    level1_click_val = compute_level_click_val(50000, model_path, image_name, pix_xy, radius)
    level2_click_val = compute_level_click_val(70000, model_path, image_name, pix_xy, radius)
    
    return level1_click_val, level2_click_val

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, deform):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    segment_objects_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segment_objects")
    pred_masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pred_masks")
    # render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    # gt_sam_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_sam_mask")
    # pseudo_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "pseudo_ins_feat")

    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    # makedirs(render_ins_feat_path, exist_ok=True)
    # makedirs(gt_sam_mask_path, exist_ok=True)
    # makedirs(pseudo_ins_feat_path, exist_ok=True)
    makedirs(segment_objects_path, exist_ok=True)
    makedirs(pred_masks_path, exist_ok=True)

    # load codebook
    root_code_book, root_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "root_code_book"))
    leaf_code_book, leaf_cluster_indices = load_code_book(os.path.join(model_path, "point_cloud", \
        f'iteration_{iteration}', "leaf_code_book"))
    root_cluster_indices = torch.from_numpy(root_cluster_indices).cuda()
    leaf_cluster_indices = torch.from_numpy(leaf_cluster_indices).cuda()
    # counts = torch.bincount(torch.from_numpy(cluster_indices), minlength=64)

    # load the saved codebook(leaf id) and instance-level language feature
    # 'leaf_feat', 'leaf_acore', 'occu_count', 'leaf_ind'       leaf_figurines_cluster_lang
    # mapping_file = os.path.join(model_path, "cluster_lang.npz")
    # saved_data = np.load(mapping_file)
    # leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()    # [num_leaf=640, 512] 每个实例的语言特征
    # leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()       # [num_leaf=640] 每个实例的得分
    # leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()  # [num_leaf=640] 每个实例在视图中出现的次数
    # leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()           # [num_pts] 每个点云对应的实例 id
    # leaf_lang_feat[leaf_occu_count < 5] *= 0.0      # 出现次数太少的聚类不考虑
    # leaf_cluster_indices = leaf_ind
    
    image_name = "00000"
    # # object_name = "apple"
    # pix_xy = (450, 217) # bag of cookies
    # pix_xy = (344, 350) # apple
    # # teatime       image_name = "frame_00002"
    # object_names = ["bear nose", "stuffed bear", "sheep", "bag of cookies", \
    #                 "plate", "three cookies", "tea in a glass", "apple", \
    #                 "coffee mug", "coffee", "paper napkin"]
    # pix_xy_list = [ (740, 80), (800, 160), (80, 240), (450, 200),
    #                 (468, 288), (438, 273), (309, 308), (343, 361),
    #                 (578, 274), (571, 260), (565, 380)]
    # figurines   image_name = "frame_00002"
    # object_names = ["rubber duck with buoy", "porcelain hand", "miffy", "toy elephant", "toy cat statue", \
    #                 "jake", "Play-Doh bucket", "rubber duck with hat", "rubics cube", "waldo", \
    #                 "twizzlers", "red toy chair", "green toy chair", "pink ice cream", "spatula", \
    #                 "pikachu", "green apple", "rabbit", "old camera", "pumpkin", \
    #                 "tesla door handle"]
    if args.points is not None:
        pix_xy_list = [eval(point) for point in args.points] if args.points is not None else None
    else:
        pix_xy_list = None
    # pix_xy_list = [ (103, 378), (552, 390), (896, 342), (720, 257), (254, 297),
    #                 (451, 197), (626, 256), (760, 166), (781, 243), (896, 136),
    #                 (927, 241), (688, 148), (538, 160), (565, 238), (575, 257),
    #                 (377, 156), (156, 244), (21, 237), (283, 152), (330, 200),
    #                 (514, 200)]
    # # ramen           image_name = "frame_00002"
    # object_names = ["clouth", "sake cup", "chopsticks", "spoon", "plate", \
    #                 "bowl", "egg", "nori", "glass of water", "napkin"]
    # pix_xy_list = [(345, 38), (276, 424), (361, 370), (419, 285), (688, 412),
    #                (489, 119), (694, 187), (810, 154), (939, 289), (428, 462)]
    # # waldo_kitchen     image_name = "frame_00001"
    # object_names = ["knife", "pour-over vessel", "glass pot1", "glass pot2", "toaster", \
    #                 "hot water pot", "metal can", "cabinet", "ottolenghi", "waldo"]
    # pix_xy_list = [(439, 76), (410, 297), (306, 127), (349, 182), (261, 256),
    #                (201, 262), (161, 267), (80, 34), (17, 141), (76, 169)]

    # for o_i, object in enumerate(object_names):
    pre_pts_mask_final = None
    all_click_leaf_indices = []
    for p_i, pix_xy in enumerate(pix_xy_list):
        # pix_xy = pix_xy_list[o_i]
        root_click_val, leaf_click_val = compute_click_values(model_path, image_name, pix_xy)
    
        # 计算离两层码本最近的聚类
        distances_root = torch.norm(root_click_val - root_code_book["ins_feat"][:, :-3].cpu(), dim=1)
        distances_leaf = torch.norm(leaf_click_val - leaf_code_book["ins_feat"][:-1, :].cpu(), dim=1)
        distances_leaf[leaf_code_book["ins_feat"][:-1].sum(-1) == 0] = 999  # 没有被分配的节点，dis 设置为一个大值
        
        # 筛选出所选中的根节点的对应的候选子节点
        min_index_root = torch.argmin(distances_root).item()
        leaf_num = (leaf_code_book["ins_feat"].shape[0] - 1) / root_code_book["ins_feat"].shape[0]
        start_id = int(min_index_root*leaf_num)
        end_id = int((min_index_root + 1)*leaf_num)
        distances_leaf_sub = distances_leaf[start_id: end_id]   # [10]

        # # (1) 选择出满足要求的多个子节点
        # click_leaf_indices = torch.nonzero(distances_leaf_sub < 0.9).squeeze() + start_id
        # if (click_leaf_indices.dim() == 0) and click_leaf_indices.numel() != 0:
        #     click_leaf_indices = click_leaf_indices.unsqueeze(0) 
        # elif click_leaf_indices.numel() == 0:
        #     click_leaf_indices = torch.argmin(distances_leaf_sub).unsqueeze(0)
        # (2) 先定位 root 码本，再找到内部最接近的 1 个叶子节点
        click_leaf_indices = torch.argmin(distances_leaf_sub).unsqueeze(0) + start_id
        # print(click_leaf_indices)
        all_click_leaf_indices.append(click_leaf_indices)
        # (3) 直接找最小的子节点，不准确
        # click_leaf_indices = torch.argmin(distances_leaf).unsqueeze(0)
        # # (4) 直接指定子节点
        # click_leaf_indices = torch.tensor([60, 66])     # 64 picachu, 60, 66 toy elephant, 65 jake, 633 green apple, 639 duck
        
        # 获得子节点对应的mask
        pre_pts_mask = (leaf_cluster_indices.unsqueeze(1) == click_leaf_indices.cuda()).any(dim=1)
        # print(pre_pts_mask)
        # print(pre_pts_mask.shape)

        # post process  modify-----
        # post_process = True
        post_process = True
        max_time = 5
        if post_process and max_time > 0:
            nearest_k_distance = pytorch3d.ops.knn_points(
                gaussians._xyz[pre_pts_mask].unsqueeze(0),
                gaussians._xyz[pre_pts_mask].unsqueeze(0),
                K=int(pre_pts_mask.sum()**0.5) * 2,
            ).dists
            mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
            # print(std_nearest_k_distance, "std_nearest_k_distance")

            # mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + std_nearest_k_distance
            mask = nearest_k_distance.mean(dim = -1) < mean_nearest_k_distance + 0.1 * std_nearest_k_distance
            # mask = nearest_k_distance.mean(dim = -1) < 2 * mean_nearest_k_distance 

            mask = mask.squeeze()
            if pre_pts_mask is not None:
                pre_pts_mask[pre_pts_mask != 0] = mask
            max_time -= 1
            
        if pre_pts_mask_final is None:
            pre_pts_mask_final = pre_pts_mask
        else:
            pre_pts_mask_final |= pre_pts_mask
        # out_dir = "ca9c2998-e"
        # splits = ["train", "train", "train", "train", "test"]
        # frame_name_list = ["frame_00053", "frame_00066", "frame_00140", "frame_00154", "frame_00089"]
        # for f_i, frame_name in enumerate(frame_name_list):
        #     base_path = f"/mnt/disk1/codes/wuyanmin/code/OpenGaussian/output/{out_dir}/{splits[f_i]}/ours_70000/renders_cluster_silhouette"
        #     target_path = f"/mnt/disk1/codes/wuyanmin/code/OpenGaussian/output/{out_dir}/{splits[f_i]}/ours_70000/result/{frame_name}"
        #     makedirs(target_path, exist_ok=True)
        #     for _, text in enumerate(waldo_kitchen_texts):
        #         pos_feat = text_features[query_texts.index(text)].unsqueeze(0)
        #         similarity_pos = F.cosine_similarity(pos_feat, leaf_lang_feat.cpu())    # [640]
        #         top_values, top_indices = torch.topk(similarity_pos, 10)   # [num_mask]
        #         print("text: {} | cluster id: {}".format(text, top_indices[0]))
        #         ori_img_name = base_path + f"/{frame_name}_cluster_{top_indices[0].item()}.png"
        #         new_name = target_path + f"/{text}.png"
                
        #         if not os.path.exists(ori_img_name):
        #             top = 10
        #             for i in range(top):
        #                 ori_img_name = target_path + f"/{frame_name}_cluster_{top_indices[i].item()}.png"
        #                 if os.path.exists(ori_img_name):
        #                     break
        #         if not os.path.exists(ori_img_name):
        #             print(f"No file found at {ori_img_name}. Operation skipped.")
        #             continue
        #         import shutil
        #         shutil.copy2(ori_img_name, new_name)

        # render
    print("Selected cls: ", all_click_leaf_indices)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # render_pkg = render(view, gaussians, pipeline, background, iteration, rescale=False)
        
        # # figurines
        # if  view.image_name not in ["frame_00041", "frame_00105", "frame_00152", "frame_00195"]:
        #     continue
        # # teatime
        # if  view.image_name not in ["frame_00002", "frame_00025", "frame_00043", "frame_00107", "frame_00129", "frame_00140"]:
        #     continue
        # # ramen
        # if  view.image_name not in ["frame_00006", "frame_00024", "frame_00060", "frame_00065", "frame_00081", "frame_00119", "frame_00128"]:
        #     continue
        # # waldo_kitchen
        # if  view.image_name not in ["frame_00053", "frame_00066", "frame_00089", "frame_00140", "frame_00154"]:
        #     continue
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        # print(torch.stack(all_click_leaf_indices))
        # NOTE 再走一遍 render，以进行聚类，并对聚类可视化【不需要可以注释掉】--------------
        render_pkg = render(view, gaussians, pipeline, background, iteration,
                            rescale=False,                #)  # wherther to re-scale the gaussian scale
                            # cluster_idx=leaf_cluster_indices,     # root id 注释掉
                            leaf_cluster_idx=leaf_cluster_indices,            # leaf id               一起出现
                            # selected_leaf_id=click_leaf_indices.cuda(),       # 选择出的 leaf id       一起出现
                            selected_leaf_id=torch.stack(all_click_leaf_indices).squeeze(-1).cuda(),       # 选择出的 leaf id       一起出现
                            render_feat_map=True, 
                            render_cluster=False,
                            better_vis=True,
                            # pre_mask=pre_pts_mask,
                            pre_mask=pre_pts_mask_final,
                            seg_rgb=True,
                            d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling)
        rendering = render_pkg["render"]
        rendered_cluster_imgs = render_pkg["leaf_clusters_imgs"]
        occured_leaf_id = render_pkg["occured_leaf_id"]
        rendered_leaf_cluster_silhouettes = render_pkg["leaf_cluster_silhouettes"]

        # save Rendered RGB
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))

        # render_cluster_path = os.path.join(model_path, name, "ours_{}".format(iteration), "click_cluster")
        # render_cluster_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "click_cluster_mask")
        # makedirs(render_cluster_path, exist_ok=True)
        # makedirs(render_cluster_silhouette_path, exist_ok=True)
        
        # for i, img in enumerate(rendered_cluster_imgs):
        # print(len(rendered_cluster_imgs))
        torchvision.utils.save_image(rendered_cluster_imgs[0][:3,:,:], os.path.join(segment_objects_path, '{0:05d}'.format(idx) + ".png"))
        # 保存 mask
        # cluster_silhouette = rendered_leaf_cluster_silhouettes[0] > 0.8
        cluster_silhouette = rendered_leaf_cluster_silhouettes[0] > 0.5
        torchvision.utils.save_image(cluster_silhouette.unsqueeze(0).expand(3, -1, -1).to(torch.float32), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
        
        # binary_mask = rendered_cluster_imgs[0][:3,:,:].clone()
        # binary_mask[binary_mask > 0.0] = 1.0
        # torchvision.utils.save_image(binary_mask.to(torch.float32), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
        
        # buffer_image[buffer_image < 0.5] = 0
        # buffer_image[buffer_image != 0] = 1
        # print(cluster_silhouette.shape)
        # print(cluster_silhouette)
        # torchvision.utils.save_image(cluster_silhouette.to(torch.float32), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
        # a = 0
        # # 聚类 -----------------------------------------------

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
    parser.add_argument('--points', nargs='+', default=None)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)