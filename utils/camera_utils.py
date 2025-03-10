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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import json
WARNED = False
from tqdm import tqdm
import os
import psutil

def loadCam(args, id, cam_info, resolution_scale):
    if not args.load_image_on_the_fly:
        orig_w, orig_h = cam_info.image.size
        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # [C, H, W]
        # print(resized_image_rgb.shape)
        # NOTE: load SAM mask. modify -----
        if cam_info.sam_mask is not None:
            # step = int(args.resolution/2)     
            step = int(max(args.resolution, 1))
            gt_sam_mask = cam_info.sam_mask[:, ::step, ::step]  # downsample for mask
            gt_sam_mask = torch.from_numpy(gt_sam_mask)
            # print(resized_image_rgb.shape[1:])
            # print(gt_sam_mask.shape[1:])
            # align resolution
            if resized_image_rgb.shape[1] != gt_sam_mask.shape[1]:
                resolution = (gt_sam_mask.shape[2], gt_sam_mask.shape[1])   # modify -----
                resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # [C, H, W]
        else:
            gt_sam_mask = None
        if cam_info.mask_feat is not None:
            mask_feat = torch.from_numpy(cam_info.mask_feat)
        else:
            mask_feat = None
        # modify -----
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None

        # if resized_image_rgb.shape[1] == 4:
        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = None
        loaded_mask = None
        
        if cam_info.sam_mask is not None:
            # step = int(args.resolution/2)     
            step = int(max(args.resolution, 1))
            gt_sam_mask = cam_info.sam_mask[:, ::step, ::step]  # downsample for mask
            gt_sam_mask = torch.from_numpy(gt_sam_mask)
            # align resolution
            # if resized_image_rgb.shape[1] != gt_sam_mask.shape[1]:
            #     resolution = (gt_sam_mask.shape[2], gt_sam_mask.shape[1])   # modify -----
            #     resized_image_rgb = PILtoTorch(cam_info.image, resolution)  # [C, H, W]
        else:
            gt_sam_mask = None
        if cam_info.mask_feat is not None:
            mask_feat = torch.from_numpy(cam_info.mask_feat)
        else:
            mask_feat = None

    
    # print(args.resolution)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  cx=cam_info.cx/args.resolution, cy=cam_info.cy/args.resolution,
                #   cx=cam_info.cx, cy=cam_info.cy,
                  image=gt_image, depth=None, gt_alpha_mask=loaded_mask,
                  gt_sam_mask=gt_sam_mask, gt_mask_feat=mask_feat,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  mask_seg_path=cam_info.mask_seg_path, mask_feat_path=cam_info.mask_feat_path, image_path=cam_info.image_path, width=cam_info.width, height=cam_info.height)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    pbar = tqdm(enumerate(cam_infos))
    pbar.set_description(f"Loading CamInfo ({len(cam_infos)} images)...")
    for id, c in pbar:
        camera_list.append(loadCam(args, id, c, resolution_scale))
        show_dict = {'Mem': f"{(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)):.1f} GB"}
        pbar.set_postfix(show_dict)
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )