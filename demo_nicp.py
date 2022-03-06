# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark, read_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh


def mesh():
    """
    demo for registering mesh
    estimate landmark for target meshes
    the face must face toward z axis
    the mesh or point cloud must be normalized with normalized mesh/pcl function before feed into the nicp process
    """

    with torch.no_grad():

        target_meshes = io3d.load_obj_as_mesh('./test_data/target.obj', device=device)
        target_meshes, _ = normalize_mesh(target_meshes)
        target_lm_index, lm_mask = read_mesh_landmark(device, 'test_data/target_lm_plus.txt')

        # target_meshes = io3d.load_obj_as_mesh('./test_data/pjanic.obj', device=device)
        # target_meshes, _ = normalize_mesh(target_meshes)
        # dummy_render = render.create_dummy_render([1, 0, 0], device=device)
        # target_lm_index, lm_mask = get_mesh_landmark(target_meshes, dummy_render)

        origin_meshes = io3d.load_obj_as_mesh('./test_data/mean.obj', device=device)
        origin_meshes, _ = normalize_mesh(origin_meshes)
        origin_lm_index, _ = read_mesh_landmark(device, './test_data/mean_lm_plus.txt')

        # origin_meshes, origin_lm_index = load_bfm_model(device)

        lm_mask = torch.all(lm_mask, dim=0)
        origin_lm_index_m = origin_lm_index[:, lm_mask]
        target_lm_index_m = target_lm_index[:, lm_mask]

    fine_config = json.load(open('config/fine_grain.json'))
    registered_mesh = non_rigid_icp_mesh2mesh(
        origin_meshes, target_meshes, origin_lm_index_m, target_lm_index_m, fine_config, device=device)
    if isinstance(registered_mesh, list):
        for m in range(len(registered_mesh)):
            io3d.save_meshes_as_objs(['results/result_%d.obj' % (m + 1)], registered_mesh[m], save_textures=False)
    else:
        io3d.save_meshes_as_objs(['results/result.obj'], registered_mesh, save_textures=False)


def point_cloud():
    # demo for registering point cloud
    # the mesh or point cloud must be normalized with normalized mesh/pcl function before feed into the nicp process
    pcls = io3d.load_ply_as_pointcloud('./test_data/test2.ply', device=device)
    norm_pcls, _ = normalize_pcl(pcls)
    pcl_lm_file = open('./test_data/test2_lm.txt')
    lm_list = []
    for line in pcl_lm_file:
        line = int(line.strip())
        lm_list.append(line)

    target_lm_index = torch.from_numpy(np.array(lm_list)).to(device)
    lm_mask = (target_lm_index >= 0)
    target_lm_index = target_lm_index.unsqueeze(0)
    bfm_meshes, bfm_lm_index = load_bfm_model(device)
    bfm_lm_index_m = bfm_lm_index[:, lm_mask]
    target_lm_index_m = target_lm_index[:, lm_mask]
    coarse_config = json.load(open('config/coarse_grain.json'))
    registered_mesh = non_rigid_icp_mesh2pcl(bfm_meshes, norm_pcls, bfm_lm_index_m, target_lm_index_m, coarse_config)
    io3d.save_meshes_as_objs(['results/final.obj'], registered_mesh, save_textures=False)


if __name__ == '__main__':
    device = torch.device('cuda:2')
    mesh()
