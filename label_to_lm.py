import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh, normalize_pcl
from landmark import get_mesh_landmark, read_mesh_landmark
from bfm_model import load_bfm_model
from nicp import non_rigid_icp_mesh2pcl, non_rigid_icp_mesh2mesh

# name = "mean"
name = "target"


def label_to_lm():
    """
    turn label points in meshlab to point index
    """
    label_points = open("./test_data/" + name + "_picked_points_plus.pp")
    lm_list = []
    for line in label_points:
        if "point" not in line:
            continue
        line = line.split(' ')
        for l in line:
            if "x=" in l:
                x = l.split('"')[1]
            elif "y=" in l:
                y = l.split('"')[1]
            elif "z=" in l:
                z = l.split('"')[1]
        lm_list.append((float(x), float(y), float(z)))
    lm_list = np.array(lm_list)
    lm_arg_list = []
    n = 0
    with open('./test_data/' + name + '.obj') as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            string = line.split(" ")
            if string[0] == "v":
                points.append((float(string[1]), float(string[2]), float(string[3])))
            if string[0] == "vt":
                break
    # points原本为列表，需要转变为矩阵，方便处理
    points = np.array(points)

    for i in range(len(lm_list)):
        lm = lm_list[i].reshape(1, -1)
        dist = (lm - points) ** 2
        dist = np.sum(dist, axis=-1)
        lm_arg = np.argmin(dist)
        lm_arg_list.append(lm_arg)

    num_str = ""
    for l in lm_arg_list:
        num_str = num_str + str(int(l)) + '\n'
    with open("test_data/" + name + "_lm_plus.txt", 'w+') as f:
        f.write(num_str)


if __name__ == '__main__':
    with torch.no_grad():
        label_to_lm()
