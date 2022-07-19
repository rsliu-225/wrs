import copy
import os
import pickle
import random

import cv2
import numpy as np
import open3d as o3d
from cv2 import aruco as aruco

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import utils.recons_utils as rcu
import visualization.panda.world as wd
import localenv.envloader as el
import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    # fo = 'nbc_pcn/plate_a_cubic'
    fo = 'nbc/plate_a_cubic'
    # fo = 'opti/plate_a_quadratic'
    # fo = 'seq/plate_a_quadratic'

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, .0155)

    x_range = (.06, .215)
    y_range = (-.15, .15)
    z_range = (.0155, .2)
    # z_range = (-.2, -.0155)
    # gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=False)
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1]]
    for i, pcd in enumerate(pcd_cropped_list):
        pcdu.show_pcd(pcd, rgba=rgba_list[i])

    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)

    base.run()
