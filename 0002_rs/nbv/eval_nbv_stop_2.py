import copy
import math
import os
import random
import json
import h5py
import numpy as np
import open3d as o3d

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import datagenerator.data_utils as du
import pcn.inference as pcn
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import visualization.panda.world as wd
import motionplanner.pcn_nbv_solver as nbv_solver
import nbv_utils as nu


if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_rlen/best_emd_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()

    # path = 'E:/liu/nbv_mesh/'
    path = 'D:/nbv_mesh/'
    res_fo = 'res_60_rlen'
    cat_list = ['plat']
    # cat_list = ['bspl_4']
    goal = .95
    prefix = 'pcn'
    for cat in cat_list:
        TP = []
        TN = []
        FP = []
        FN = []
        if not os.path.exists(os.path.join(path, cat, res_fo)):
            os.makedirs(os.path.join(path, cat, res_fo))
        for f in os.listdir(os.path.join(path, cat, 'mesh')):
            print(f'-----------{f}------------')
            res_dict = json.load(open(os.path.join(path, cat, res_fo,
                                                   f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))

            for i in range(5):
                if str(i - 1) in res_dict.keys() or i == 0:
                    if i == 0:
                        pcd_i = np.asarray(res_dict[str(i)]['input'])
                        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model)
                        complete = nu.is_complete(pcd_o, pcd_i)
                        gt = res_dict['init_coverage']
                    else:
                        pcd_i = np.asarray(res_dict[str(i - 1)]['input'] + res_dict[str(i - 1)]['add'])
                        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model)
                        complete = nu.is_complete(pcd_o, pcd_i)
                        gt = res_dict[str(i - 1)]['coverage']
                    print(prefix, i, gt, complete)
                    if gt >= goal and complete:
                        TP.append(gt)
                    elif gt < goal and not complete:
                        TN.append(gt)
                    elif gt < goal and complete:
                        FP.append(gt)
                    elif gt >= goal and not complete:
                        FN.append(gt)
        print(f'-----------{cat}------------')
        print('TP', len(TP), np.asarray(TP).mean(), TP)
        print('TF', len(TN), np.asarray(TN).mean(), TN)
        print('FP', len(FP), np.asarray(FP).mean(), FP)
        print('FN', len(FN), np.asarray(FN).mean(), FN)
        precision = (len(TP) + len(TN)) / (len(TP) + len(TN) + len(FP) + len(FN))
        acc = len(TP) / (len(TP) + len(FP))
        recall = len(TP) / (len(TP) + len(FN))
        print(round(precision, 3), round(acc, 3), round(recall, 3))
