import visualization.panda.world as wd
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import json
import os
import open3d as o3d

import utils.pcd_utils as pcdu
import localenv.envloader as el
import motionplanner.pcn_nbv_solver as nbv_solver
import nbv_utils as nu
import basis.o3dhelper as o3dh
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    rbt = el.loadXarm(showrbt=True)

    path = 'D:/nbv_mesh/'
    cat = 'bspl_5'
    fo = 'res_75'
    coverage_pcn = []
    coverage_org = []

    coverage_tor = .001
    toggledebug = True
    f = '0000.ply'

    cov_list = []
    cov_opt_list = []

    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        print(f'-----------------{f}-----------------')
        res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))
        o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))

        pcd_i = np.asarray(res_pcn['0']['input'])
        pcd_add = np.asarray(res_pcn['0']['add'])
        pcd_o = np.asarray(res_pcn['0']['pcn_output'])
        print('coverage:', round(res_pcn['0']['coverage'], 3))
        cov_list.append(round(res_pcn['0']['coverage'], 3))
        pcd_res = np.asarray(res_pcn['final'])
        pcd_gt = np.asarray(res_pcn['gt'])

        o3dpcd = o3dh.nparray2o3dpcd(pcd_i)
        # pcd_i = np.asarray(o3d.io.read_point_cloud(os.path.join(os.getcwd(), 'tmp', f'0_i.pcd')).points)
        # pcd_o = np.asarray(o3d.io.read_point_cloud(os.path.join(os.getcwd(), 'tmp', f'0_o.pcd')).points)
        # print(len(pcd_i))
        seedjntagls = rbt.get_jnt_values('arm')

        nbv_opt = nbv_solver.NBVOptimizer(rbt, toggledebug=False)
        trans, rot = nbv_opt.solve(seedjntagls, pcd_i, cam_pos, method='COBYLA')

        o3dpcd_tmp = nu.gen_partial_o3dpcd(o3dmesh, rot=rot, trans=trans, rot_center=(0, 0, 0))
        o3dpcd += o3dpcd_tmp

        coverage = pcdu.cal_coverage(np.asarray(o3dpcd.points), pcd_gt, tor=coverage_tor)
        print('coverage(opt):', round(coverage, 3))
        cov_opt_list.append(round(coverage, 3))

        # pcdu.show_pcd(pcd_i)
        # pcdu.show_pcd(pcd_o, rgba=(1, 1, 0, .5))
        # pcdu.show_pcd(pcdu.trans_pcd(pcd_i, transmat4), rgba=(1, 0, 0, 1))
        # base.run()

    print(cov_list)
    print(cov_opt_list)
