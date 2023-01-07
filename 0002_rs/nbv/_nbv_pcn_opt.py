import visualization.panda.world as wd
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import json
import os
import open3d as o3d
import motionplanner.robot_helper as rbt_helper
import modeling.geometric_model as gm
import basis.robot_math as rm
import copy
import utils.pcd_utils as pcdu
import localenv.envloader as el
import motionplanner.nbv_solver as nbv_solver

if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    rbt = el.loadXarm(showrbt=False)

    path = 'D:/nbv_mesh/'
    cat = 'bspl'
    fo = 'res_75'
    coverage_pcn = []
    coverage_org = []

    coverage_tor = .001
    toggledebug = True
    f = '0002.ply'

    res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))

    pcd_i = np.asarray(res_pcn['0']['input'])
    pcd_add = np.asarray(res_pcn['0']['add'])
    pcd_o = np.asarray(res_pcn['0']['pcn_output'])
    pcd_res = np.asarray(res_pcn['final'])
    pcd_gt = np.asarray(res_pcn['gt'])
    # pcd_i = np.asarray(o3d.io.read_point_cloud(os.path.join(os.getcwd(), 'tmp', f'0_i.pcd')).points)
    # pcd_o = np.asarray(o3d.io.read_point_cloud(os.path.join(os.getcwd(), 'tmp', f'0_o.pcd')).points)
    print(len(pcd_i))
    seedjntagls = rbt.get_jnt_values('arm')
    nbs_opt = nbv_solver.NBVOptimizer(rbt)
    _, transmat4, _ = nbs_opt.solve(seedjntagls, pcd_i, cam_pos)
    # transmat4 = np.asarray([[-0.52302792, 0.81827995, 0.23845274, 0.],
    #                         [0.81827995, 0.56036126, -0.12811393, 0.],
    #                         [-0.23845274, 0.12811393, -0.96266667, 0.],
    #                         [0., 0., 0., 1., ]])
    print(transmat4)
    # pcdu.show_pcd(pcd_i)
    # pcdu.show_pcd(pcd_o, rgba=(1, 1, 0, .5))
    # pcdu.show_pcd(pcdu.trans_pcd(pcd_i, transmat4), rgba=(1, 0, 0, 1))
    #
    # base.run()
