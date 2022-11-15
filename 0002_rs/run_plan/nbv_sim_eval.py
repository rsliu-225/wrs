import copy
import os
import pickle
import json
import matplotlib.pyplot as plt
import math
import basis.o3dhelper as o3dh
import random
import numpy as np
import open3d as o3d

import basis.robot_math as rm
# import localenv.envloader as el
import modeling.geometric_model as gm
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd
import pcn.inference as pcn
import datagenerator.data_utils as du


def transpose(data):
    mat = [[]]
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            if len(mat) > j:
                mat[j].append(v)
            else:
                mat.append([v])
    return mat


if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'E:/liu/nbv_mesh/'
    cat = 'plat'
    coverage_pcn = []
    coverage_org = []
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        print(f'-----------{f}------------')
        try:
            res_pcn = json.load(open(os.path.join(path, cat, 'res', f'pcn_{f.split(".ply")[0]}.json'), 'rb'))
            res_org = json.load(open(os.path.join(path, cat, 'res', f'org_{f.split(".ply")[0]}.json'), 'rb'))
        except:
            break
        pcd_gt = res_pcn['gt']
        coverage_pcn_tmp = [res_pcn['init_coverage']]
        coverage_org_tmp = [res_org['init_coverage']]
        for k, v in res_pcn.items():
            if k == 'gt' or k == 'init_coverage' or k == 'final':
                continue
            print('org', k, v['coverage'])
            coverage_pcn_tmp.append(v['coverage'])
        for k, v in res_org.items():
            if k == 'gt' or k == 'init_coverage' or k == 'final':
                continue
            print('pcn', k, v['coverage'])
            coverage_org_tmp.append(v['coverage'])

        coverage_pcn.append(coverage_pcn_tmp)
        coverage_org.append(coverage_org_tmp)

    coverage_pcn = transpose(coverage_pcn)
    coverage_org = transpose(coverage_org)

    cnt_pcn = []
    cnt_org = []
    print('org')
    for i, r in enumerate(coverage_org):
        print(i, len(r))
        cnt_org.append(len(r))
    print('pcn')
    for i, r in enumerate(coverage_pcn):
        print(i, len(r))
        cnt_pcn.append(len(r))
    x = [0, 1, 2, 3, 4, 5]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    # ax1.set_xticks(x)
    # ax2.set_xticks(x)
    ax3.set_xticks(x)
    ax1.set_title('Original Method')
    ax1.boxplot(coverage_org)
    ax2.set_title('PCN')
    ax2.boxplot(coverage_pcn)
    ax3.set_title('')
    ax3.plot(x, cnt_org)
    ax3.plot(x, cnt_pcn)
    plt.show()
    # base.run()
