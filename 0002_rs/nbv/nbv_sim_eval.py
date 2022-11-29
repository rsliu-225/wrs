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


def load_cov(prefix='pcn'):
    cov_list = []
    max_list = []
    cnt_list = [0] * 5
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        print(f'-----------{f}------------')
        try:
            res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
        except:
            break
        pcd_gt = res_dict['gt']
        cov_list_tmp = [res_dict['init_coverage']]
        max_tmp = [res_dict['init_coverage']]
        max = 0
        for i in range(5):
            if str(i) in res_dict.keys():
                print(prefix, i, res_dict[str(i)]['coverage'])
                cov_list_tmp.append(res_dict[str(i)]['coverage'])
                max = res_dict[str(i)]['coverage']
                cnt_list[i] += 1
            max_tmp.append(max)
        max_list.append(max_tmp)
        cov_list.append(cov_list_tmp)
    return cov_list, max_list, [cnt_list[0]] + cnt_list


def plot_box(ax, data, clr, positions):
    box = ax.boxplot(data, positions=positions)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box[item], color=clr)
    # plt.setp(box["boxes"], facecolor=clr)
    plt.setp(box["fliers"], markeredgecolor=clr)


if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'E:/liu/nbv_mesh/'
    cat = 'bspl'
    fo = 'res_75'
    coverage_pcn, max_pcn, cnt_pcn = load_cov(prefix='pcn')
    coverage_org, max_org, cnt_org = load_cov(prefix='org')
    coverage_random, max_random, cnt_random = load_cov(prefix='random')

    coverage_pcn = transpose(coverage_pcn)
    coverage_org = transpose(coverage_org)
    coverage_random = transpose(coverage_random)

    max_pcn = transpose(max_pcn)
    max_org = transpose(max_org)
    max_random = transpose(max_random)

    x = [0, 1, 2, 3, 4, 5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20

    ax1.set_title('Coverage')
    # ax1.grid()
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax1.set_xticks([3 * v + 1 for v in x], labels=['0', '1', '2', '3', '4', '5'])
    ax1.axhline(y=.95)
    ax2.set_title('Num. of Attempts')
    ax2.grid()
    ax2.set_xticks(x)

    # plot_box(ax1, coverage_random, 'tab:blue', positions=[3 * v + .25 for v in x])
    # plot_box(ax1, coverage_org, 'tab:orange', positions=[3 * v + 1 for v in x])
    # plot_box(ax1, coverage_pcn, 'tab:green', positions=[3 * v + 2 - .25 for v in x])
    plot_box(ax1, max_random, 'tab:blue', positions=[3 * v + .25 for v in x])
    plot_box(ax1, max_org, 'tab:orange', positions=[3 * v + 1 for v in x])
    plot_box(ax1, max_pcn, 'tab:green', positions=[3 * v + 2 - .25 for v in x])

    ax2.plot(x, cnt_random)
    ax2.plot(x, cnt_org)
    ax2.plot(x, cnt_pcn)

    plt.show()
    # base.run()
