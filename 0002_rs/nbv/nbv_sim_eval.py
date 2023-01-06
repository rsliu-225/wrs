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
import matplotlib.ticker as mticker
import nbv_utils as nbv_utl

if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    # path = 'E:/liu/nbv_mesh/'
    path = 'D:/nbv_mesh/'
    cat = 'plat'
    fo = 'res_75'
    coverage_pcn, max_pcn, cnt_pcn = nbv_utl.load_cov(path, cat, fo, prefix='pcn')
    coverage_org, max_org, cnt_org = nbv_utl.load_cov(path, cat, fo, prefix='org')
    coverage_rnd, max_rnd, cnt_rnd = nbv_utl.load_cov(path, cat, fo, prefix='random')

    coverage_pcn = nbv_utl.transpose(coverage_pcn)
    coverage_org = nbv_utl.transpose(coverage_org)
    coverage_rnd = nbv_utl.transpose(coverage_rnd)
    max_pcn = nbv_utl.transpose(max_pcn)
    max_org = nbv_utl.transpose(max_org)
    max_rnd = nbv_utl.transpose(max_rnd)

    x = [0, 1, 2, 3, 4, 5]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 22))
    ax1.set_title('Coverage')
    ax1.axhline(y=.95, color='r', linewidth='0.5', linestyle=':')
    ax2.set_title('Num. of Attempts')

    # plot_box(ax1, coverage_rnd, 'tab:blue', positions=[3 * v + .25 for v in x])
    # plot_box(ax1, coverage_org, 'tab:orange', positions=[3 * v + 1 for v in x])
    # plot_box(ax1, coverage_pcn, 'tab:green', positions=[3 * v + 2 - .25 for v in x])
    nbv_utl.plot_box(ax1, max_rnd, 'tab:blue', positions=[3 * v + .25 for v in x])
    nbv_utl.plot_box(ax1, max_org, 'tab:orange', positions=[3 * v + 1 for v in x])
    nbv_utl.plot_box(ax1, max_pcn, 'tab:green', positions=[3 * v + 2 - .25 for v in x])

    ax1.set_xticks([3 * v + 1 for v in x])
    ax1.set_xticklabels(x)
    ax1.set_yticks(np.linspace(.3, 1, 7))

    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(base=1 / 5))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(base=1 / 20))

    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    ax2.set_xticks(x)
    ax2.set_yticks(np.linspace(0, 100, 10))
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(base=100 / 5))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(base=100 / 20))
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    ax2.plot(x, cnt_rnd)
    ax2.plot(x, cnt_org)
    ax2.plot(x, cnt_pcn)

    print(nbv_utl.cal_avg(cnt_rnd))
    print(nbv_utl.cal_avg(cnt_org))
    print(nbv_utl.cal_avg(cnt_pcn))

    plt.show()
    # base.run()
