import pickle
import bendplanner.BendSim as b_sim
import copy
import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import bendplanner.BendSim
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bendplanner.bend_utils as bu
import bendplanner.BendOpt as bopt
import bendplanner.bender_config as bconfig
import utils.math_utils as mu
from direct.stdpy import threading
import config


def align(pseq_tgt, pseq_src):
    init_rot = bu.get_init_rot(pseq_tgt)
    pseq_src = rm.homomat_transform_points(rm.homomat_from_posrot(rot=np.linalg.inv(init_rot),
                                                                  pos=np.asarray([0, 0, 0]) - pseq_src[0]), pseq_src)
    return pseq_src


if __name__ == '__main__':
    f_name = 'random_curve'
    obj_type = 'max'
    # method = 'cmaes'
    method = 'SLSQP'

    '''
    load files
    '''
    opt_max_res = pickle.load(open(f'{f_name}_{method}_{obj_type}.pkl', 'rb'))
    # goal_pseq = pickle.load(open(f'../goal/pseq/{f_name}.pkl', 'rb'))
    # goal_rotseq = None

    init_max_err_list = []
    opt_max_err_list = []
    init_avg_err_list = []
    opt_avg_err_list = []
    init_sum_err_list = []
    opt_sum_err_list = []
    time_cost_list = []

    x = []
    for k, v in opt_max_res.items():
        if int(k) > 24:
            break
        x.append(int(k))
        goal_pseq = v['goal_pseq']

        init_bendset = v['init_bendset']
        init_err = v['init_err']
        init_res_pseq = v['init_res_pseq']
        init_res_kpts = v['init_res_kpts']

        opt_bendset = v['opt_bendset']
        opt_err = v['opt_err']
        opt_res_pseq = v['opt_res_pseq']
        opt_res_kpts = v['opt_res_kpts']
        opt_time_cost = v['opt_time_cost']

        time_cost_list.append(opt_time_cost)
        init_diff = np.linalg.norm(np.asarray(init_res_kpts) - np.asarray(goal_pseq), axis=1)
        init_max_err_list.append(init_diff.max() * 1e3)
        init_sum_err_list.append(init_diff.sum() * 1e3)
        init_avg_err_list.append(init_diff.mean() * 1e3)
        if opt_err is not None:
            opt_diff = np.linalg.norm(np.asarray(opt_res_kpts) - np.asarray(goal_pseq), axis=1)
            opt_max_err_list.append(opt_diff.max() * 1e3)
            opt_sum_err_list.append(opt_diff.sum() * 1e3)
            opt_avg_err_list.append(opt_diff.mean() * 1e3)
        else:
            opt_max_err_list.append(None)
            opt_sum_err_list.append(None)
            opt_avg_err_list.append(None)
        # ax = plt.axes(projection='3d')
        # center = np.mean(goal_pseq, axis=0)
        # ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
        # ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
        # ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
        # ax.set_xlabel('X(mm)')
        # ax.set_ylabel('Y(mm)')
        # ax.set_zlabel('Z(mm)')
        # bu.plot_pseq(ax, init_res_pseq, c='r')
        # bu.plot_pseq(ax, opt_res_pseq, c='g')
        # bu.plot_pseq(ax, goal_pseq, c='black')
        # plt.show()

    fig = plt.figure(figsize=(18, 5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.grid()
    ax1.plot(x, init_max_err_list, color='r')
    ax1.plot(x, opt_max_err_list, color='g')
    ax1.set_xlabel('Num. of key point')
    ax1.set_ylabel('Max. point to point error(mm)')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.grid()
    ax2.plot(x, init_avg_err_list, color='r')
    ax2.plot(x, opt_avg_err_list, color='g')
    ax2.set_xlabel('Num. of key point')
    ax2.set_ylabel('Avg. point to point error(mm)')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.grid()
    ax3.plot(x, time_cost_list, color='black')
    ax3.set_xlabel('Num. of key point')
    ax3.set_ylabel('Time cost(s)')

    plt.show()
