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


def _load_result_dict(res_dict):
    init_max_err_list = []
    opt_max_err_list = []
    init_avg_err_list = []
    opt_avg_err_list = []
    init_sum_err_list = []
    opt_sum_err_list = []
    time_cost_list = []
    bend_num_list = []

    x = []
    print(res_dict.keys())

    # for k, v in opt_res_dict.items():
    #     opt_res_dict[k]['init_res_kpts'] = np.asarray(v['init_res_kpts'])/1000
    #     opt_res_dict[k]['opt_res_kpts'] = np.asarray(v['opt_res_kpts'])/1000
    # pickle.dump(opt_res_dict, open(f'{f_name}', 'wb'))
    for k, v in res_dict.items():
        x.append(int(k))
        goal_pseq = v['goal_pseq']

        init_bendset = v['init_bendset']
        init_err = v['init_err']
        init_res_pseq = v['init_res_pseq']
        init_res_kpts = v['init_res_kpts']

        opt_bendset = v['opt_bendset']
        print(opt_bendset)

        opt_err = v['opt_err']
        opt_res_pseq = v['opt_res_pseq']
        opt_res_kpts = v['opt_res_kpts']
        opt_time_cost = v['opt_time_cost']

        bend_num_list.append(len(opt_bendset))
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

    index_list = np.argsort(x)
    x = np.asarray(x)[index_list]
    init_max_err_list = np.asarray(init_max_err_list)[index_list]
    opt_max_err_list = np.asarray(opt_max_err_list)[index_list]
    init_avg_err_list = np.asarray(init_avg_err_list)[index_list]
    opt_avg_err_list = np.asarray(opt_avg_err_list)[index_list]
    init_sum_err_list = np.asarray(init_sum_err_list)[index_list]
    opt_sum_err_list = np.asarray(opt_sum_err_list)[index_list]
    time_cost_list = np.asarray(time_cost_list)[index_list]
    bend_num_list = np.asarray(bend_num_list)[index_list]

    return x, init_max_err_list, opt_max_err_list, init_avg_err_list, opt_avg_err_list, \
           init_sum_err_list, opt_sum_err_list, time_cost_list, bend_num_list


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major', linestyle='-')
    ax.grid(b=True, which='minor', linestyle='--')


def show_sgl_method(f_name):
    opt_res_dict = pickle.load(open(f_name, 'rb'))
    # goal_pseq = pickle.load(open(f'../goal/pseq/{f_name}.pkl', 'rb'))
    # goal_rotseq = None

    x, init_max_err_list, opt_max_err_list, init_avg_err_list, opt_avg_err_list, \
    init_sum_err_list, opt_sum_err_list, time_cost_list, bend_num_list = _load_result_dict(opt_res_dict)

    fig = plt.figure(figsize=(18, 5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24

    ax1 = fig.add_subplot(1, 3, 1)
    grid_on(ax1)
    ax1.plot(x, init_max_err_list, color='darkorange')
    ax1.plot(x, opt_max_err_list, color='g')
    ax1.axvline(x=11, c='r', linestyle='dashed')
    # ax1.set_xlabel('Num. of key point')
    # ax1.set_ylabel('Max. point to point error(mm)')

    ax2 = fig.add_subplot(1, 3, 2)
    grid_on(ax2)
    ax2.plot(x, init_avg_err_list, color='darkorange')
    ax2.plot(x, opt_avg_err_list, color='g')
    ax2.axvline(x=11, c='r', linestyle='dashed')
    # ax2.set_xlabel('Num. of key point')
    # ax2.set_ylabel('Avg. point to point error(mm)')

    ax3 = fig.add_subplot(1, 3, 3)
    grid_on(ax3)
    ax3.plot(x, time_cost_list, color='black')
    print(bend_num_list)
    # ax3.set_xlabel('Num. of key point')
    # ax3.set_ylabel('Time cost(s)')

    plt.show()


def compare(f_name_1, f_name_2):
    opt_res_dict_1 = pickle.load(open(f_name_1, 'rb'))
    opt_res_dict_2 = pickle.load(open(f_name_2, 'rb'))
    # goal_pseq = pickle.load(open(f'../goal/pseq/{f_name}.pkl', 'rb'))
    # goal_rotseq = None
    x, init_max_err_list_1, opt_max_err_list_1, init_avg_err_list_1, opt_avg_err_list_1, \
    init_sum_err_list_1, opt_sum_err_list_1, time_cost_list_1, bend_num_list_1 = _load_result_dict(opt_res_dict_1)
    x, init_max_err_list_2, opt_max_err_list_2, init_avg_err_list_2, opt_avg_err_list_2, \
    init_sum_err_list_2, opt_sum_err_list_2, time_cost_list_2, bend_num_list_2 = _load_result_dict(opt_res_dict_2)

    fig = plt.figure(figsize=(18, 5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.grid()
    ax1.plot(x, init_max_err_list_1, color='r')
    ax1.plot(x, opt_max_err_list_1, color='g')
    ax1.plot(x, opt_max_err_list_2, color='dodgerblue')
    ax1.set_xlabel('Num. of key point')
    ax1.set_ylabel('Max. point to point error(mm)')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.grid()
    ax2.plot(x, init_avg_err_list_1, color='r')
    ax2.plot(x, opt_avg_err_list_1, color='g')
    ax2.plot(x, opt_avg_err_list_2, color='dodgerblue')
    ax2.set_xlabel('Num. of key point')
    ax2.set_ylabel('Avg. point to point error(mm)')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.grid()
    ax3.plot(x, time_cost_list_1, color='g')
    ax3.plot(x, time_cost_list_2, color='dodgerblue')
    ax3.set_xlabel('Num. of key point')
    ax3.set_ylabel('Time cost(s)')

    plt.show()


if __name__ == '__main__':
    goal_f_name = 'randomc'
    obj_type = 'avg'
    # method = 'cmaes'
    method = 'SLSQP'

    '''
    load files
    '''
    f_name = f'{goal_f_name}_{method}_{obj_type}.pkl'
    show_sgl_method(f_name)

    # compare(f'{goal_f_name}_{method}_max_10.pkl', f'{goal_f_name}_{method}_avg_10.pkl')
