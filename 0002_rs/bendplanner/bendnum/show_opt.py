import copy
import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import utils.math_utils as mu
from direct.stdpy import threading
import config
import bendplanner.BendSim as b_sim
import pickle


def align(pseq_tgt, pseq_src):
    init_rot = bu.get_init_rot(pseq_tgt)
    pseq_src = rm.homomat_transform_points(rm.homomat_from_posrot(rot=np.linalg.inv(init_rot),
                                                                  pos=np.asarray([0, 0, 0]) - pseq_src[0]), pseq_src)
    return pseq_src


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle='--', alpha=.2)


def get_fit_err(bs, goal_pseq, goal_rotseq, bend_num_range, init_pseq, init_rotseq):
    fit_max_err_list = []
    bend_max_err_list = []
    fit_avg_err_list = []
    bend_avg_err_list = []
    m_list = []
    fit_pseq_list = []
    final_pseq_list = []
    goal_pseq_list = []

    for i in range(bend_num_range[0], bend_num_range[1]):
        bs.reset(init_pseq, init_rotseq)
        fit_pseq, fit_rotseq, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=i, toggledebug=False)
        # fit_pseq, fit_rotseq, res_id = bu.decimate_pseq_by_cnt_uni(goal_pseq, cnt=i, toggledebug=False)
        init_rot = bu.get_init_rot(fit_pseq)
        init_bendset = bu.pseq2bendset(fit_pseq, bend_r=bs.bend_r, toggledebug=False)
        m_list.append(len(init_bendset))

        bs.gen_by_bendseq(init_bendset, cc=False)
        goal_pseq_trans, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)
        bs.show(rgba=(0, 1, 0, 1))

        fit_max_err, _ = bu.mindist_err(fit_pseq, goal_pseq, toggledebug=False, type='max')
        bend_max_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='max')
        fit_max_err_list.append(fit_max_err)
        bend_max_err_list.append(bend_max_err)

        fit_avg_err, _ = bu.mindist_err(fit_pseq, goal_pseq, toggledebug=False, type='avg')
        bend_avg_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='avg')
        fit_avg_err_list.append(fit_avg_err)
        bend_avg_err_list.append(bend_avg_err)
        fit_pseq_list.append(fit_pseq)
        final_pseq_list.append(bs.pseq[1:])
        goal_pseq_list.append(goal_pseq_trans)
    return fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, m_list, \
           fit_pseq_list, final_pseq_list, goal_pseq_list


def find_best_n(err_list, threshold=1.):
    min_err = np.inf
    inx = 0
    for i, v in enumerate(err_list):
        if v < min_err:
            min_err = v
            inx = i
        else:
            break
        if v < threshold:
            break
    return inx, min_err


def plot_err(fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, m_list):
    fig = plt.figure(figsize=(3.5, 20))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    ax1 = fig.add_subplot(3, 1, 1)
    grid_on(ax1)
    ax1.plot(x, fit_max_err_list, color='darkorange', linestyle="dashed")
    ax1.plot(x, bend_max_err_list, color='darkorange')
    best_n, min_err = find_best_n(bend_max_err_list, threshold=1)
    # ax1.set_ylim(0, 6)
    # ax1.set_yticks(np.arange(0, 7, 2))
    # ax1.set_yticks(np.arange(0, 7, .4), minor=True)

    print(x[best_n], min_err)
    # ax1.set_xlabel('Num. of key point')
    # ax1.set_ylabel('Max. point to point error(mm)')

    ax2 = fig.add_subplot(3, 1, 2)
    grid_on(ax2)
    ax2.plot(x, fit_avg_err_list, color='darkorange', linestyle="dashed")
    ax2.plot(x, bend_avg_err_list, color='darkorange')
    # ax2.set_ylim(0, 3)
    # ax2.set_yticks(np.arange(0, 3, 1))
    # ax2.set_yticks(np.arange(0, 3, .2), minor=True)

    best_n, min_err = find_best_n(bend_avg_err_list, threshold=.5)
    print(x[best_n], min_err)

    # ax2.set_xlabel('Num. of key point')
    # ax2.set_ylabel('Avg. point to point error(mm)')

    ax3 = fig.add_subplot(3, 1, 3)
    grid_on(ax3)
    ax3.plot(x, m_list, color='black')
    ax3.set_ylim(0, 51)
    ax3.set_yticks(np.arange(0, 51, 20))
    ax3.set_yticks(np.arange(0, 51, 4), minor=True)

    ax1.axvline(x=x[best_n], color='r', linestyle="dashed")
    ax2.axvline(x=x[best_n], color='r', linestyle="dashed")
    ax3.axvline(x=x[best_n], color='r', linestyle="dashed")
    plt.show()


def gen_ax():
    fig = plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(0, 1.5)
    ax.set_xlim(5.5, 16.5)
    ax.set_xticks(range(6, 17))
    grid_on(ax)
    return ax


if __name__ == '__main__':
    bend_num_range = (5, 50)
    x = range(bend_num_range[0], bend_num_range[1])
    f_list = ['bspl_10', 'bspl_15', 'bspl_20']
    c_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red', 'tab:cyan']
    pos_list = [0, 0, 0, 0, 0, 0]
    goal_list = pickle.load(open('./bspl_goal.pkl', 'rb'))
    ax = gen_ax()

    for idx, f in enumerate(f_list):
        print('-------')
        try:
            res_list = pickle.load(open(f'./{f}_uni_opt.pkl', 'rb'))
        except:
            continue

        eps = .05
        best_n_list = []
        avg_err_list = []
        max_err_list = []
        opt_max_err_list = []
        opt_avg_err_list = []
        for i, res_dict in enumerate(res_list):
            goal_pseq = goal_list[i]

            init_bendset = res_dict['init_bendset']
            init_err = res_dict['init_err']
            init_res_pseq = res_dict['init_res_pseq']
            init_res_kpts = res_dict['init_res_kpts']

            opt_bendset = res_dict['opt_bendset']
            opt_err = res_dict['opt_err']
            opt_res_pseq = res_dict['opt_res_pseq']
            opt_res_kpts = res_dict['opt_res_kpts']
            opt_time_cost = res_dict['opt_time_cost']
            best_n_list.append(len(init_bendset))
            avg_err_list.append(init_err)
            opt_avg_err_list.append(opt_err)
            # print(init_err, opt_err, len(opt_bendset))

            init_max_err, _ = bu.mindist_err(init_res_pseq, res_dict['goal_pseq'], toggledebug=False, type='max')
            opt_max_err, _ = bu.mindist_err(opt_res_pseq, res_dict['goal_pseq'], toggledebug=False, type='max')
            opt_max_err_list.append(opt_max_err)
            max_err_list.append(init_max_err)

            # ax = plt.axes(projection='3d')
            # center = np.asarray(goal_pseq).mean(axis=0)
            # ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
            # ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
            # ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)
            # bu.plot_pseq(ax, init_res_pseq, c='darkorange')
            # bu.plot_pseq(ax, opt_res_pseq, c='green')
            # bu.plot_pseq(ax, goal_pseq, c='k')
            # plt.show()

            # for i in range(len(bend_pseq_list)):
            #     ax = plt.axes(projection='3d')
            #     center = np.asarray(bend_pseq_list[i]).mean(axis=0)
            #     ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
            #     ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
            #     ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)
            #     bu.plot_pseq(ax, bend_pseq_list[i], c='darkorange')
            #     # bu.scatter_pseq(ax, bend_pseq_list[i][1:-2], c='r')
            #     # bu.scatter_pseq(ax, goal_pseq_list[i][:1], c='g', s=10)
            #     bu.plot_pseq(ax, goal_pseq_list[i], c='k')
            #     plt.show()

        print(best_n_list)
        ax.scatter([n + pos_list[idx] for n in best_n_list], avg_err_list, s=50, marker='x', c=c_list[idx])
        ax.scatter([n + pos_list[idx] for n in best_n_list], opt_avg_err_list, s=50, edgecolors=c_list[idx],
                   facecolor='none')
        ax.axhline(y=np.mean(avg_err_list), linestyle='--', c=c_list[idx], alpha=.5)
        ax.axhline(y=np.mean(opt_avg_err_list), linestyle='-.', c=c_list[idx], alpha=.5)
        print(np.round(np.mean(best_n_list), decimals=2), '±', np.round(np.std(best_n_list), decimals=2),
              np.round(max(best_n_list), decimals=2), np.round(min(best_n_list), decimals=2))
        print(np.round(np.mean(avg_err_list), decimals=2), '±', np.round(np.std(avg_err_list), decimals=2),
              np.round(max(avg_err_list), decimals=2), np.round(min(avg_err_list), decimals=2))
        print(np.round(np.mean(opt_avg_err_list), decimals=2), '±', np.round(np.std(opt_avg_err_list), decimals=2),
              np.round(max(opt_avg_err_list), decimals=2), np.round(min(opt_avg_err_list), decimals=2))

        print(np.round(np.mean(max_err_list), decimals=2), '±', np.round(np.std(max_err_list), decimals=2),
              np.round(max(max_err_list), decimals=2), np.round(min(max_err_list), decimals=2))
        print(np.round(np.mean(opt_max_err_list), decimals=2), '±', np.round(np.std(opt_max_err_list), decimals=2),
              np.round(max(opt_max_err_list), decimals=2), np.round(min(opt_max_err_list), decimals=2))

    plt.show()
