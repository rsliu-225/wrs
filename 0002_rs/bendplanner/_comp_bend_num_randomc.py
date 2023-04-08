import copy
import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import BendSim
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bend_utils as bu
import bender_config as bconfig
import utils.math_utils as mu
from direct.stdpy import threading
import config
import bendplanner.BendSim as b_sim
import pickle


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle='--', alpha=.2)


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


def get_fit_err(bs, goal_pseq, goal_rotseq, bend_num_range, init_pseq, init_rotseq):
    fit_max_err_list = []
    bend_max_err_list = []
    fit_avg_err_list = []
    bend_avg_err_list = []
    m_list = []
    fit_pseq_list = []
    bend_pseq_list = []
    goal_pseq_list = []

    for i in range(bend_num_range[0], bend_num_range[1]):
        bs.reset(init_pseq, init_rotseq)
        # fit_pseq, fit_rotseq, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=i, toggledebug=False)
        # fit_pseq, fit_rotseq, res_id = bu.decimate_pseq_by_cnt_uni(goal_pseq, cnt=i, toggledebug=False)
        fit_pseq, fit_rotseq, res_id = bu.decimate_pseq_by_cnt_curvature(goal_pseq, cnt=i, toggledebug=False)
        init_rot = bu.get_init_rot(fit_pseq)
        init_bendset = bu.pseq2bendset(fit_pseq, bend_r=bs.bend_r, toggledebug=False)
        m_list.append(len(init_bendset))

        bs.gen_by_bendseq(init_bendset, cc=False)
        goal_pseq_trans, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)
        fit_pseq, fit_rotseq = bu.align_with_init(bs, fit_pseq, init_rot, fit_rotseq)
        bs.show(rgba=(0, 1, 0, 1))

        fit_max_err, _ = bu.mindist_err(fit_pseq, goal_pseq_trans, toggledebug=False, type='max')
        bend_max_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='max')
        fit_max_err_list.append(fit_max_err)
        bend_max_err_list.append(bend_max_err)

        fit_avg_err, _ = bu.mindist_err(fit_pseq, goal_pseq_trans, toggledebug=False, type='avg')
        bend_avg_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='avg')
        fit_avg_err_list.append(fit_avg_err)
        bend_avg_err_list.append(bend_avg_err)
        fit_pseq_list.append(fit_pseq)
        bend_pseq_list.append(bs.pseq[1:])
        goal_pseq_list.append(goal_pseq_trans)
    return fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, m_list, \
           fit_pseq_list, bend_pseq_list, goal_pseq_list


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


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')
    bend_num_range = (5, 50)
    x = range(bend_num_range[0], bend_num_range[1])

    snum = 20
    res_list = []
    # goal_list = []
    goal_list = pickle.load(open('./bendnum/bspl_goal.pkl', 'rb'))

    for goal_pseq in goal_list:
        # goal_pseq = bu.gen_bspline(kp_num=random.choice([4, 5, 6]), length=.2, y_max=.04)
        goal_rotseq = None
        # curvature_list, r_list, torsion_list = bu.cal_curvature(goal_pseq)
        # if min(r_list) < .005:
        #     continue

        init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
        init_rotseq = [np.eye(3), np.eye(3)]
        try:
            fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, m_list, \
            fit_pseq_list, bend_pseq_list, goal_pseq_list = \
                get_fit_err(bs, goal_pseq, goal_rotseq, bend_num_range, init_pseq, init_rotseq)
        except:
            continue

        best_n, _ = find_best_n(bend_avg_err_list, threshold=.5)
        print('Best n:', best_n + 6)
        if bend_max_err_list[best_n] > 5 or best_n > 14:
            print('Large Error!', bend_max_err_list)
            # continue

        # ax = plt.axes(projection='3d')
        # bu.plot_pseq(ax, fit_pseq_list[best_n], c='darkorange', linestyle='--')
        # bu.plot_pseq(ax, bend_pseq_list[best_n], c='darkorange')
        # bu.plot_pseq(ax, goal_pseq_list[best_n], c='k')
        # plt.show()
        # plot_err(fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, m_list)

        res_list.append([fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list,
                         m_list, fit_pseq_list, bend_pseq_list, goal_pseq_list])
        # goal_list.append(goal_pseq)

        pickle.dump(res_list, open('./bendnum/bspl_20_curv.pkl', 'wb'))
        # pickle.dump(goal_list, open('./bendnum/bspl_goal.pkl', 'wb'))
