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


def get_fit_err(bs, goal_pseq, goal_rotseq, bend_num_range):
    fit_max_err_list = []
    bend_max_err_list = []
    fit_avg_err_list = []
    bend_avg_err_list = []

    for i in range(bend_num_range[0], bend_num_range[1]):
        bs.reset(init_pseq, init_rotseq)
        fit_pseq, fit_rotseq = bu.decimate_pseq_by_cnt(goal_pseq, cnt=i, toggledebug=False)
        init_rot = bu.get_init_rot(fit_pseq)
        init_bendset = bu.pseq2bendset(fit_pseq, bend_r=bs.bend_r, toggledebug=False)

        bs.gen_by_bendseq(init_bendset, cc=False)
        goal_pseq_trans, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq,
                                                          init_pos=np.asarray((bs.bend_r, 0, 0)))
        fit_pseq_trans, goal_rotseq = bu.align_with_init(bs, fit_pseq, init_rot, goal_rotseq,
                                                          init_pos=np.asarray((bs.bend_r, 0, 0)))
        bs.show(rgba=(0, 1, 0, 1))

        fit_max_err, _ = bu.mindist_err(fit_pseq, goal_pseq, toggledebug=False, type='max')
        bend_max_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='max')
        fit_max_err_list.append(fit_max_err)
        bend_max_err_list.append(bend_max_err)

        fit_avg_err, _ = bu.mindist_err(fit_pseq, goal_pseq, toggledebug=False, type='avg')
        bend_avg_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False, type='avg')
        fit_avg_err_list.append(fit_avg_err)
        bend_avg_err_list.append(bend_avg_err)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18
        ax = plt.axes(projection='3d')
        center = np.mean(fit_pseq_trans, axis=0)
        ax.set_xlim([center[0] - 50, center[0] + 50])
        ax.set_ylim([center[1] - 50, center[1] + 50])
        ax.set_zlim([center[2] - 50, center[2] + 50])
        # ax.set_xlabel('X(mm)')
        # ax.set_ylabel('Y(mm)')
        # ax.set_zlabel('Z(mm)')
        bu.plot_pseq(ax, fit_pseq_trans * 1000, c='darkorange', linestyle="dashed")
        bu.plot_pseq(ax, np.asarray(bs.pseq[1:]) * 1000, c='darkorange')
        bu.plot_pseq(ax, goal_pseq_trans * 1000, c='black')
        plt.show()

    return fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')
    goal_pseq = pickle.load(open('goal/pseq/randomc.pkl', 'rb'))
    goal_rotseq = None

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    bend_num_range = (11, 100)

    r = .02 / 2
    bs.set_r_center(r)
    fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list = \
        get_fit_err(bs, goal_pseq, goal_rotseq, bend_num_range)

    x = range(bend_num_range[0], bend_num_range[1])

    fig = plt.figure(figsize=(12, 5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.grid()
    ax1.plot(x, bend_max_err_list, color='#ED7D31')
    ax1.plot(x, fit_max_err_list, color='#ED7D31', linestyle="dashed")

    # ax1.set_xlabel('Num. of key point')
    # ax1.set_ylabel('Max. point to point error(mm)')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.grid()
    ax2.plot(x, bend_avg_err_list, color='#ED7D31')
    ax2.plot(x, fit_avg_err_list, color='#ED7D31', linestyle="dashed")

    # ax2.set_xlabel('Num. of key point')
    # ax2.set_ylabel('Avg. point to point error(mm)')

    plt.show()
