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

from matplotlib import pyplot as p
from scipy import optimize


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle='--', alpha=.2)


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
    # goal_pseq = pickle.load(open('goal/pseq/randomc.pkl', 'rb'))
    # goal_pseq = bu.gen_bspline(kp_num=5, length=.4, y_max=.04)
    # pickle.dump(goal_pseq, open('goal/pseq/bspl.pkl', 'wb'))
    goal_pseq = pickle.load(open('goal/pseq/helix.pkl', 'rb'))[:220]
    goal_pseq = np.asarray(goal_pseq)
    goal_rotseq = bu.get_rotseq_by_pseq(goal_pseq)

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    for i in range(5, 50):
        pseq, rotseq, res_pids = bu.decimate_pseq_by_cnt(goal_pseq, cnt=i)
        # pseq, rotseq, res_pids = bu.decimate_pseq_by_cnt_uni(goal_pseq, cnt=i)
        # pseq, rotseq, res_pids = bu.decimate_pseq_by_cnt_curvature(goal_pseq, cnt=i, toggledebug=True)
        bs.reset(init_pseq, init_rotseq)

        # ax = plt.axes(projection='3d')
        # bu.plot_pseq(ax, goal_pseq)
        # bu.plot_pseq(ax, pseq)
        # plt.show()

        init_rot = bu.get_init_rot(pseq)
        init_bendset = bu.pseq2bendset(pseq, bend_r=bs.bend_r, toggledebug=False)

        curvature_list_goal, r_list_goal, torsion_list_goal = bu.cal_curvature(goal_pseq)
        curvature_list, r_list, torsion_list = bu.cal_curvature(pseq)

        bs.gen_by_bendseq(init_bendset, cc=False)
        goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)

        ax = plt.axes(projection='3d')
        eps = .05
        center = np.asarray(goal_pseq).mean(axis=0)
        ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
        ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
        ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)
        bu.plot_pseq(ax, goal_pseq, c='k')
        bu.plot_pseq(ax, bs.pseq, c='darkorange')
        plt.show()

        # plt.plot(range(len(goal_pseq))[1:-1], np.asarray(curvature_list_goal) / 1000)
        # plt.plot(res_pids[1:-1], np.asarray(curvature_list) / 1000)
        # plt.hlines(y=1 / (bconfig.R_BEND * 1000), xmin=0, xmax=len(goal_pseq), colors='gray', linestyles='--')
        # # plt.ylim(0, .1)
        # plt.show()
