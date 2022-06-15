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

if __name__ == '__main__':

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')
    goal_pseq = pickle.load(open('goal/pseq/random_curve.pkl', 'rb'))
    goal_rotseq = None

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    fit_err_list = []
    bend_err_list = []
    for i in range(5, 15):
        bs.reset(init_pseq, init_rotseq)
        fit_pseq, fit_rotseq = bu.decimate_pseq_by_cnt(goal_pseq, cnt=i, toggledebug=False)
        init_rot = bu.get_init_rot(fit_pseq)
        init_bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)
        bs.gen_by_bendseq(init_bendset, cc=False)
        goal_pseq_trans, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)
        bs.show(rgba=(0, 1, 0, 1))

        fit_err, _ = bu.mindist_err(fit_pseq, goal_pseq, toggledebug=False)
        bend_err, _ = bu.mindist_err(bs.pseq[1:], goal_pseq_trans, toggledebug=False)

        fit_err_list.append(fit_err)
        bend_err_list.append(bend_err)

    x = np.linspace(5, 15, len(fit_err_list))
    fig, ax = plt.subplots()
    ax.grid()
    # plt.scatter(x, fit_err_list, color='r')
    ax.plot(x, fit_err_list, color='r')
    # plt.scatter(x, bend_err_list, color='g')
    ax.plot(x, bend_err_list, color='g')
    plt.show()
