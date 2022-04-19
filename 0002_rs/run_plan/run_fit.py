import copy
import math
import pickle

import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import time
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import bendplanner.BendRbtPlanner as br_planner
import utils.panda3d_utils as p3u
import bendplanner.InvalidPermutationTree as ip_tree
import config
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])

    bs = b_sim.BendSim(show=True, granularity=np.pi / 30)

    transmat4 = rm.homomat_from_posrot((.9, -.35, .78 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # goal_pseq = bu.gen_polygen(5, .01)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.05, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([(0, 0, 0), (0, .2, 0), (.2, .2, 0), (.2, .3, .2), (0, .3, 0), (0, .3, -.2)])
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]]) * .4
    # goal_pseq = np.asarray([[.1, 0, .1], [0, 0, .1], [0, 0, 0]]) * .4

    # pickle.dump(goal_pseq, open('goal_pseq.pkl', 'wb'))
    goal_pseq = pickle.load(open('goal_pseq.pkl', 'rb'))
    init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    fit_pseq = bu.decimate_pseq(goal_pseq, r=bconfig.R_CENTER, tor=.0002, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)
    init_rot = bu.get_init_rot(fit_pseq)
    bs.reset(init_pseq, init_rotseq, extend=True)
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    bs.show(rgba=(0, .7, .7, .7), show_frame=True)
    # bs.show_bendresseq(bendresseq, is_success)
    # base.run()
    goal_pseq, res_pseq = bu.align_with_goal(bs, goal_pseq, init_rot)
    fit_pseq, _ = bu.align_with_goal(bs, fit_pseq, init_rot)

    fit_rotseq = bu.get_rotseq_by_pseq(fit_pseq)
    goal_cm = bu.gen_stick(fit_pseq, fit_rotseq, bconfig.THICKNESS / 2)
    goal_cm.attach_to(base)
    # err, _ = bu.avg_polylines_dist_err(res_pseq, goal_pseq, toggledebug=True)
    kpts2 = bu.mindist_err(res_pseq, goal_pseq, toggledebug=True)

    # pickle.dump(res_pseq, open('res.pkl', 'wb'))
    res_pseq_tmp = pickle.load(open('res.pkl', 'rb'))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    # z_max = max([abs(np.max(z1)), abs(np.max(z2))])
    # ax.set_xlim([0, 0.08])
    # ax.set_ylim([-0.04, 0.04])
    # ax.set_zlim([-0.04, 0.04])
    ax.set_xlim([-0.05, 0.05])
    ax.set_ylim([-0.02, 0.08])
    ax.set_zlim([-0.05, 0.05])
    bu.plot_pseq(ax, res_pseq, c='r')
    bu.plot_pseq(ax, res_pseq_tmp, c='g')
    bu.plot_pseq(ax, goal_pseq, c='black')
    plt.show()

    base.run()
