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

    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')

    transmat4 = rm.homomat_from_posrot((.9, -.35, .78 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # goal_pseq = bu.gen_hook()
    # goal_pseq = bu.gen_helix(r=.02, lift_a=np.radians(-20), rot_num=2)
    # goal_pseq = bu.gen_sprial(200)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([(0, 0, 0), (0, .2, 0), (.2, .2, 0), (.2, .3, .2), (0, .3, 0), (0, .3, -.2)])
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]]) * .4
    # goal_pseq = np.asarray([[.1, 0, .1], [0, 0, .1], [0, 0, 0]]) * .4

    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.05, toggledebug=False)
    # pickle.dump(goal_pseq[40:][::-1], open(f'{config.ROOT}/bendplanner/goal/pseq/sprial.pkl', 'wb'))
    goal_pseq = pickle.load(open(f'{config.ROOT}/bendplanner/goal/pseq/randomc.pkl', 'rb'))

    ax = plt.axes(projection='3d')
    ax.plot3D(goal_pseq[:, 0], goal_pseq[:, 1], goal_pseq[:, 2], color='black')
    plt.show()

    fit_pseq, fit_rotseq, _ = bu.decimate_pseq(goal_pseq, tor=.002, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    # for b in bendset:
    #     print(b)
    init_rot = bu.get_init_rot(fit_pseq)

    bs.reset(init_pseq, init_rotseq, extend=True)
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=True, toggledebug=False)
    bs.show_bendresseq(bendresseq, is_success)
    base.run()

    goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot)
    fit_pseq, _ = bu.align_with_init(bs, fit_pseq, init_rot)
    goal_cm = bu.gen_stick(fit_pseq, fit_rotseq, bconfig.THICKNESS / 2)
    goal_cm.attach_to(base)
    # err, _ = bu.avg_polylines_dist_err(res_pseq, goal_pseq, toggledebug=True)
    err, kpts2 = bu.mindist_err(bs.pseq[1:], goal_pseq, toggledebug=True)

    res_pseq_tmp = bs.pseq[1:]
    ax = plt.axes(projection='3d')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.2, 0.8])
    ax.set_zlim([-0.5, 0.5])
    # bu.plot_pseq(ax, res_pseq, c='r')
    bu.plot_pseq(ax, res_pseq_tmp, c='g')
    bu.plot_pseq(ax, goal_pseq, c='black')
    plt.show()
    bs.show(rgba=(0, .7, 0, .7), show_frame=True)

    base.run()
