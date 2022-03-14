import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import time
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import bendplanner.BendRbtPlanner as br_planner
import utils.panda3d_utils as p3u
import bendplanner.InvalidPermutationTree as ip_tree
from scipy import interpolate
import matplotlib.pyplot as plt
import config

if __name__ == '__main__':
    import pickle
    import localenv.envloader as el

    gripper = rtqhe.RobotiqHE()
    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()

    bs = b_sim.BendSim(show=True)
    mp = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")

    transmat4 = rm.homomat_from_posrot((.7, -.2, .78 + bconfig.BENDER_H), np.eye(3))
    f_name = 'chair'
    # goal_pseq = bu.gen_polygen(5, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([(0, 0, 0), (0, .02, 0), (.02, .02, 0), (.02, .03, .02), (0, .03, 0), (0, .03, -.02)])
    goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
                            [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
                            [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0]])

    # fit_pseq = bu.iter_fit(goal_pseq, tor=.002, toggledebug=False)
    # bendset = brp.pseq2bendset(fit_pseq, pos=.1, toggledebug=False)[::-1]
    # init_rot = brp.get_init_rot(fit_pseq)
    # pickle.dump([goal_pseq, bendset], open(f'{config.ROOT}/bendplanner/planres/{f_name}_bendseq.pkl', 'wb'))
    # goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{f_name}_bendseq.pkl', 'rb'))
    result, _, _, _, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/bendresseq/5_0.pkl', 'rb'))


    for b in bendset:
        print(b)

    init_pseq = [(0, 0, 0), (0, .1 + max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('stick')
    # grasp_list = grasp_list[:200]

    brp.set_up(bendset, grasp_list, transmat4)
    brp.run(f_name=f_name)

    bendresseq, seqs = result[-1]
    for x in np.linspace(.4, .9, 10):
        for y in np.linspace(-.5, 0, 10):
            print(x, y)
            transmat4 = rm.homomat_from_posrot((x, y, .78 + bconfig.BENDER_H), np.eye(3))

            brp.set_up(bendset, grasp_list, transmat4)
            fail_index, armjntsseq_list = brp.check_ik(bendresseq, grasp_l=0)
            if fail_index != -1:
                continue
            else:
                brp.show_bendresseq_withrbt(bendresseq, transmat4, armjntsseq_list[0][1])
                base.run()
