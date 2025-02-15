import copy
import math
import os
import config
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import bendplanner.BendRbtPlanner as br_planner
import matplotlib.pyplot as plt
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yumigripper

if __name__ == '__main__':
    import pickle
    import localenv.envloader as el

    # gripper = rtqhe.RobotiqHE()
    gripper = yumigripper.YumiGripper()
    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=False)
    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    # transmat4 = rm.homomat_from_posrot((.9, -.35, .78 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    transmat4 = rm.homomat_from_posrot((.45, 0, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    f = 'tri'
    fo = 'stick'

    # goal_pseq = bu.gen_polygen(3, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=4, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([(0, 0, 0), (0, .02, 0), (.02, .02, 0), (.02, .03, .02), (0, .03, 0), (0, .03, -.02)])
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0]])

    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_bendset.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_pathseq.pkl', 'rb'))

    init_pseq = [(0, 0, 0), (0, .1 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('stick_yumi')
    grasp_list = grasp_list

    fit_pseq, _, _ = bu.decimate_pseq(goal_pseq, tor=.002, toggledebug=False)
    # bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=True)
    init_rot = bu.get_init_rot(fit_pseq)

    brp.set_up(bendset, grasp_list, transmat4)
    brp.pre_grasp_reasoning()
    base.run()

    brp.show_bendresseq_withrbt(bendresseq, armjntsseq_list[0][1])
    brp.show_bendresseq(bendresseq)
    base.run()

    for g_tmp, armjntsseq in armjntsseq_list:
        _, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = g_tmp
        hndmat4 = rm.homomat_from_posrot(hnd_pos, hnd_rotmat)
        brp.gripper.fix_to(hndmat4[:3, 3], hndmat4[:3, :3])
        hnd = brp.gripper.gen_meshmodel(rgba=(0, 1, 0, .4))
        hnd.attach_to(base)
    base.run()

    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=True, toggledebug=False)
    print('Result Flag:', is_success)

    # goal_pseq, res_pseq = bu.align_with_goal(bs, goal_pseq, init_rot)
    # err, _ = bu.avg_distance_between_polylines(res_pseq, goal_pseq, toggledebug=False)
    # pickle.dump(bendresseq, open('./penta_bendresseq.pkl', 'wb'))

    bs.show(rgba=(.7, .7, .7, .7))
    bu.show_pseq(bs.pseq, rgba=(1, 0, 0, 1))
    # bu.show_pseq(bu.linear_inp3d_by_step(goal_pseq), rgba=(0, 1, 0, 1))
    base.run()
