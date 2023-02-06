import os

import numpy as np

import basis.robot_math as rm
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import motionplanner.motion_planner as m_planner

if __name__ == '__main__':
    import pickle
    import copy
    import localenv.envloader as el
    import modeling.geometric_model as gm

    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.7, -.2, .78 + bconfig.BENDER_H), np.eye(3))

    base, env = el.loadEnv_yumi(camp=[2.5, -2, 1.8], lookatpos=[.45, .1, .1])
    # base, env = el.loadEnv_yumi(camp=[.45, .1, 1.8], lookatpos=[.45, .1, 0])
    rbt = el.loadYumi(showrbt=False)

    transmat4 = rm.homomat_from_posrot((.45, .1, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # transmat4 = rm.homomat_from_posrot((.4, -.1, bconfig.BENDER_H))

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    f = 'tri'
    fo = 'stick'

    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f}.pkl'), 'rb'))
    # goal_pseq = bu.gen_polygen(3, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]]) * .5
    # pickle.dump(goal_pseq, open(f'{config.ROOT}/bendplanner/goal/pseq/{f_name}.pkl', 'wb'))

    '''
    plan
    '''
    fit_pseq, _, _ = bu.decimate_pseq(goal_pseq, tor=.01, toggledebug=False)
    # fit_pseq, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=13, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=False)
    init_rot = bu.get_init_rot(fit_pseq)
    pickle.dump([goal_pseq, bendset], open(f'{config.ROOT}/bendplanner/planres/{fo}/{f}_bendseq.pkl', 'wb'))

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('stick_yumi')
    # grasp_list = grasp_list[200:300]

    brp.set_up(bendset, grasp_list, transmat4)
    # brp.pre_grasp_reasoning()
    # brp.run(f_name=f_name, fo=fo)

    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_bendset.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/yumi/{f}_pathseq.pkl', 'rb'))
    min_f_list, f_list = brp.check_force(bendresseq, armjntsseq_list, show_step=0)
    armjntsseq_list = np.asarray(armjntsseq_list)[np.argsort(min_f_list)[::-1]]

    # brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])
    # for i, armjnts in enumerate(pathseq_list[0][1][1][1:-2]):
    #     if i == 0:
    #         continue
    #     eepos_s, _ = mp.get_ee(pathseq_list[0][1][1][i])
    #     eepos_e, _ = mp.get_ee(pathseq_list[0][1][1][i - 1])
    #     # gm.gen_arrow(np.asarray(eepos_s), np.asarray(eepos_e), thickness=.002, rgba=(0, 1, 0, 1)).attach_to(base)
    #     # gm.gen_sphere(np.asarray(eepos_s), radius=.002, rgba=(0, 1, 0, 1)).attach_to(base)
    #     if i % 2 == 0:
    #         mp.ah.show_armjnts(armjnts=armjnts, rgba=(0, 1, 0, .5))
    # base.run()

    # brp.show_bend(bendresseq[0], show_start=True, show_end=True)
    brp.show_bend(bendresseq[-1], show_start=True, show_end=True)
    # mp.ah.show_armjnts(armjnts=pathseq_list[0][1][1][0], rgba=(.7, .7, .7, .5))
    # mp.ah.show_armjnts(armjnts=pathseq_list[0][1][1][-1], rgba=(.7, .7, .7, .5))
    # mp.ah.show_armjnts(armjnts=pathseq_list[0][1][1][0])
    mp.ah.show_armjnts(armjnts=pathseq_list[0][1][1][-1])
    brp.set_bs_stick_sec(180)
    # brp.show_bendresseq(bendresseq)

    base.run()
