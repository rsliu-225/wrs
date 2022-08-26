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
    import localenv.envloader as el

    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.7, -.2, .78 + bconfig.BENDER_H), np.eye(3))

    base, env = el.loadEnv_yumi(camp=[.4, 0, 1.7], lookatpos=[.4, 0, 0])
    rbt = el.loadYumi(showrbt=False)
    transmat4 = rm.homomat_from_posrot((.45, .1, bconfig.BENDER_H + .03), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # transmat4 = rm.homomat_from_posrot((.4, -.1, bconfig.BENDER_H))

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    # f_name = 'randomc'
    # f_name = 'chair'
    f_name = 'tri'
    fo = 'stick'

    # goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f_name}.pkl'), 'rb'))
    goal_pseq = bu.gen_polygen(3, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]]) * .5
    pickle.dump(goal_pseq, open(f'{config.ROOT}/bendplanner/goal/pseq/{f_name}.pkl', 'wb'))

    '''
    plan
    '''
    fit_pseq, _ = bu.decimate_pseq(goal_pseq, tor=.01, toggledebug=False)
    # fit_pseq, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=13, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=True)
    init_rot = bu.get_init_rot(fit_pseq)
    pickle.dump([goal_pseq, bendset],
                open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendseq.pkl', 'wb'))

    for b in bendset:
        print(b)

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('stick_yumi')
    # grasp_list = grasp_list[:100]

    brp.set_up(bendset, grasp_list, transmat4)
    brp.run(f_name=f_name, folder_name=fo)
    brp.pre_grasp_reasoning()

    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendseq.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_pathseq.pkl', 'rb'))

    brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])
    pathseq_list, min_f_list = brp.check_force(bendresseq, pathseq_list)
    print(min_f_list)
    base.run()
