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

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=False)
    transmat4 = rm.homomat_from_posrot((.4, .3, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # transmat4 = rm.homomat_from_posrot((.4, -.1, bconfig.BENDER_H))

    bs = b_sim.BendSim(show=True)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    f_name = 'chair'
    folder_name = 'stick'

    # goal_pseq = pickle.load(open(f'../bendplanner/goal/pseq/{f_name}.pkl', 'rb'))
    # goal_pseq = bu.gen_polygen(5, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    goal_pseq = np.asarray([[.1, 0, .1], [0, 0, .1], [0, 0, 0],
                            [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
                            [.1, .1, .1]]) * .4
    pickle.dump(goal_pseq, open(f'{config.ROOT}/bendplanner/goal/pseq/{f_name}.pkl', 'wb'))

    plan = True

    '''
    plan
    '''
    if plan:
        fit_pseq, _ = bu.decimate_pseq(goal_pseq, tor=.002, toggledebug=False)
        bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=True)[::-1]
        init_rot = bu.get_init_rot(fit_pseq)
        pickle.dump([goal_pseq, bendset],
                    open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendseq.pkl', 'wb'))

        for b in bendset:
            print(b)

        init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
        init_rotseq = [np.eye(3), np.eye(3)]
        brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

        grasp_list = mp.load_all_grasp('stick')
        grasp_list = grasp_list[:200]

        brp.set_up(bendset, grasp_list, transmat4)
        brp.run(f_name=f_name, folder_name=folder_name)

    '''
    show result
    '''
    goal_pseq, bendset = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendseq.pkl', 'rb'))
    _, bendresseq = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendresseq.pkl', 'rb'))
    pathseq_list = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_pathseq.pkl', 'rb'))

    for bendres in bendresseq:
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    brp.set_up(bendset, [], transmat4)
    brp.set_bs_stick_sec(180)
    brp.show_motion_withrbt(bendresseq, pathseq_list[1][1])
    for p in pathseq_list:
        grasp, pathseq = p
    base.run()
