import pickle

import matplotlib.pyplot as plt
import numpy as np

import basis.robot_math as rm
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import localenv.envloader as el
import motionplanner.motion_planner as m_planner
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe

if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16

    gripper = rtqhe.RobotiqHE()
    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.7, -.2, .78 + bconfig.BENDER_H), np.eye(3))

    base, env = el.loadEnv_yumi(camp=[.4, .075, 1.7], lookatpos=[.4, .075, 0])
    rbt = el.loadYumi(showrbt=False)

    bs = b_sim.BendSim(show=True)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    result, _, _, _, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/bendresseq/180/4_1.pkl', 'rb'))
    bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
    _,seqs = result[-1]
    print(seqs)
    bendseq = [bendset[i] for i in seqs]
    for b in bendseq:
        print(b)
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=False, prune=True, toggledebug=False)
    init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[-1]

    ax = plt.axes(projection='3d')
    bu.plot_pseq(ax, pseq_end, c='k')
    bu.scatter_pseq(ax, pseq_end[:-1], c='r')
    plt.show()

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('stick_yumi')
    grasp_list = grasp_list[100:]

    for x in np.linspace(.4, .6, 5):
        for y in np.linspace(.075, .3, 5):
            print(x, y)
            transmat4 = rm.homomat_from_posrot((x, y, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
            brp.set_up(bendset, grasp_list, transmat4)
            fail_index, armjntsseq_list = brp.check_ik(bendresseq, grasp_l=0)
            if fail_index != -1:
                continue
            else:
                brp.show_bendresseq_withrbt(bendresseq, armjntsseq_list[0][1])
                base.run()

    transmat4 = rm.homomat_from_posrot((.4, .075, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    brp.set_up(bendset, grasp_list, transmat4)
    # brp.pre_grasp_reasoning()
    # base.run()

    brp.show_bendresseq(bendresseq)
    base.run()
    brp.run(f_name='6_1')
