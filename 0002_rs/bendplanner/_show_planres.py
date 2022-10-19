import os

import numpy as np

import basis.robot_math as rm
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import motionplanner.motion_planner as m_planner
import matplotlib.pyplot as plt
import utils.phoxi as phoxi
import utils.vision_utils as vu
import modeling.geometric_model as gm
import bendplanner.BendOpt as b_opt

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])

if __name__ == '__main__':
    import pickle
    import localenv.envloader as el
    import utils.pcd_utils as pcdu

    # f_name = 'randomc'
    # f_name = 'chair'
    f_name = 'tri'
    fo = 'stick'
    rbt_name = 'yumi'

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=False)
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.45, .15, bconfig.BENDER_H + .035), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # transmat4 = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_transmat4.pkl', 'rb'))
    transmat4 = rm.homomat_from_posrot((.45, .1, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    # gm.gen_frame(transmat4[:3, 3], transmat4[:3, :3]).attach_to(base)

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f_name}.pkl'), 'rb'))

    '''
    show result
    '''
    goal_pseq, bendset = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_bendset.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_pathseq.pkl', 'rb'))
    print('Num. of solution', len(pathseq_list))
    print('Num. of solution', len(armjntsseq_list))
    for bendres in bendresseq:
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres
    print(seqs)

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    brp.set_up(bendset, [], transmat4)
    brp.set_bs_stick_sec(180)

    # _, _, _, _, _, pseq, _ = bendresseq[-1]
    # pseq = np.asarray(pseq)
    # pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
    # ax = plt.axes(projection='3d')
    # center = np.mean(pseq, axis=0)
    # ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
    # ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
    # ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
    # bu.plot_pseq(ax, pseq, c='k')
    # bu.scatter_pseq(ax, pseq[1:-2], c='r')
    # bu.scatter_pseq(ax, pseq[:1], c='g')
    # plt.show()

    min_f_list, f_list = brp.check_force(bendresseq, armjntsseq_list, show_step=0)
    armjntsseq_list = np.asarray(armjntsseq_list)[np.argsort(min_f_list)]

    # for i, f in enumerate(min_f_list):
    #     mp.ah.show_armjnts(armjnts=pathseq_list[i][1][0][-1],
    #                        rgba=(0, (f / max(min_f_list)), 1 - f / max(min_f_list), .5))
    # brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])

    show_step = 0
    f_list_step = f_list[:, show_step]
    brp.show_bend(bendresseq[show_step])
    print(min_f_list)
    print([np.argsort(min_f_list)[::-1]])
    mp.ah.show_armjnts(armjnts=armjntsseq_list[0][1][show_step], rgba=None)
    mp.ah.show_armjnts(armjnts=armjntsseq_list[-1][1][show_step], rgba=(0, 0, 1, .5))
    # for i, f in enumerate(f_list_step):
    #     scale = max(f_list_step) - min(f_list_step)
    #     mp.ah.show_armjnts(armjnts=armjntsseq_list[i][1][show_step],
    #                        rgba=(0, (f - min(f_list_step)) / scale, 1 - (f - min(f_list_step)) / scale, .5))
    # brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])
    # brp.show_bendresseq_withrbt(bendresseq, armjntsseq_list[0][1])
    # brp.show_bendresseq(bendresseq)
    base.run()
