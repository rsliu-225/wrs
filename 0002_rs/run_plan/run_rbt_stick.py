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


def get_transmat4_marker():
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    textureimg, depthimg, pcd = phxi.getalldata()
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))

    textureimg = vu.enhance_grayimg(textureimg)
    centermat4 = vu.get_axis_aruco(textureimg, pcd)

    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
    gm.gen_sphere(centermat4[:3, 3], radius=.005).attach_to(base)
    gm.gen_frame(centermat4[:3, 3], centermat4[:3, :3]).attach_to(base)
    residual = (3, -3, 0)

    center_pillar_pos = centermat4[:3, 3] + \
                        centermat4[:3, 0] * (75 - 45.43 + residual[0]) / 1000 + \
                        centermat4[:3, 1] * (-54 - 50 + residual[1]) / 1000 + \
                        centermat4[:3, 2] * (52.75 + residual[2]) / 1000

    print('Center pillar pos:', center_pillar_pos)

    return rm.homomat_from_posrot(center_pillar_pos, centermat4[:3, :3])


if __name__ == '__main__':
    import pickle
    import localenv.envloader as el
    import utils.pcd_utils as pcdu

    f_name = 'randomc'
    # f_name = 'chair'
    # f_name = 'penta'
    fo = 'stick'

    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=True)
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.45, 0, bconfig.BENDER_H + .035), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    transmat4 = get_transmat4_marker()
    pickle.dump(transmat4, open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_transmat4.pkl', 'wb'))
    transmat4 = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_transmat4.pkl', 'rb'))

    gm.gen_frame(transmat4[:3, 3], transmat4[:3, :3]).attach_to(base)

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f_name}.pkl'), 'rb'))
    # goal_pseq = bu.gen_polygen(5, .05)
    # goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]])[::-1] * .5
    # goal_pseq = np.asarray([[0, 0, 0], [0, 0, .1], [.1, 0, .1], [.1, .1, .1], [0, .1, .1], [0, .1, 0]]) * .5
    # goal_pseq = np.asarray([[0, .1, .1], [0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, 0, .1], [.1, .1, .1]])[::1] * .5
    # pickle.dump(goal_pseq, open(f'{config.ROOT}/bendplanner/goal/pseq/{f_name}.pkl', 'wb'))

    plan = True
    opt = True

    '''
    plan
    '''
    if plan:
        # fit_pseq, _ = bu.decimate_pseq(goal_pseq, tor=.01, toggledebug=False)
        fit_pseq, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=13, toggledebug=False)
        bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=True)[::-1]

        init_rot = bu.get_init_rot(fit_pseq)
        pickle.dump([goal_pseq, bendset], open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendseq.pkl', 'wb'))
        for b in bendset:
            print(b)
        init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
        init_rotseq = [np.eye(3), np.eye(3)]

        if opt:
            opt = b_opt.BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=1,  obj_type='avg')
            bendset, _, _ = opt.solve(tor=None, cnt=13)
            for b in bendset:
                print(bendset)

        brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

        grasp_list = mp.load_all_grasp('stick')
        # grasp_list = grasp_list[140:190]
        transmat4 = rm.homomat_from_posrot(transmat4[:3, 3] + np.asarray([0, 0, .008]), transmat4[:3, :3])
        brp.set_up(bendset, grasp_list, transmat4)
        brp.run(f_name=f_name, fo=fo)

    '''
    show result
    '''
    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendseq.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_pathseq.pkl', 'rb'))
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

    pathseq_list, min_f_list, f_list = brp.check_force(bendresseq, pathseq_list)

    # for i, f in enumerate(min_f_list):
    #     mp.ah.show_armjnts(armjnts=pathseq_list[i][1][0][-1],
    #                        rgba=(0, (f / max(min_f_list)), 1 - f / max(min_f_list), .5))
    # brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])

    show_step = 2
    f_list_step = f_list[:, show_step]
    brp.show_bend(bendresseq[show_step])
    print(f_list_step)
    for i, f in enumerate(f_list_step):
        scale = max(f_list_step) - min(f_list_step)
        mp.ah.show_armjnts(armjnts=pathseq_list[i][1][show_step][-1],
                           rgba=(0, (f - min(f_list_step)) / scale, 1 - (f - min(f_list_step)) / scale, .5))
    brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])
    base.run()
