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
import pickle
import localenv.envloader as el
import utils.pcd_utils as pcdu

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


# affine_mat = np.asarray([[6.01298773e-02, -9.78207659e-01, 1.98731412e-01, 5.16091421e+02]
#                          [-9.79046435e-01, -1.89910797e-02, 2.02749641e-01, -1.70789291e+02]
#                          [-1.94557128e-01, -2.06758591e-01, -9.58852652e-01, 1.75997120e+03]
#                          [0, 0, 0, 1]])


def get_transmat4_marker():
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    textureimg, depthimg, pcd = phxi.getalldata()
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))

    textureimg = vu.enhance_grayimg(textureimg)
    centermat4 = vu.get_axis_aruco(textureimg, pcd)

    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
    gm.gen_sphere(centermat4[:3, 3], radius=.005).attach_to(base)
    gm.gen_frame(centermat4[:3, 3], centermat4[:3, :3]).attach_to(base)
    residual = (0, -3, 0)

    center_pillar_pos = centermat4[:3, 3] + \
                        centermat4[:3, 0] * (75 - 45.43 + residual[0]) / 1000 + \
                        centermat4[:3, 1] * (-54 - 50 + residual[1]) / 1000 + \
                        centermat4[:3, 2] * (52.75 + residual[2]) / 1000

    print('Center pillar pos:', center_pillar_pos)

    return rm.homomat_from_posrot(center_pillar_pos, centermat4[:3, :3])


def cal_pseq_lenght(pseq):
    l = 0
    for i in range(1, len(pseq)):
        l += np.linalg.norm(pseq[i] - pseq[i - 1])
    return l


if __name__ == '__main__':
    # f_name = 'randomc'
    # f_name = 'chair'
    f_name = 'helix'
    # f_name = 'sprial'
    # f_name = 'penta'

    fo = 'stick'
    rbt_name = 'yumi'

    plan = True
    opt = False
    calibrate = True
    refine = True

    if rbt_name == 'yumi':
        base, env = el.loadEnv_yumi()
        rbt = el.loadYumi(showrbt=False)
        if calibrate:
            transmat4 = get_transmat4_marker()
            pickle.dump(transmat4,
                        open(f'{config.ROOT}/bendplanner/planres_rev/stick/{rbt_name}/{f_name}_transmat4.pkl', 'wb'))
        transmat4 = \
            pickle.load(open(f'{config.ROOT}/bendplanner/planres_rev/stick/{rbt_name}/{f_name}_transmat4.pkl', 'rb'))
        transmat4 = rm.homomat_from_posrot(transmat4[:3, 3] + np.asarray([0, 0, .008]),
                                           transmat4[:3, :3])
        grasp_f_name = 'stick_yumi'
        gm.gen_frame(transmat4[:3, 3], transmat4[:3, :3]).attach_to(base)
        # base.run()
    else:
        base, env = el.loadEnv_wrs()
        rbt = el.loadUr3e()
        transmat4 = rm.homomat_from_posrot((.8, .2, bconfig.BENDER_H + .8),
                                           rm.rotmat_from_axangle((0, 0, 1), np.pi))
        grasp_f_name = 'stick'

    bs = b_sim.BendSim(show=True, cm_type=fo)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f_name}.pkl'), 'rb'))
    grasp_list = mp.load_all_grasp(grasp_f_name)

    '''
    plan
    '''
    if plan:
        fit_pseq, _, _ = bu.decimate_pseq(goal_pseq, tor=.002, toggledebug=False)
        # fit_pseq, _, _ = bu.decimate_pseq_by_cnt(goal_pseq, cnt=11, toggledebug=False)
        bendset = bu.pseq2bendset(fit_pseq, init_l=.1, toggledebug=True)
        if fo == 'plate':
            for i, b in enumerate(bendset):
                if abs(b[2]) >= np.pi:
                    bendset[i][2] = 0
                    bendset[i][0] = -bendset[i][0]

        print('Num. of bend candidate', len(bendset))
        init_rot = bu.get_init_rot(fit_pseq)
        init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
        init_rotseq = [np.eye(3), np.eye(3)]

        '''
        gen bending action set
        '''
        # brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
        # is_success, bendresseq, _ = brp._bs.gen_by_bendseq(bendset, cc=True, prune=True, toggledebug=False)
        # bs.show_bendresseq(bendresseq, is_success)
        # base.run()

        if opt:
            opt = b_opt.BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=1, obj_type='avg')
            bendset, _, _ = opt.solve(tor=None, cnt=11)

        pickle.dump([goal_pseq, bendset],
                    open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_bendset.pkl', 'wb'))
        _, bendset = pickle.load(
            open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_bendset.pkl', 'rb'))

        print(len(bendset))

        # for i in range(len(bendset)):
        #     bendset[i][-1] = bendset[i][-1] -.05
        brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
        # grasp_list = grasp_list[140:190]
        brp.set_up(bendset, grasp_list, transmat4)
        # brp.run(f_name=f_name, fo=f'{fo}/{rbt_name}')
        # brp.run_premutation(f_name=f_name, fo=f'{fo}/{rbt_name}')
        brp.run(f_name=f_name, fo=f'{fo}/{rbt_name}')
        # base.run()

    '''
    show result
    '''
    goal_pseq, bendset = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_bendset.pkl', 'rb'))
    seq, _, bendresseq = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_bendresseq.pkl', 'rb'))
    path_armjntsseq_list = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_pathseq.pkl', 'rb'))
    print('Num. of solution', len(pathseq_list), '/', len(path_armjntsseq_list))
    for bendres in bendresseq:
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres
    print(seq)
    bendseq = [bendset[i] for i in seq]
    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
    brp.set_up(bendset, grasp_list, transmat4)
    if refine:
        brp.fine_tune(seq, f_name=f_name, fo=f'{fo}/{rbt_name}')
        pathseq_list = pickle.load(
            open(f'{config.ROOT}/bendplanner/planres_rev/{fo}/{rbt_name}/{f_name}_pathseq.pkl', 'rb'))
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

    path_armjntsseq_list = []
    for g, pathseq in pathseq_list:
        armjntsseq = []
        for p in pathseq:
            armjntsseq.append(p[-1])
        path_armjntsseq_list.append([g, armjntsseq])
    min_f_list, f_list = brp.check_force(bendresseq, path_armjntsseq_list)
    print(min_f_list)
    pathseq_list = np.asarray(pathseq_list)[np.argsort(min_f_list)[::-1]]
    # armjntsseq_list = np.asarray(armjntsseq_list)[np.argsort(min_f_list)[::-1]]

    '''
    show force
    '''
    # show_step = 2
    # f_list_step = f_list[:, show_step]
    # brp.show_bend(bendresseq[show_step])
    # for i, f in enumerate(f_list_step):
    #     scale = max(f_list_step) - min(f_list_step)
    #     mp.ah.show_armjnts(armjnts=pathseq_list[i][1][show_step][-1],
    #                        rgba=(0, (f - min(f_list_step)) / scale, 1 - (f - min(f_list_step)) / scale, .5))

    # show_step = 1
    # brp.show_bendres_withrbt(bendresseq[show_step], armjntsseq_list[0][1][show_step])
    # brp.show_bend_crop(bendresseq[show_step], bendseq[show_step][-1])
    # base.run()

    brp.show_motion_withrbt(bendresseq, pathseq_list[0][1])
    # brp.show_bendresseq_withrbt(bendresseq, armjntsseq_list[0][1])
    base.run()
