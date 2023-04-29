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
    f_name = 'sprial'
    # f_name = 'sprial'
    # f_name = 'penta'

    fo = 'stick'
    rbt_name = 'yumi'

    plan = False
    opt = False
    calibrate = True
    refine = True

    bs = b_sim.BendSim(show=False, cm_type=fo)
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f_name}.pkl'), 'rb'))[:220]

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
    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    _, _, _, _, _, pseq, _ = bendresseq[-1]
    pseq = np.asarray(pseq)
    pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
    ax = plt.axes(projection='3d')
    center = np.mean(pseq, axis=0)
    ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
    ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
    ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
    bu.plot_pseq(ax, pseq, c='k')
    bu.scatter_pseq(ax, pseq[1:-2], c='r')
    bu.scatter_pseq(ax, pseq[:1], c='g')
    plt.show()
