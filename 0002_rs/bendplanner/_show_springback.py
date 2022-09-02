import pickle
import os
import config
import numpy as np
import cv2
import basis.robot_math as rm
import utils.pcd_utils as pcdu
import localenv.envloader as el
import basis.o3dhelper as o3dh
import modeling.geometric_model as gm
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import motionplanner.motion_planner as m_planner

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def show_wire(fo, f, center, z_range=(.15, .18), rgba=(1, 0, 0, 1)):
    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f), 'rb'))
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))

    pcd_crop = pcdu.crop_pcd(pcd, x_range=(center[0] - .12, center[0] + .2),
                             y_range=(center[1] - .1, center[1] + .2), z_range=z_range)
    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
    pcdu.show_pcd(pcd_crop, rgba=rgba)

    lines = pcdu.extract_lines_from_pcd(textureimg, pcd_crop, z_range=None, line_thresh=.0025,
                                        line_size_thresh=300, toggledebug=False)
    dist_list = []
    for _, line_pts in lines:
        kdt, _ = pcdu.get_kdt(line_pts)
        dist_list.append(pcdu.get_min_dist(center, kdt))

    sort_inx = np.argsort(np.asarray(dist_list))
    print(sort_inx, dist_list)
    angle = np.degrees(rm.angle_between_vectors(lines[sort_inx[0]][0], lines[sort_inx[1]][0]))
    print(angle)
    # pcdu.show_pcd(lines[sort_inx[0]][1], rgba=rgba)
    # pcdu.show_pcd(lines[sort_inx[1]][1], rgba=rgba)


#
# def show_wire(fo, f, pseq_sim, z_range=(.15, .18), rgba=(1, 0, 0, 1)):
#     textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f), 'rb'))
#     pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
#
#     pcd_crop = pcdu.crop_pcd(pcd, x_range=(center[0] - .2, center[0] + .2),
#                              y_range=(center[1] - .2, center[1] + .2), z_range=z_range)
#     pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
#     pcdu.show_pcd(pcd_crop, rgba=(1, 1, 0, .5))
#     # pcdu.show_pcd(pcd_crop, rgba=rgba)
#
#     lines = pcdu.extract_lines_from_pcd(textureimg, pcd_crop, z_range=None, line_thresh=.0025,
#                                         line_size_thresh=300, toggledebug=False)
#     dist_list = []
#     for _, line_pts in lines:
#         kdt, _ = pcdu.get_kdt(line_pts)
#         dist_list.append(pcdu.get_min_dist(center, kdt))
#
#     sort_inx = np.argsort(np.asarray(dist_list))
#     print(sort_inx, dist_list)
#     angle = np.degrees(rm.angle_between_vectors(lines[sort_inx[0]][0], lines[sort_inx[1]][0]))
#     print(angle)
#     pcdu.show_pcd(lines[sort_inx[0]][1], rgba=rgba)
#     pcdu.show_pcd(lines[sort_inx[1]][1], rgba=rgba)


if __name__ == '__main__':
    f_name = 'penta'
    fo = 'stick'

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=True)

    transmat4 = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_transmat4.pkl', 'rb'))
    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendseq.pkl', 'rb'))
    seqs, _, bendresseq = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_bendresseq.pkl', 'rb'))
    armjntsseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{f_name}_pathseq.pkl', 'rb'))

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
    brp.set_up(bendset, [], transmat4)

    # pseq_init, pseq_end = brp.show_bend(bendresseq[1])
    # for p in pseq_end:
    #     gm.gen_sphere(p, radius=0.002, rgba=(1, 1, 0, 1)).attach_to(base)

    # show_wire(f'{fo}/{f_name}', '0_release.pkl', center=transmat4[:3, 3], z_range=(.15, .18), rgba=(1, 1, 0, 1))
    # show_wire(f'{fo}/{f_name}', '0_goal.pkl', center=transmat4[:3, 3], z_range=(.15, .18), rgba=(0, 1, 0, 1))
    # show_wire(f'{fo}/{f_name}', '0_refine.pkl', center=transmat4[:3, 3], z_range=(.15, .18), rgba=(0, 1, 0, 1))
    show_wire(f'{fo}/{f_name}', '1_release.pkl', center=transmat4[:3, 3], z_range=(.15, .18), rgba=(1, 1, 0, 1))
    show_wire(f'{fo}/{f_name}', '1_goal.pkl', center=transmat4[:3, 3], z_range=(.15, .18), rgba=(0, 1, 0, 1))

    base.run()
