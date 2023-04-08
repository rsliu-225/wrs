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
import utils.vision_utils as vu
import bendplanner.bender_config as bconfig
import config
import motionplanner.motion_planner as m_planner
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

AFFINE_MAT4 = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                          [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                          [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                          [0.0, 0.0, 0.0, 1.0]])


# AFFINE_MAT4 = np.asarray([[6.01298773e-02, -9.78207659e-01, 1.98731412e-01, 5.16091421e+02],
#                          [-9.79046435e-01, -1.89910797e-02, 2.02749641e-01, -1.70789291e+02],
#                          [-1.94557128e-01, -2.06758591e-01, -9.58852652e-01, 1.75997120e+03],
#                          [0, 0, 0, 1]])


def show_wire(fo, f, center, z_range=(.15, .18), rgba=(1, 0, 0, 1)):
    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f), 'rb'))
    pcd = rm.homomat_transform_points(AFFINE_MAT4, np.asarray(pcd))
    pcd = np.asarray(pcd)
    textureimg = vu.enhance_grayimg(textureimg)
    cv2.imshow('', textureimg)
    cv2.waitKey(0)
    pcd_crop = pcdu.crop_pcd(pcd, x_range=(center[0] - .12, center[0] + .2),
                             y_range=(center[1] - .1, center[1] + .2), z_range=z_range)

    # pcdu.show_pcd(pcd, rgba=(1, 1, 1, .4))
    # pcdu.show_pcd(pcd_crop, rgba=rgba)

    # lines = pcdu.extract_lines_from_pcd(textureimg, pcd_crop, z_range=None, line_thresh=.0025,
    #                                     line_size_thresh=300, toggledebug=False)
    # dist_list = []
    # for _, line_pts in lines:
    #     kdt, _ = pcdu.get_kdt(line_pts)
    #     dist_list.append(pcdu.get_min_dist(center, kdt))
    #
    # sort_inx = np.argsort(np.asarray(dist_list))
    # print(sort_inx, dist_list)
    # angle = np.degrees(rm.angle_between_vectors(lines[sort_inx[0]][0], lines[sort_inx[1]][0]))
    # print(angle)
    # pcdu.show_pcd(lines[sort_inx[0]][1], rgba=rgba)
    # pcdu.show_pcd(lines[sort_inx[1]][1], rgba=rgba)
    return np.asarray(pcd_crop)


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

def filter_pcd(pcd_gt, pcd, max_dist=.02):
    kdt, _ = pcdu.get_kdt(pcd_gt)
    pcd_res = []
    for p in pcd:
        dist = pcdu.get_min_dist(p, kdt)
        if dist < max_dist:
            pcd_res.append(p)
    return np.asarray(pcd_res)


if __name__ == '__main__':
    f_name = 'randomc'
    fo = 'stick'
    rbt_name = 'yumi'

    if rbt_name == 'yumi':
        base, env = el.loadEnv_yumi(camp=[2, -1, 1.5], lookatpos=[.5, 0, 0])
        rbt = el.loadYumi(showrbt=True)
    else:
        base, env = el.loadEnv_wrs()
        rbt = el.loadUr3e(showrbt=True)

    transmat4 = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_transmat4.pkl', 'rb'))
    # transmat4 = rm.homomat_from_posrot((.8, .2, bconfig.BENDER_H + .8), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    goal_pseq, bendset = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_bendset.pkl', 'rb'))
    seqs, _, bendresseq = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_bendresseq.pkl', 'rb'))
    armjntsseq_list = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_armjntsseq.pkl', 'rb'))
    pathseq_list = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f_name}_pathseq.pkl', 'rb'))
    bendset = [bendset[i] for i in seqs]

    transmat4 = rm.homomat_from_posrot(transmat4[:3, 3] + np.asarray([0, 0, .008]), transmat4[:3, :3])
    # transmat4 = rm.homomat_from_posrot(transmat4[:3, 3] + np.asarray([0, .01, -.01]), transmat4[:3, :3])

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    # print(armjntsseq_list[0])
    # mp.ah.show_armjnts(armjnts=armjntsseq_list[1][1][1])

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
    brp.set_up(bendset, [], transmat4)

    i = 2
    print('goal:', np.degrees(bendset[i][0]))
    bendresseq_new = []
    for j, v in enumerate(bendresseq):
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = v
        pseq_init[-1] = pseq_init[-1] + rotseq_init[-1][:, 1] * .01
        pseq_end[-1] = pseq_end[-1] + rotseq_end[-1][:, 1] * .01
        bendresseq_new.append([init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end])
    bendresseq = bendresseq_new

    obj_init, obj_end = brp.show_bend(bendresseq[i], show_end=True, show_start=True,
                                      start_rgba=(.7, .7, .7, .7), end_rgba=(1, 1, 1, .5))
    obj_res = brp.show_bend_crop(bendresseq[i], bendset[i][-1])
    obj_res.set_rgba((1, 0, 0, 1))
    obj_res.attach_to(base)
    # base.run()
    pcd_init = np.asarray(obj_init.sample_surface(radius=.001)[0])
    pcd_gt = np.asarray(obj_res.sample_surface(radius=.001)[0])
    # gm.gen_pointcloud(pcd_init, [[1, 1, 0, 1]] * len(pcd_init)).attach_to(base)
    # base.run()

    # pseq_init, pseq_end = brp.show_bend(bendresseq[1])
    # for p in pseq_end:
    #     gm.gen_sphere(p, radius=0.002, rgba=(1, 1, 0, 1)).attach_to(base)
    if rbt_name == 'yumi':
        z_range = (.1, .2)
    else:
        z_range = (.91, .92)
    pcd_release = show_wire(f'{fo}/{f_name}', f'{str(i)}_release.pkl', center=transmat4[:3, 3], z_range=z_range,
                            rgba=(1, 1, 0, 1))
    pcd_goal = show_wire(f'{fo}/{f_name}', f'{str(i)}_goal.pkl', center=transmat4[:3, 3], z_range=z_range,
                         rgba=(0, 1, 0, 1))
    # base.run()
    pcd_release_filtered = filter_pcd(pcd_gt, pcd_release, max_dist=.01)
    pcd_goal_filtered = filter_pcd(pcd_gt, pcd_goal, max_dist=.01)

    pcd_gt = pcdu.get_objpcd_partial_bycampos(obj_res, cam_pos=[.4, 0, -1], smp_num=5000)
    pcdu.show_pcd(pcd_gt)
    pcdu.show_pcd(pcd_release_filtered, rgba=(1, 1, 0, 1))
    pcdu.show_pcd(pcd_goal_filtered, rgba=(0, 1, 0, 1))
    # base.run()

    _, _, trans_release = \
        o3dh.registration_ptpt(pcd_gt, np.asarray(pcd_release_filtered), downsampling_voxelsize=.002,
                               toggledebug=False)
    _, _, trans_goal = \
        o3dh.registration_ptpt(pcd_gt, np.asarray(pcd_goal_filtered), downsampling_voxelsize=.002,
                               toggledebug=False)

    # _, _, trans_rel = \
    #     o3dh.registration_ptpt(np.asarray(pcd_release_filtered), np.asarray(pcd_goal_filtered),
    #                            downsampling_voxelsize=.002, toggledebug=False)
    gm.gen_frame(pos=transmat4[:3, 3], rotmat=trans_goal[:3, :3], thickness=.002).attach_to(base)
    gm.gen_frame(pos=transmat4[:3, 3], rotmat=trans_release[:3, :3], thickness=.002).attach_to(base)
    sb_angle = rm.angle_between_vectors(trans_goal[:3, 0], trans_release[:3, 0])
    print(np.degrees(sb_angle))
    # sb_angle = rm.rotmat_to_euler(trans_rel)

    pcdu.show_pcd(pcd_release_filtered, rgba=(1, 1, 0, 1))
    pcdu.show_pcd(pcd_goal_filtered, rgba=(0, 1, 0, 1))
    # pcdu.show_pcd(pcd_res, rgba=(1, 1, 0, 1))
    base.run()
