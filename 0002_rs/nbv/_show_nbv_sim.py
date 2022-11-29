import copy
import json
import math
import os
import pickle
import random
import h5py
import numpy as np
import open3d as o3d

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import datagenerator.data_utils as du
import pcn.inference as pcn
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import visualization.panda.world as wd

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def show_pcn_res_pytorch(result_path, test_path):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    for i in range(10, len(test_f['complete_pcds'])):
        o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][i]))
        o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][i]))
        o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][i]))
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
        o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])


def read_pcn_res_pytorch(result_path, test_path, id, toggledebug=False):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    if toggledebug:
        o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][id]))
        o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][id]))
        o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][id]))
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
        o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])
    return np.asarray(test_f['complete_pcds'][id]), \
           np.asarray(test_f['incomplete_pcds'][id]), \
           np.asarray(res_f['results'][id])


if __name__ == '__main__':
    import modeling.geometric_model as gm
    import datagenerator.data_utils as du

    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    gm.gen_cone(epos=[0, 0, .1], radius=.05, sections=60).attach_to(base)

    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))

    path = 'E:/liu/nbv_mesh/'
    cat = 'bspl'
    fo = 'res_90'
    coverage_pcn = []
    coverage_org = []

    coverage_tor = .001
    toggledebug = True
    f = '0000.ply'

    res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))

    pcd_i = np.asarray(res_pcn['0']['input'])
    pcd_add = np.asarray(res_pcn['0']['add'])
    pcd_o = np.asarray(res_pcn['0']['pcn_output'])
    pcd_res = np.asarray(res_pcn['final'])
    pcd_gt = np.asarray(res_pcn['gt'])
    pcdu.show_pcd(pcd_add, rgba=(0, .7, 0, 1))

    pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn_kpts(pcd_i, pcd_o, theta=None, toggledebug=True)
    # pts, nrmls, confs = pcdu.cal_conf(pcd_i, voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=None)
    # pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)

    rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
    rot = np.linalg.inv(rot)
    pcd_i_new = pcdu.trans_pcd(pcd_i, rm.homomat_from_posrot((0, 0, 0), rot))
    pcdu.show_pcd(pcd_i_new, rgba=(.7, .7, .7, .5))
    gm.gen_arrow(np.dot(rot, pts_nbv[0]),
                 np.dot(rot, pts_nbv[0]) + np.dot(rot, nrmls_nbv[0]) * .04, thickness=.002).attach_to(base)

    gm.gen_sphere(pts_nbv[0], radius=.01, rgba=[1, 1, 1, .2]).attach_to(base)
    base.run()

    width = .008
    thickness = .002
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    # gm.gen_pointcloud(res).attach_to(base)
    gm.gen_pointcloud(pcd_i).attach_to(base)
    kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_res, n_components=16, show=True)
    cov = pcdu.cal_coverage(pcd_i, pcd_gt, voxel_size=.001, tor=coverage_tor, toggledebug=True)
    print(cov)
    # objcm = du.gen_swap(kpts, kpts_rotseq, cross_sec)
    # objcm.attach_to(base)

    base.run()
