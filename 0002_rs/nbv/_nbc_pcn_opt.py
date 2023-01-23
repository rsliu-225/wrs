import copy

import visualization.panda.world as wd
import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import json
import os
import open3d as o3d

import utils.pcd_utils as pcdu
import localenv.envloader as el
import motionplanner.pcn_nbc_solver as nbc_solver
import nbv_utils as nu
import basis.o3dhelper as o3dh
import basis.robot_math as rm
import modeling.geometric_model as gm

if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)

    path = 'D:/nbv_mesh/'
    cat = 'bspl_4'
    fo = 'res_75'
    coverage_pcn = []
    coverage_org = []

    coverage_tor = .001
    toggledebug = True
    f = '0010.ply'

    cov_list = []
    cov_opt_list = []
    relmat4 = rm.homomat_from_posrot([.02, 0, 0], np.eye(3))

    for f in os.listdir(os.path.join(path, cat, 'mesh'))[0:]:
        print(f'-----------------{f}-----------------')
        res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))
        o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
        objcm = o3dh.o3dmesh2cm(o3dmesh)

        pcd_gt = np.asarray(res_pcn['gt'])
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
        o3dpcd_i = nu.gen_partial_o3dpcd(o3dmesh, rot_center=(0, 0, 0), toggledebug=True)
        o3d.visualization.draw_geometries([o3dpcd_i, coord])
        pcd_i = np.asarray(o3dpcd_i.points)
        o3d.io.write_point_cloud(f'./tmp.pcd', o3dpcd_i)

        # o3dpcd_i = o3d.io.read_point_cloud(f'./tmp.pcd')
        # pcd_i = np.asarray(o3dpcd_i.points)
        # pcd_i = pcdu.trans_pcd(pcd_i, relmat4)

        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        print('coverage(init):', round(coverage, 3))

        o3dpcd = o3dh.nparray2o3dpcd(pcd_i)
        seedjntagls = rbt.get_jnt_values('arm')
        rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot)

        nbv_opt = nbc_solver.PCNNBCOptimizer(rbt, releemat4=relmat4, toggledebug=False)
        jnts = nbv_opt.solve(seedjntagls, pcd_i, cam_pos, method='COBYLA')
        print(jnts)
        # jnts = np.asarray([0.11868811, -1.07578064, -0.16183396, 0.27275529, 0.38601172, -0.42238339, -0.17258872])
        rbt.fk('arm', jnts)
        rbt.gen_meshmodel().attach_to(base)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)

        transmat4 = np.linalg.inv(init_eemat4).dot(eemat4)

        # o3dpcd_tmp = nu.gen_partial_o3dpcd(o3dmesh, rot=transmat4[:3, :3], rot_center=(0, 0, 0),
        #                                    trans=transmat4[:3, 3] + np.asarray([.02, 0, 0]), toggledebug=True)
        # o3dpcd += o3dpcd_tmp
        # o3dpcd.paint_uniform_color(nu.COLOR[0])
        # o3d.visualization.draw_geometries([o3dpcd], mesh_show_back_face=True)
        #
        # coverage = pcdu.cal_coverage(np.asarray(o3dpcd.points), pcd_gt, tor=coverage_tor)
        # print('coverage(opt):', round(coverage, 3))
        # cov_opt_list.append(round(coverage, 3))
        #
        # objcm_init = copy.deepcopy(objcm)
        # objcm_init.set_homomat(init_eemat4)
        # objcm_init.attach_to(base)
        # objcm.set_homomat(eemat4)
        # objcm.attach_to(base)
        # pcdu.show_pcd(pcdu.trans_pcd(pcd_i, eemat4), rgba=(1, 0, 0, 1))
        # pcdu.show_pcd(pcdu.trans_pcd(pcd_i, init_eemat4), rgba=(1, 1, 0, .4))
        # # pcdu.show_cam(rm.homomat_from_posrot(init_eerot.dot(cam_pos) + init_eepos,
        # #                                      rot=rm.rotmat_from_axangle((0, 0, 1), np.pi / 2)))
        #
        # # pcdu.show_pcd(pcdu.trans_pcd(pcd_o, eemat4), rgba=(1, 1, 0, .5))
        # # pcdu.show_pcd(pcd_i, rgba=(1, 0, 0, 1))
        # base.run()

    print(cov_list)
    print(cov_opt_list)
    # gm.gen_frame().attach_to(base)
    # base.run()
