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
import motionplanner.nbc_pcn_opt_solver as nbc_solver
import nbv_utils as nu
import basis.o3dhelper as o3dh
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    campos = [0, 0, .4]

    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    path = 'D:/nbv_mesh/'
    if not os.path.exists(path):
        path = 'E:/liu/nbv_mesh/'

    cat = 'bspl_4'
    fo = 'res_75'

    cov_tor = .001
    goal = .95
    visible_threshold = np.radians(60)
    toggledebug = True

    cov_list = []
    cov_opt_list = []
    relmat4 = rm.homomat_from_posrot([.02, 0, 0], np.eye(3))
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    seedjntagls = rbt.get_jnt_values('arm')

    for f in os.listdir(os.path.join(path, cat, 'mesh'))[0:]:
        print(f'-----------------{f}-----------------')
        res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))
        o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
        o3dmesh.compute_vertex_normals()
        objcm = o3dh.o3dmesh2cm(o3dmesh)

        # o3dpcd_i = nu.gen_partial_o3dpcd(o3dmesh, toggledebug=False)
        pcd_gt = np.asarray(res_pcn['gt'])
        pcd_i = np.asarray(res_pcn['0']['input'])
        o3dpcd_i = o3dh.nparray2o3dpcd(pcd_i)
        o3dpcd_gt = o3dh.nparray2o3dpcd(pcd_gt)
        o3dpcd_i.paint_uniform_color(nu.COLOR[0])
        o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
        cnt = 0
        coverage = 0
        exp_dict = {}

        init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        print('coverage(init):', round(init_coverage, 3))

        exp_dict['gt'] = pcd_gt.tolist()
        exp_dict['init_coverage'] = init_coverage

        while coverage < .95:
            pcd_i = np.asarray(o3dpcd_i.points)
            o3d.visualization.draw_geometries([coord, o3dpcd_gt, o3dpcd_i], mesh_show_back_face=True)

            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
            init_eepos, init_eerot = rbt.get_gl_tcp()
            init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

            nbc_opt = nbc_solver.PCNNBCOptimizer(rbt, releemat4=relmat4, toggledebug=False)
            jnts, transmat4, _, time_cost = nbc_opt.solve(seedjntagls, pcd_i, campos, method='COBYLA')
            print(jnts)
            # jnts = np.asarray([0.29210199, -0.9822004, -0.22015057, 0.25273244, -0.05857504, -0.07970981, -0.05283248])

            rbt.fk('arm', jnts)
            rbt.gen_meshmodel().attach_to(base)
            # eepos, eerot = rbt.get_gl_tcp()
            # eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)
            # transmat4 = np.linalg.inv(init_eemat4).dot(eemat4)

            rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10)
            rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
            o3dpcd_nxt = nu.gen_partial_o3dpcd(o3dmesh, othermesh=[rbt_o3dmesh], toggledebug=toggledebug,
                                               trans=transmat4[:3, 3], rot=transmat4[:3, :3], rot_center=[0, 0, 0],
                                               fov=True, campos=campos)

            o3d.visualization.draw_geometries([coord, o3dpcd_gt, o3dpcd_nxt])
            o3dpcd_i += o3dpcd_nxt
            coverage = pcdu.cal_coverage(np.asarray(o3dpcd_i.points), pcd_gt, tor=cov_tor)
            print(f'coverage({str(cnt)}):', round(coverage, 3))
            cnt += 1

    # for f in os.listdir(os.path.join(path, cat, 'mesh'))[0:]:
    #     print(f'-----------------{f}-----------------')
    #     res_pcn = json.load(open(os.path.join(path, cat, fo, f'pcn_{f.split(".ply")[0]}.json'), 'rb'))
    #     o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
    #     objcm = o3dh.o3dmesh2cm(o3dmesh)
    #
    #     pcd_gt = np.asarray(res_pcn['gt'])
    #     o3dpcd_i = o3d.io.read_point_cloud(f'./tmp.pcd')
    #     pcd_i = pcdu.trans_pcd(np.asarray(o3dpcd_i.points), relmat4)
    #     pcd_gt = pcdu.trans_pcd(pcd_gt, relmat4)
    #
    #     coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    #     print('coverage(init):', round(coverage, 3))
    #
    #     o3dpcd = o3dh.nparray2o3dpcd(pcd_i)
    #     seedjntagls = rbt.get_jnt_values('arm')
    #     rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
    #     init_eepos, init_eerot = rbt.get_gl_tcp()
    #     init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot)
    #
    #     jnts = np.asarray([0.56723981, -0.96024077, -0.27171356, 0.46924616, 1.52918051, 0.07926629, 0.1277457])
    #     rbt.fk('arm', jnts)
    #     rbt.gen_meshmodel().attach_to(base)
    #     eepos, eerot = rbt.get_gl_tcp()
    #     eemat4 = rm.homomat_from_posrot(eepos, eerot)
    #
    #     transmat4 = np.linalg.inv(init_eemat4).dot(eemat4)
    #
    #     coverage = pcdu.cal_coverage(np.asarray(o3dpcd.points), pcd_gt, tor=cov_tor)
    #     print('coverage(opt):', round(coverage, 3))
    #     cov_opt_list.append(round(coverage, 3))
    #
    #     objcm_init = copy.deepcopy(objcm)
    #     objcm_init.set_homomat(init_eemat4.dot(relmat4))
    #     objcm_init.attach_to(base)
    #     objcm.set_homomat(eemat4.dot(relmat4))
    #     objcm.attach_to(base)
    #
    #     pcdu.show_pcd(pcdu.trans_pcd(pcd_i, eemat4), rgba=(1, 0, 0, 1))
    #     pcdu.show_pcd(pcdu.trans_pcd(pcd_gt, eemat4), rgba=(0, 1, 0, 1))
    #     pcdu.show_pcd(pcdu.trans_pcd(pcd_i, init_eemat4), rgba=(1, 1, 0, .4))
    #     pcdu.show_pcd(pcdu.trans_pcd(pcd_gt, init_eemat4), rgba=(0, 1, 1, 1))
    #
    #     pcdu.show_cam(rm.homomat_from_posrot(init_eerot.dot(cam_pos) + init_eepos,
    #                                          rot=rm.rotmat_from_axangle((0, 0, 1), np.pi / 2)))
    #
    #     base.run()

    print(cov_list)
    print(cov_opt_list)
    # gm.gen_frame().attach_to(base)
    # base.run()
