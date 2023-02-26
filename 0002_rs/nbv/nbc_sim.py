import copy
import math
import os
import random
import json
import h5py
import numpy as np
import open3d as o3d

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import config
import datagenerator.data_utils as du
import pcn.inference as pcn
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import visualization.panda.world as wd
import motionplanner.nbc_solver as nbcs
import motionplanner.nbc_pcn_opt_solver as nbcs_conf

import nbv_utils as nu
import modeling.geometric_model as gm
from multiprocessing import Process


def run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, cov_tor=.001, goal=.5,
            vis_threshold=np.radians(75), toggledebug=False, toggledebug_p3d=False):
    flag = True
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    o3dpcd_init.paint_uniform_color(nu.COLOR[0])
    o3dpcd_gt.paint_uniform_color(nu.COLOR[1])

    rot_center = [0, 0, 0]
    max_a = np.pi / 18
    max_dist = 1.5
    coverage = 0
    cnt = 0
    exp_dict = {}
    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    seedjntagls = rbt.get_jnt_values('arm')

    print(f'-----------pcn------------')
    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    exp_dict['init_jnts'] = np.asarray(seedjntagls).tolist()
    o3dpcd = copy.deepcopy(o3dpcd_init)
    o3dpcd.paint_uniform_color(nu.COLOR[0])

    rbt.fk('arm', seedjntagls)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos_origin = pcdu.trans_pcd([cam_pos], np.linalg.inv(init_eemat4))[0]
    pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))

    while coverage < goal and cnt < 9:
        rbt.fk('arm', seedjntagls)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        pcd_inhnd = pcdu.trans_pcd(pcd_i, init_eemat4)
        pcd_o_inhnd = pcdu.trans_pcd(pcd_o, init_eemat4)

        o3dpcd_inhnd = o3dh.nparray2o3dpcd(pcd_inhnd)
        o3dpcd_o_inhnd = o3dh.nparray2o3dpcd(pcd_o_inhnd)
        o3dpcd_inhnd.paint_uniform_color(nu.COLOR[0])
        o3dpcd_o_inhnd.paint_uniform_color(nu.COLOR[2])
        pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv = \
            pcdu.cal_nbv_pcn(pcd_inhnd, pcd_o_inhnd, cam_pos=cam_pos, theta=None, toggledebug=toggledebug_p3d)

        exp_dict[cnt]['pts_nbv'] = pts_nbv_inhnd.tolist()
        exp_dict[cnt]['nrmls_nbv'] = nrmls_nbv_inhnd.tolist()
        rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10)

        if toggledebug:
            coord_inhnd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            coord_inhnd.transform(init_eemat4)
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord_inhnd, o3dpcd_inhnd])
            nu.show_nbv_o3d(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, o3dpcd_inhnd, coord_inhnd)
        if toggledebug_p3d:
            gm.gen_frame().attach_to(base)
            pcdu.show_pcd(pcd_inhnd)
            rbt.gen_meshmodel().attach_to(base)
            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)

        nbc_solver = nbcs.NBCOptimizerVec(rbt, max_a=max_a, max_dist=max_dist, toggledebug=False)
        jnts, transmat4, _, time_cost = nbc_solver.solve(seedjntagls, pts_nbv_inhnd[0], nrmls_nbv_inhnd[0], cam_pos)

        if jnts is None:
            print('Planning Failed!')
            flag = False
            break

        rbt.fk('arm', jnts)
        rbt_o3dmesh_nxt = nu.rbt2o3dmesh(rbt, link_num=10)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)

        if toggledebug_p3d:
            pcd_next = pcdu.trans_pcd(pcd_i, eemat4)
            pts_nbv_inhnd = pcdu.trans_pcd(pts_nbv_inhnd, transmat4)
            nrmls_nbv_inhnd = pcdu.trans_pcd(nrmls_nbv_inhnd, transmat4)
            pcdu.show_pcd(pcd_next, rgba=(0, 1, 0, 1))
            rbt.gen_meshmodel(rgba=(0, 1, 0, .5)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)
            base.run()

        '''
        get new pcd
        '''
        rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt.transform(np.linalg.inv(init_eemat4))
        transmat4_origin = np.linalg.inv(init_eemat4).dot(eemat4)
        coord_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        coord_nxt.transform(transmat4_origin)
        if toggledebug:
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, o3dpcd])
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, rbt_o3dmesh_nxt, coord_nxt, o3dpcd])

        o3dpcd_nxt = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], othermesh=[rbt_o3dmesh_nxt],
                                      trans=transmat4_origin[:3, 3], rot=transmat4_origin[:3, :3],
                                      rot_center=rot_center, vis_threshold=vis_threshold, toggledebug=toggledebug,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)

        exp_dict[cnt]['add'] = np.asarray(o3dpcd_nxt.points).tolist()
        exp_dict[cnt]['jnts'] = np.asarray(jnts).tolist()
        exp_dict[cnt]['time_cost'] = time_cost

        if toggledebug:
            o3dpcd_nxt.paint_uniform_color(nu.COLOR[5])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_nxt, coord])

        o3dpcd += o3dpcd_nxt
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        if toggledebug:
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_gt, coord])
        exp_dict[cnt]['coverage'] = coverage

        print('coverage:', coverage)
        cnt += 1
        seedjntagls = jnts

    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json'), 'w'))
    return flag


def run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, cov_tor=.001, goal=.5,
                vis_threshold=np.radians(75), toggledebug=False, toggledebug_p3d=False):
    flag = True
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    o3dpcd_init.paint_uniform_color(nu.COLOR[0])
    o3dpcd_gt.paint_uniform_color(nu.COLOR[1])

    rot_center = [0, 0, 0]
    max_a = np.pi / 18
    max_dist = 1.5
    coverage = 0
    cnt = 0
    exp_dict = {}
    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    seedjntagls = rbt.get_jnt_values('arm')

    print(f'-----------pcn_opt------------')
    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    exp_dict['init_jnts'] = np.asarray(seedjntagls).tolist()
    o3dpcd = copy.deepcopy(o3dpcd_init)
    o3dpcd.paint_uniform_color(nu.COLOR[0])

    rbt.fk('arm', seedjntagls)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos_origin = pcdu.trans_pcd([cam_pos], np.linalg.inv(init_eemat4))[0]
    pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))

    while coverage < goal and cnt < 9:
        rbt.fk('arm', seedjntagls)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        pcd_inhnd = pcdu.trans_pcd(pcd_i, init_eemat4)
        pcd_o_inhnd = pcdu.trans_pcd(pcd_o, init_eemat4)

        o3dpcd_inhnd = o3dh.nparray2o3dpcd(pcd_inhnd)
        o3dpcd_o_inhnd = o3dh.nparray2o3dpcd(pcd_o_inhnd)
        o3dpcd_inhnd.paint_uniform_color(nu.COLOR[0])
        o3dpcd_o_inhnd.paint_uniform_color(nu.COLOR[2])
        pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv = \
            pcdu.cal_nbv_pcn(pcd_inhnd, pcd_o_inhnd, cam_pos=cam_pos, theta=None, toggledebug=toggledebug_p3d)

        exp_dict[cnt]['pts_nbv'] = pts_nbv_inhnd.tolist()
        exp_dict[cnt]['nrmls_nbv'] = nrmls_nbv_inhnd.tolist()
        rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10)

        if toggledebug:
            coord_inhnd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            coord_inhnd.transform(init_eemat4)
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord_inhnd, o3dpcd_inhnd])
            nu.show_nbv_o3d(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, o3dpcd_inhnd, coord_inhnd)
        if toggledebug_p3d:
            gm.gen_frame().attach_to(base)
            pcdu.show_pcd(pcd_inhnd)
            rbt.gen_meshmodel().attach_to(base)
            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)

        nbc_opt = nbcs_conf.PCNNBCOptimizer(rbt, releemat4=relmat4, toggledebug=False)
        jnts, transmat4, _, time_cost = nbc_opt.solve(seedjntagls, pcd_i, cam_pos, method='COBYLA')

        if jnts is None:
            print('Planning Failed!')
            flag = False
            break

        rbt.fk('arm', jnts)
        rbt_o3dmesh_nxt = nu.rbt2o3dmesh(rbt, link_num=10)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)

        if toggledebug_p3d:
            pcd_next = pcdu.trans_pcd(pcd_i, eemat4)
            pts_nbv_inhnd = pcdu.trans_pcd(pts_nbv_inhnd, transmat4)
            nrmls_nbv_inhnd = pcdu.trans_pcd(nrmls_nbv_inhnd, transmat4)
            pcdu.show_pcd(pcd_next, rgba=(0, 1, 0, 1))
            rbt.gen_meshmodel(rgba=(0, 1, 0, .5)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)
            base.run()

        '''
        get new pcd
        '''
        rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt.transform(np.linalg.inv(init_eemat4))
        transmat4_origin = np.linalg.inv(init_eemat4).dot(eemat4)
        coord_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        coord_nxt.transform(transmat4_origin)
        if toggledebug:
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, o3dpcd])
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, rbt_o3dmesh_nxt, coord_nxt, o3dpcd])

        o3dpcd_nxt = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], othermesh=[rbt_o3dmesh_nxt],
                                      trans=transmat4_origin[:3, 3], rot=transmat4_origin[:3, :3],
                                      rot_center=rot_center, vis_threshold=vis_threshold, toggledebug=toggledebug,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)

        exp_dict[cnt]['add'] = np.asarray(o3dpcd_nxt.points).tolist()
        exp_dict[cnt]['jnts'] = np.asarray(jnts).tolist()
        exp_dict[cnt]['time_cost'] = time_cost

        if toggledebug:
            o3dpcd_nxt.paint_uniform_color(nu.COLOR[5])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_nxt, coord])

        o3dpcd += o3dpcd_nxt
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        if toggledebug:
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_gt, coord])
        exp_dict[cnt]['coverage'] = coverage

        print('coverage:', coverage)
        cnt += 1
        seedjntagls = jnts

    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_opt_{f.split(".ply")[0]}.json'), 'w'))
    return flag


def run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, cov_tor=.001, goal=.5,
            vis_threshold=np.radians(75), toggledebug=True, toggledebug_p3d=False):
    flag = True
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    o3dpcd_init.paint_uniform_color(nu.COLOR[0])
    o3dpcd_gt.paint_uniform_color(nu.COLOR[1])

    rot_center = [0, 0, 0]
    max_a = np.pi / 18
    max_dist = 1.5
    coverage = 0
    cnt = 0
    exp_dict = {}
    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    seedjntagls = rbt.get_jnt_values('arm')

    print(f'-----------org------------')
    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    exp_dict['init_jnts'] = np.asarray(seedjntagls).tolist()
    o3dpcd = copy.deepcopy(o3dpcd_init)
    o3dpcd.paint_uniform_color(nu.COLOR[0])

    rbt.fk('arm', seedjntagls)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos_origin = pcdu.trans_pcd([cam_pos], np.linalg.inv(init_eemat4))[0]
    pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))

    while coverage < goal and cnt < 9:
        rbt.fk('arm', seedjntagls)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

        exp_dict[cnt] = {'input': pcd_i.tolist()}
        pcd_inhnd = pcdu.trans_pcd(pcd_i, init_eemat4)

        o3dpcd_inhnd = o3dh.nparray2o3dpcd(pcd_inhnd)
        o3dpcd_inhnd.paint_uniform_color(nu.COLOR[0])
        pts, nrmls, confs = pcdu.cal_conf(pcd_inhnd, voxel_size=.005, cam_pos=cam_pos, theta=None)
        pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)
        exp_dict[cnt]['pts_nbv'] = pts_nbv_inhnd.tolist()
        exp_dict[cnt]['nrmls_nbv'] = nrmls_nbv_inhnd.tolist()
        rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10)

        if toggledebug:
            coord_inhnd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            coord_inhnd.transform(init_eemat4)
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord_inhnd, o3dpcd_inhnd])
            nu.show_nbv_o3d(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, o3dpcd_inhnd, coord_inhnd)
        if toggledebug_p3d:
            gm.gen_frame().attach_to(base)
            pcdu.show_pcd(pcd_inhnd)
            rbt.gen_meshmodel().attach_to(base)
            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)

        nbc_solver = nbcs.NBCOptimizerVec(rbt, max_a=max_a, max_dist=max_dist, toggledebug=False)
        jnts, transmat4, _, time_cost = nbc_solver.solve(seedjntagls, pts_nbv_inhnd[0], nrmls_nbv_inhnd[0], cam_pos)

        if jnts is None:
            print('Planning Failed!')
            flag = False
            break

        rbt.fk('arm', jnts)
        rbt_o3dmesh_nxt = nu.rbt2o3dmesh(rbt, link_num=10)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)

        if toggledebug_p3d:
            pcd_next = pcdu.trans_pcd(pcd_i, eemat4)
            pts_nbv_inhnd = pcdu.trans_pcd(pts_nbv_inhnd, transmat4)
            nrmls_nbv_inhnd = pcdu.trans_pcd(nrmls_nbv_inhnd, transmat4)
            pcdu.show_pcd(pcd_next, rgba=(0, 1, 0, 1))
            rbt.gen_meshmodel(rgba=(0, 1, 0, .5)).attach_to(base)
            nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .04)
            base.run()

        '''
        get new pcd
        '''
        rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt.transform(np.linalg.inv(init_eemat4))
        transmat4_origin = np.linalg.inv(init_eemat4).dot(eemat4)
        coord_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        coord_nxt.transform(transmat4_origin)
        if toggledebug:
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, o3dpcd])
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, rbt_o3dmesh_nxt, coord_nxt, o3dpcd])

        o3dpcd_nxt = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], othermesh=[rbt_o3dmesh_nxt],
                                      trans=transmat4_origin[:3, 3], rot=transmat4_origin[:3, :3],
                                      rot_center=rot_center, vis_threshold=vis_threshold,
                                      toggledebug=toggledebug,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        o3dpcd_nxt = nu.filer_pcd_by_campos(o3dpcd_nxt, cam_pos_origin)

        exp_dict[cnt]['add'] = np.asarray(o3dpcd_nxt.points).tolist()
        exp_dict[cnt]['jnts'] = np.asarray(jnts).tolist()
        exp_dict[cnt]['time_cost'] = time_cost

        if toggledebug:
            o3dpcd_nxt.paint_uniform_color(nu.COLOR[5])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_nxt, coord])

        o3dpcd += o3dpcd_nxt
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        if toggledebug:
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_gt, coord])
        exp_dict[cnt]['coverage'] = coverage

        print('coverage:', coverage)
        cnt += 1
        seedjntagls = jnts

    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json'), 'w'))
    return flag


def run_random(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, cov_tor=.001, goal=.05,
               vis_threshold=np.radians(75), toggledebug=False, toggledebug_p3d=False):
    flag = True
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
    o3dpcd_init.paint_uniform_color(nu.COLOR[0])
    o3dpcd_gt.paint_uniform_color(nu.COLOR[1])

    rot_center = [0, 0, 0]
    max_a = np.pi / 18
    max_dist = 1.5
    coverage = 0
    cnt = 0
    exp_dict = {}
    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    seedjntagls = rbt.get_jnt_values('arm')

    print(f'-----------random------------')
    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    exp_dict['init_jnts'] = np.asarray(seedjntagls).tolist()
    o3dpcd = copy.deepcopy(o3dpcd_init)
    o3dpcd.paint_uniform_color(nu.COLOR[0])

    rbt.fk('arm', seedjntagls)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos_origin = pcdu.trans_pcd([cam_pos], np.linalg.inv(init_eemat4))[0]
    if toggledebug_p3d:
        pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))

    while coverage < goal and cnt < 9:
        rbt.fk('arm', seedjntagls)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

        exp_dict[cnt] = {'input': pcd_i.tolist()}
        pcd_inhnd = pcdu.trans_pcd(pcd_i, init_eemat4)

        o3dpcd_inhnd = o3dh.nparray2o3dpcd(pcd_inhnd)
        o3dpcd_inhnd.paint_uniform_color(nu.COLOR[0])
        pts_nbv, nrmls_nbv, confs_nbv = gen_random_vec(pcd_inhnd, threshold=np.pi / 1800)
        # if len(nrmls_nbv) == 0:
        #     pts_nbv, nrmls_nbv, confs_nbv = gen_random_vec(pcd_inhnd, threshold=np.pi / 180)
        random.shuffle(nrmls_nbv)

        exp_dict[cnt]['pts_nbv'] = pts_nbv.tolist()
        exp_dict[cnt]['nrmls_nbv'] = nrmls_nbv.tolist()
        rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10)

        if toggledebug:
            coord_inhnd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
            coord_inhnd.transform(init_eemat4)
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord_inhnd, o3dpcd_inhnd])
        if toggledebug_p3d:
            gm.gen_frame().attach_to(base)
            nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, .04)
            pcdu.show_pcd(pcd_inhnd)
            rbt.gen_meshmodel().attach_to(base)
            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)

        nbc_solver = nbcs.NBCOptimizerVec(rbt, max_a=max_a, max_dist=max_dist, toggledebug=False)
        jnts, transmat4, _, time_cost = nbc_solver.solve(seedjntagls, pts_nbv[0], nrmls_nbv[0], cam_pos)

        if jnts is None:
            print('Planning Failed!')
            flag = False
            break

        rbt.fk('arm', jnts)
        rbt_o3dmesh_nxt = nu.rbt2o3dmesh(rbt, link_num=10)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)

        if toggledebug_p3d:
            pcd_next = pcdu.trans_pcd(pcd_i, eemat4)
            pts_nbv = pcdu.trans_pcd(pts_nbv, transmat4)
            nrmls_nbv = pcdu.trans_pcd(nrmls_nbv, transmat4)
            nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, .04)
            pcdu.show_pcd(pcd_next, rgba=(0, 1, 0, 1))
            rbt.gen_meshmodel(rgba=(0, 1, 0, .5)).attach_to(base)
            base.run()

        '''
        get new pcd
        '''
        rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt.transform(np.linalg.inv(init_eemat4))
        transmat4_origin = np.linalg.inv(init_eemat4).dot(eemat4)
        coord_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        coord_nxt.transform(transmat4_origin)
        if toggledebug:
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, o3dpcd])
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, rbt_o3dmesh_nxt, coord_nxt, o3dpcd])

        o3dpcd_nxt = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], othermesh=[rbt_o3dmesh_nxt],
                                      trans=transmat4_origin[:3, 3], rot=transmat4_origin[:3, :3],
                                      rot_center=rot_center, vis_threshold=vis_threshold,
                                      toggledebug=toggledebug,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        o3dpcd_nxt = nu.filer_pcd_by_campos(o3dpcd_nxt, cam_pos_origin)

        exp_dict[cnt]['add'] = np.asarray(o3dpcd_nxt.points).tolist()
        exp_dict[cnt]['jnts'] = np.asarray(jnts).tolist()
        exp_dict[cnt]['time_cost'] = time_cost

        if toggledebug:
            o3dpcd_nxt.paint_uniform_color(nu.COLOR[5])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_nxt, coord])

        o3dpcd += o3dpcd_nxt
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        if toggledebug:
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_gt, coord])
        exp_dict[cnt]['coverage'] = coverage

        print('coverage:', coverage)
        cnt += 1
        seedjntagls = jnts

    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'random_{f.split(".ply")[0]}.json'), 'w'))
    return flag


def gen_random_vec(pts, threshold=np.pi / 6, toggledebug=False):
    import basis.trimesh as trm
    o3dpcd = o3dh.nparray2o3dpcd(pts)
    o3dpcd.estimate_normals()
    nrmls = np.asarray(o3dpcd.normals)
    icos = trm.creation.icosphere(1)
    nbvs = []
    center = np.mean(np.asarray(pts), axis=0)
    for i, vt in enumerate(icos.vertices):
        flag = True
        vt = np.asarray(vt) * .1
        if toggledebug:
            gm.gen_sphere(vt + center, rgba=(.5, .5, .5, .5), radius=.002).attach_to(base)
        for nrml in nrmls:
            if min([np.pi - rm.angle_between_vectors(nrml, vt), rm.angle_between_vectors(nrml, vt)]) < threshold:
                flag = False
                break
        if flag:
            nbvs.append(vt)
            if toggledebug:
                gm.gen_sphere(vt + center, rgba=(0, 0, 1, 1), radius=.002).attach_to(base)
                gm.gen_arrow(center, vt + center, rgba=(0, 0, 1, 1)).attach_to(base)
    return np.tile(center, (len(nbvs), 1)), np.asarray(nbvs), [0] * len(nbvs)


def runInParallel(fn, args):
    proc = []
    for arg in args:
        p = Process(target=fn, args=arg)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


if __name__ == '__main__':
    import localenv.envloader as el

    model_name = 'pcn'
    load_model = 'pcn_emd_rlen/best_emd_network.pth'
    RES_FO_NAME = 'res_75_rbt'

    path = 'D:/nbv_mesh/'
    if not os.path.exists(path):
        path = 'E:/liu/nbv_mesh/'

    cat_list = ['bspl_3', 'bspl_4', 'bspl_5']
    # cat_list = ['plat', 'tmpl']
    # cat_list = ['rlen_3', 'rlen_4', 'rlen_5']
    cam_pos = [0, 0, .4]
    cov_tor = .001
    goal = .95
    vis_threshold = np.radians(75)
    relmat4 = rm.homomat_from_posrot([.02, 0, 0], np.eye(3))

    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos = pcdu.trans_pcd([cam_pos], init_eemat4)[0]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 1])

    for cat in cat_list:
        if not os.path.exists(os.path.join(path, cat, RES_FO_NAME)):
            os.makedirs(os.path.join(path, cat, RES_FO_NAME))
        for f in os.listdir(os.path.join(path, cat, 'mesh')):
            print(f'-----------{f}------------')
            if os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json')) and \
                    os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_opt_{f.split(".ply")[0]}.json')) and \
                    os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json')) and \
                    os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'random_{f.split(".ply")[0]}.json')):
                continue
            o3dpcd_init = \
                nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], np.eye(3), [0, 0, 0],
                                          rnd_occ_ratio_rng=(.2, .4), nrml_occ_ratio_rng=(.2, .6),
                                          vis_threshold=vis_threshold, toggledebug=False,
                                          occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
                                          noise_cnt=random.randint(1, 5),
                                          add_occ_vt=True, add_noise_vt=False, add_occ_rnd=False, add_noise_pts=True)
            # o3d.io.write_point_cloud('./tmp/nbc/init.pcd', o3dpcd_init)
            # o3dpcd_init = o3d.io.read_point_cloud('./tmp/nbc/init.pcd')
            o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
            o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')
            run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, goal=goal, cov_tor=cov_tor,
                    vis_threshold=vis_threshold, toggledebug=False, toggledebug_p3d=False)
            run_random(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, goal=goal, cov_tor=cov_tor,
                       vis_threshold=vis_threshold, toggledebug=False, toggledebug_p3d=False)
            run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, goal=goal,
                    cov_tor=cov_tor, vis_threshold=vis_threshold, toggledebug=False, toggledebug_p3d=False)
            run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, goal=goal,
                        cov_tor=cov_tor, vis_threshold=vis_threshold, toggledebug=False,
                        toggledebug_p3d=False)
