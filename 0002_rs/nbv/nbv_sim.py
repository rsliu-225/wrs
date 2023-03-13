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
import datagenerator.data_utils as du
import pcn.inference as pcn
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import visualization.panda.world as wd
import motionplanner.nbv_pcn_opt_solver as nbv_solver
import nbv_utils as nu


def _o3d_debug(pts_nbv, nrmls_nbv, confs_nbv, o3dmesh, o3dpcd, o3dpcd_tmp, rot, o3dpcd_o=None):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
    coord_tmp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
    coord_tmp.rotate(rot, center=(0, 0, 0))
    o3dpcd_tmp.paint_uniform_color(nu.COLOR[5])
    o3dpcd_tmp.normals = o3d.utility.Vector3dVector([])

    o3d.visualization.draw_geometries([o3dpcd, coord])
    o3d.visualization.draw_geometries([o3dmesh, coord])
    o3d.visualization.draw_geometries([o3dpcd, o3dmesh, coord])
    if o3dpcd_o is not None:
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])
    nu.show_nbv_o3d(pts_nbv, nrmls_nbv, confs_nbv, o3dpcd, coord, o3dpcd_o)
    o3dmesh.compute_vertex_normals()
    o3dmesh.rotate(rot, center=(0, 0, 0))
    # o3dpcd.paint_uniform_color(nu.COLOR[0])
    o3dpcd_tmp_vis = copy.deepcopy(o3dpcd_tmp)
    o3dpcd_tmp_vis.rotate(rot, center=(0, 0, 0))

    o3d.visualization.draw_geometries([o3dpcd_tmp_vis, coord_tmp])
    o3d.visualization.draw_geometries([o3dmesh, coord_tmp])
    o3d.visualization.draw_geometries([o3dpcd_tmp_vis, o3dmesh, coord_tmp])
    o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd, coord])


def _o3d_debug_opt(o3dmesh, o3dpcd, o3dpcd_tmp, rot, trans, o3dpcd_o):
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
    coord_tmp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
    coord_tmp.rotate(rot, center=(0, 0, 0))
    coord_tmp.translate(trans)
    o3dpcd_tmp.paint_uniform_color(nu.COLOR[5])
    o3dpcd_tmp.normals = o3d.utility.Vector3dVector([])

    o3d.visualization.draw_geometries([o3dpcd, coord])
    o3d.visualization.draw_geometries([o3dmesh, coord])
    o3d.visualization.draw_geometries([o3dpcd, o3dmesh, coord])
    o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])

    o3dmesh.compute_vertex_normals()
    o3dmesh.rotate(rot, center=(0, 0, 0))
    o3dmesh.translate(trans)
    o3dpcd_tmp_vis = copy.deepcopy(o3dpcd_tmp)
    o3dpcd_tmp_vis.rotate(rot, center=(0, 0, 0))
    o3dpcd_tmp_vis.translate(trans)

    o3d.visualization.draw_geometries([o3dpcd_tmp_vis, coord_tmp])
    o3d.visualization.draw_geometries([o3dmesh, coord_tmp])
    o3d.visualization.draw_geometries([o3dpcd_tmp_vis, o3dmesh, coord_tmp])
    o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd, coord])


def run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, cov_tor=.001, goal=.05,
            vis_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------pcn------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)

    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

    while coverage < goal and cnt < 5:
        cnt += 1
        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt - 1] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        o3dpcd_o = o3dh.nparray2o3dpcd(pcd_o)
        if toggledebug:
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            o3d.io.write_point_cloud(os.path.join(os.getcwd(), 'tmp', f'{cnt}_i.pcd'), o3dpcd)
            o3d.io.write_point_cloud(os.path.join(os.getcwd(), 'tmp', f'{cnt}_o.pcd'), o3dpcd_o)

        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_i, pcd_o, cam_pos=cam_pos, theta=None, toggledebug=True)
        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        trans = pts_nbv[0]
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      vis_threshold=vis_threshold, toggledebug=False,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            _o3d_debug(pts_nbv, nrmls_nbv, confs_nbv, o3dmesh, o3dpcd, o3dpcd_tmp, rot, o3dpcd_o)
        o3dpcd += o3dpcd_tmp
        if toggledebug:
            pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
            o3dpcd_o = du.nparray2o3dpcd(pcd_o)
            o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            o3d.visualization.draw_geometries([o3dpcd, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json'), 'w'))


def run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, cov_tor=.001, goal=.05,
                vis_threshold=np.radians(75), toggledebug=False):
    # nbv_opt = nbv_solver.NBVOptimizer(model_name=model_name, load_model=load_model, toggledebug=False)
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------pcn+opt------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)

    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

    while coverage < goal and cnt < 5:
        cnt += 1
        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        o3dpcd_o = du.nparray2o3dpcd(pcd_o)
        exp_dict[cnt - 1] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, coord])
            o3d.visualization.draw_geometries([o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])
            o3d.io.write_point_cloud(os.path.join(os.getcwd(), 'tmp', f'{cnt}_i.pcd'), o3dpcd)
            o3d.io.write_point_cloud(os.path.join(os.getcwd(), 'tmp', f'{cnt}_o.pcd'), o3dpcd_o)

        # trans, rot = nbv_opt.solve(pcd_i, cam_pos, method='COBYLA')
        trans, rot, time_cost = pcdu.opt_nbv_pcn(pcd_i, model_name, load_model, cam_pos=cam_pos)
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      vis_threshold=vis_threshold, toggledebug=False,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            # coord_tmp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            # coord_tmp.rotate(rot, center=(0, 0, 0))
            #
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            # o3dmesh.rotate(rot, center=(0, 0, 0))
            # # o3dpcd.paint_uniform_color(nu.COLOR[0])
            # # o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            # o3dpcd_tmp_vis = copy.deepcopy(o3dpcd_tmp)
            # o3dpcd_tmp_vis.rotate(rot, center=(0, 0, 0))
            # o3dpcd_tmp_vis.paint_uniform_color(nu.COLOR[5])
            # o3dpcd_tmp.paint_uniform_color(nu.COLOR[5])
            #
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])
            # o3d.visualization.draw_geometries([o3dpcd_tmp_vis, coord_tmp])
            # o3d.visualization.draw_geometries([o3dmesh, coord_tmp])
            # o3d.visualization.draw_geometries([o3dpcd_tmp_vis, o3dmesh, coord_tmp])
            # o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd, coord])
            _o3d_debug_opt(o3dmesh, o3dpcd, o3dpcd_tmp, rot, trans, o3dpcd_o)
        o3dpcd += o3dpcd_tmp
        if toggledebug:
            pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
            o3dpcd_o = du.nparray2o3dpcd(pcd_o)
            o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            o3d.visualization.draw_geometries([o3dpcd, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_opt_{f.split(".ply")[0]}.json'), 'w'))


def run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, cov_tor=.001, goal=.05,
            vis_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------org------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)
    o3dpcd.paint_uniform_color(nu.COLOR[0])

    while coverage < goal and cnt < 5:
        cnt += 1
        exp_dict[cnt - 1] = {'input': pcd_i.tolist()}
        pts, nrmls, confs = pcdu.cal_conf(pcd_i, voxel_size=.005, cam_pos=cam_pos, theta=None)
        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)
        # print(len(confs_nbv))
        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        trans = pts_nbv[0]
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      vis_threshold=vis_threshold, toggledebug=False,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            _o3d_debug(pts_nbv, nrmls_nbv, confs_nbv, o3dmesh, o3dpcd, o3dpcd_tmp, rot)
        o3dpcd += o3dpcd_tmp
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json'), 'w'))


def run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, cov_tor=.001, goal=.05,
               vis_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------random------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360))
    icomats = [x for row in icomats for x in row]

    while coverage < goal and cnt < 5:
        cnt += 1
        exp_dict[cnt - 1] = {'input': pcd_i.tolist()}
        rot = random.choice(icomats)
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center,
                                      vis_threshold=vis_threshold, toggledebug=False,
                                      add_noise_vt=False, add_occ_vt=False, add_occ_rnd=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_tmp.paint_uniform_color(nu.COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_tmp, o3dmesh, coord])
        o3dpcd += o3dpcd_tmp
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=cov_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'random_{f.split(".ply")[0]}.json'), 'w'))


if __name__ == '__main__':
    RES_FO_NAME = 'res_60_rlen'

    model_name = 'pcn'
    load_model = 'pcn_emd_all/best_emd_network.pth'
    cam_pos = [0, 0, .4]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'D:/nbv_mesh/'
    if not os.path.exists(path):
        path = 'E:/liu/nbv_mesh/'
    # cat_list = ['plat', 'tmpl']
    cat_list = ['bspl_4']
    cov_tor = .001
    goal = .95
    vis_threshold = np.radians(60)
    for cat in cat_list:
        if not os.path.exists(os.path.join(path, cat, RES_FO_NAME)):
            os.makedirs(os.path.join(path, cat, RES_FO_NAME))
        for f in os.listdir(os.path.join(path, cat, 'mesh')):
            print(f'-----------{f}------------')
            # if os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json')) and \
            #         os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_opt_{f.split(".ply")[0]}.json')) and \
            #         os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json')) and \
            #         os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'random_{f.split(".ply")[0]}.json')):
            #     continue
            f = '0001.ply'

            o3dpcd_init = \
                nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], np.eye(3), [0, 0, 0],
                                          rnd_occ_ratio_rng=(.2, .4), nrml_occ_ratio_rng=(.2, .6),
                                          vis_threshold=vis_threshold, toggledebug=True,
                                          occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
                                          noise_cnt=random.randint(1, 5), fov=False,
                                          add_occ_vt=True, add_noise_vt=False, add_occ_rnd=False, add_noise_pts=True)

            o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
            o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')

            run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                    goal=goal, cov_tor=cov_tor, vis_threshold=vis_threshold, toggledebug=True)
            run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                        goal=goal, cov_tor=cov_tor, vis_threshold=vis_threshold, toggledebug=True)
            run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, cov_tor=cov_tor,
                    vis_threshold=vis_threshold, toggledebug=False)
            run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, goal=goal, cov_tor=cov_tor,
                       vis_threshold=vis_threshold, toggledebug=False)
