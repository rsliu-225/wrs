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
import motionplanner.pcn_nbv_solver as nbv_solver
import nbv_utils as nu

RES_FO_NAME = 'res_60_rlen'


def run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, coverage_tor=.001, goal=.05,
            visible_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------pcn------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
    print('init coverage:', init_coverage)

    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

    while coverage < goal and cnt < 5:
        cnt += 1
        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt - 1] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        o3dpcd_o = du.nparray2o3dpcd(pcd_o)
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

        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_pcn(pcd_i, pcd_o, cam_pos=cam_pos, theta=None, toggledebug=True)

        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        trans = pts_nbv[0]
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      visible_threshold=visible_threshold, toggledebug=False,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            nbv_mesh_list = []
            for i in range(len(pts_nbv)):
                nbv_mesh_list.append(nu.gen_o3d_arrow(pts_nbv[i], pts_nbv[i] + rm.unit_vector(nrmls_nbv[i]) * .02,
                                                      rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]]))
            circle_mesh = nu.gen_o3d_sphere(pts_nbv[0], radius=.005, rgb=[0, 0, 1])
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            o3dmesh.rotate(rot, center=(0, 0, 0))
            # o3dpcd.paint_uniform_color(nu.COLOR[0])
            # o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_tmp_vis = copy.deepcopy(o3dpcd_tmp)
            o3dpcd_tmp_vis.rotate(rot, center=(0, 0, 0))
            o3dpcd_tmp_vis.paint_uniform_color(nu.COLOR[4])
            o3dpcd_tmp.paint_uniform_color(nu.COLOR[4])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord, circle_mesh] + nbv_mesh_list)
            o3d.visualization.draw_geometries([o3dpcd_tmp_vis, coord])
            o3d.visualization.draw_geometries([o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd_tmp_vis, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd, coord])
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
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json'), 'w'))


def run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model, coverage_tor=.001, goal=.05,
                visible_threshold=np.radians(75), toggledebug=False):
    import localenv.envloader as el

    rbt = el.loadXarm(showrbt=True)
    seedjntagls = rbt.get_jnt_values('arm')
    nbv_opt = nbv_solver.NBVOptimizer(rbt, model_name=model_name, load_model=load_model, toggledebug=False)
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------pcn+opt------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
    print('init coverage:', init_coverage)

    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

    while coverage < goal and cnt < 5:
        cnt += 1
        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt - 1] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            o3dpcd_o = du.nparray2o3dpcd(pcd_o)
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

        trans, rot = nbv_opt.solve(seedjntagls, pcd_i, cam_pos, method='COBYLA')
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      visible_threshold=visible_threshold, toggledebug=False,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            o3dmesh.rotate(rot, center=(0, 0, 0))
            # o3dpcd.paint_uniform_color(nu.COLOR[0])
            # o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_tmp_vis = copy.deepcopy(o3dpcd_tmp)
            o3dpcd_tmp_vis.rotate(rot, center=(0, 0, 0))
            o3dpcd_tmp_vis.paint_uniform_color(nu.COLOR[0])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02)
            o3d.visualization.draw_geometries([o3dpcd_tmp_vis, coord])
            o3d.visualization.draw_geometries([o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd_tmp_vis, o3dmesh, coord])
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
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_opt_{f.split(".ply")[0]}.json'), 'w'))


def run_pcn_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, coverage_tor=.001, goal=.05,
                visible_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------pcn+nbv------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
    print('init coverage:', init_coverage)

    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

    while coverage < goal and cnt < 5:
        cnt += 1
        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        exp_dict[cnt - 1] = {'input': pcd_i.tolist(), 'pcn_output': pcd_o.tolist()}
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd_o = du.nparray2o3dpcd(pcd_o)
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dpcd_gt, coord])

        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_i, pcd_o, cam_pos=cam_pos, theta=None, toggledebug=True)
        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        trans = pts_nbv[0]
        o3dpcd_tmp = \
            nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, trans,
                                      visible_threshold=visible_threshold, toggledebug=False,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False)
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
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'nbvpcn_{f.split(".ply")[0]}.json'), 'w'))


def run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, coverage_tor=.001, goal=.05,
            visible_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------org------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
    print('init coverage:', init_coverage)
    exp_dict['gt'] = pcd_gt.tolist()
    exp_dict['init_coverage'] = init_coverage
    o3dpcd = copy.deepcopy(o3dpcd_init)

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
                                      visible_threshold=visible_threshold, toggledebug=False,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            nbv_mesh_list = []
            for i in range(len(pts_nbv)):
                nbv_mesh_list.append(nu.gen_o3d_arrow(pts_nbv[i], pts_nbv[i] + nrmls_nbv[i] * .02,
                                                      rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]]))
            circle_mesh = nu.gen_o3d_sphere(pts_nbv[0], radius=.005, rgb=[.7, .7, .7])
            # o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd.paint_uniform_color(nu.COLOR[0])
            o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            o3dpcd_tmp.paint_uniform_color(nu.COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_tmp, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, coord, circle_mesh] + nbv_mesh_list)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_tmp, coord])
        o3dpcd += o3dpcd_tmp
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json'), 'w'))


def run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, coverage_tor=.001, goal=.05,
               visible_threshold=np.radians(75), toggledebug=False):
    coverage = 0
    cnt = 0
    exp_dict = {}
    print(f'-----------random------------')
    rot_center = [0, 0, 0]

    pcd_i = np.asarray(o3dpcd_init.points)
    pcd_gt = np.asarray(o3dpcd_gt.points)

    init_coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
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
                                      visible_threshold=visible_threshold, toggledebug=False,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False)
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
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'random_{f.split(".ply")[0]}.json'), 'w'))


if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_rlen/best_emd_network.pth'
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()

    # path = 'E:/liu/nbv_mesh/'
    path = 'D:/nbv_mesh/'
    cat_list = ['bspl_3', 'bspl_4', 'bspl_5']
    # cat_list = ['plat', 'tmpl']
    # cat_list = ['rlen_3', 'rlen_4', 'rlen_5']
    coverage_tor = .001
    goal = .95
    visible_threshold = np.radians(60)
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
                                          visible_threshold=visible_threshold, toggledebug=False,
                                          occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
                                          noise_cnt=random.randint(1, 5),
                                          add_vt_occ=True, add_noise=False, add_rnd_occ=False, add_noise_pts=True)
            # o3dpcd_init, ind = o3dpcd_init.remove_radius_outlier(nb_points=50, radius=0.005)

            o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
            o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')

            run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                    goal=goal, coverage_tor=coverage_tor, visible_threshold=visible_threshold, toggledebug=False)
            run_pcn_opt(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                        goal=goal, coverage_tor=coverage_tor, visible_threshold=visible_threshold, toggledebug=False)
            run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
                    visible_threshold=visible_threshold, toggledebug=False)
            run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
                       visible_threshold=visible_threshold, toggledebug=False)
