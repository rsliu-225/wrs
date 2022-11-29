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

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255
RES_FO_NAME = 'res_75'


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


def gen_partial_view(path, f, rot, rot_center, resolusion=(1280, 720),
                     rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6), visible_threshold=np.radians(75),
                     occ_vt_ratio=1.0, noise_vt_ratio=1.0, noise_cnt=random.randint(0, 5),
                     add_noise=False, add_vt_occ=False, add_rnd_occ=True, add_noise_pts=True, toggledebug=False):
    o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, 'mesh', f + '.ply'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    o3dmesh.rotate(rot, center=rot_center)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f'{f}_tmp.pcd'), do_render=False, convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    o3dpcd_nrml = np.asarray(o3dpcd.normals)
    vis_idx = np.argwhere(np.arccos(abs(o3dpcd_nrml.dot(np.asarray([0, 0, 1])))) < visible_threshold).flatten()
    o3dpcd = o3dpcd.select_by_index(vis_idx)
    if add_rnd_occ:
        o3dpcd = du.add_random_occ(o3dpcd, occ_ratio_rng=rnd_occ_ratio_rng)
    if add_vt_occ:
        o3dpcd = du.add_random_occ_by_nrml(o3dpcd, occ_ratio_rng=nrml_occ_ratio_rng)
        o3dpcd = du.add_random_occ_by_vt(o3dpcd, np.asarray(o3dmesh.vertices),
                                         edg_radius=5e-4, edg_sigma=5e-4, ratio=occ_vt_ratio)
    if add_noise:
        o3dpcd = du.add_guassian_noise_by_vt(o3dpcd, np.asarray(o3dmesh.vertices), np.asarray(o3dmesh.vertex_normals),
                                             noise_mean=1e-3, noise_sigma=1e-4, ratio=noise_vt_ratio)
    if add_noise_pts:
        o3dpcd = du.add_noise_pts_by_vt(o3dpcd, noise_cnt=noise_cnt, size=.01)

    o3dpcd = du.resample(o3dpcd, smp_num=2048)
    vis.destroy_window()

    if toggledebug:
        o3dpcd_org = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))
        o3dpcd_org.paint_uniform_color([0, 0.7, 1])
        o3dpcd.paint_uniform_color(COLOR[0])
        o3d.visualization.draw_geometries([o3dmesh])
        o3d.visualization.draw_geometries([o3dpcd_org])
        o3d.visualization.draw_geometries([o3dpcd])
    os.remove(os.path.join(path, f'{f}_tmp.pcd'))
    return o3dpcd


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
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd_o = du.nparray2o3dpcd(pcd_o)
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_o.paint_uniform_color(COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, coord])

        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn_kpts(pcd_i, pcd_o, theta=None, toggledebug=True)

        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        o3dpcd_tmp = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center,
                                      visible_threshold=visible_threshold,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False,
                                      toggledebug=False)
        o3dpcd_tmp.rotate(np.linalg.inv(rot), center=rot_center)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dmesh.compute_vertex_normals()
            o3dmesh.rotate(rot, center=(0, 0, 0))
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_tmp.paint_uniform_color(COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_tmp, o3dmesh, coord])
        o3dpcd += o3dpcd_tmp
        pcd_i = np.asarray(o3dpcd.points)
        coverage = pcdu.cal_coverage(pcd_i, pcd_gt, tor=coverage_tor)
        exp_dict[cnt - 1]['coverage'] = coverage
        print('coverage:', coverage)
    exp_dict['final'] = pcd_i.tolist()
    if not toggledebug:
        json.dump(exp_dict, open(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json'), 'w'))


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
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_o.paint_uniform_color(COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, o3dpcd_gt, coord])

        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_i, pcd_o, theta=None, toggledebug=True)
        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        o3dpcd_tmp = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center,
                                      visible_threshold=visible_threshold,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False,
                                      toggledebug=False)
        o3dpcd_tmp.rotate(np.linalg.inv(rot), center=rot_center)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_tmp.paint_uniform_color(COLOR[2])
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
        pts, nrmls, confs = pcdu.cal_conf(pcd_i, voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=None)
        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)

        rot = rm.rotmat_between_vectors(np.asarray(cam_pos), nrmls_nbv[0])
        rot = np.linalg.inv(rot)
        o3dpcd_tmp = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center,
                                      visible_threshold=visible_threshold,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False,
                                      toggledebug=False)
        o3dpcd_tmp.rotate(np.linalg.inv(rot), center=rot_center)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_tmp.paint_uniform_color(COLOR[2])
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.01)
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_tmp, o3dmesh, coord])
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
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]

    while coverage < goal and cnt < 5:
        cnt += 1
        exp_dict[cnt - 1] = {'input': pcd_i.tolist()}
        rot = random.choice(icomats)
        o3dpcd_tmp = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center,
                                      visible_threshold=visible_threshold,
                                      add_noise=False, add_vt_occ=False, add_rnd_occ=False, add_noise_pts=False,
                                      toggledebug=False)
        o3dpcd_tmp.rotate(np.linalg.inv(rot), center=rot_center)
        exp_dict[cnt - 1]['add'] = np.asarray(o3dpcd_tmp.points).tolist()
        if toggledebug:
            o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
            o3dpcd.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_tmp.paint_uniform_color(COLOR[2])
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
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()

    path = 'E:/liu/nbv_mesh/'
    cat = 'plat'
    coverage_tor = .0016
    goal = .95
    visible_threshold = np.radians(75)

    if not os.path.exists(os.path.join(path, cat, RES_FO_NAME)):
        os.makedirs(os.path.join(path, cat, RES_FO_NAME))
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        print(f'-----------{f}------------')
        if os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json')) and \
                os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json')):
            continue

        o3dpcd_init = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], np.eye(3), [0, 0, 0],
                                       rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6),
                                       visible_threshold=visible_threshold,
                                       occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
                                       add_vt_occ=True, add_noise=False, add_rnd_occ=False, add_noise_pts=True,
                                       toggledebug=False)
        # o3dpcd_init, ind = o3dpcd_init.remove_radius_outlier(nb_points=50, radius=0.005)

        o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
        o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')

        run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                goal=goal, coverage_tor=coverage_tor, visible_threshold=visible_threshold, toggledebug=False)
        run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
                visible_threshold=visible_threshold, toggledebug=False)
        run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
                   visible_threshold=visible_threshold, toggledebug=False)
        # run_pcn_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
        #             visible_threshold=visible_threshold, toggledebug=False)
