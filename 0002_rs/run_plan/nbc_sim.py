import os
import h5py
import random
import numpy as np
import open3d as o3d

import basis.robot_math as rm
# import localenv.envloader as el
import modeling.geometric_model as gm
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd
import pcn.inference as pcn
import datagenerator.data_utils as du

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255


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
                     rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6),
                     occ_vt_ratio=1, noise_vt_ratio=1, noise_cnt=random.randint(0, 5),
                     add_noise=False, add_vt_occ=False, add_rnd_occ=True, add_noise_pts=True, toggledebug=False):
    o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, 'mesh', f + '.ply'))
    o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, 'prim', f + '.ply'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)

    o3dmesh.rotate(rot, center=rot_center)
    o3dmesh_gt.rotate(rot, center=rot_center)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f'{f}_tmp.pcd'), do_render=False, convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))
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
        o3dpcd = du.add_noise_pts_by_vt(o3dpcd, np.asarray(o3dmesh.vertices), noise_cnt=noise_cnt, size=.01)

    o3dpcd = du.resample(o3dpcd, smp_num=2048)
    o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f}.pcd'), o3dpcd)
    o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, method='possion', smp_num=2048)
    vis.destroy_window()

    if toggledebug:
        o3dpcd_org = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))
        o3dpcd_org.paint_uniform_color([0, 0.7, 1])
        o3dpcd.paint_uniform_color(COLOR[0])
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3d.visualization.draw_geometries([o3dmesh])
        o3d.visualization.draw_geometries([o3dpcd_gt])
        o3d.visualization.draw_geometries([o3dpcd_org])
        o3d.visualization.draw_geometries([o3dpcd])
        print(len(o3dpcd.points), len(o3dpcd_gt.points))
    os.remove(os.path.join(path, f'{f}_tmp.pcd'))
    return o3dpcd


if __name__ == '__main__':
    import math
    import basis.o3dhelper as o3dh

    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [.5, .5, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    result_path = f'D:/mvp/data_flat/results_pcn_cd.h5'
    test_path = f'D:/mvp/data_flat/test.h5'
    # pcd_gt, pcd_i, pcd_o = read_pcn_res_pytorch(result_path, test_path, 1, toggledebug=True)
    # pcd_res = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=True)
    #
    # # pcdu.show_pcd(pcd_gt, rgba=(0, 1, 0, 1))
    # pcdu.show_pcd(pcd_i, rgba=(0, 0, 1, 1))
    # pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_i, pcd_res, theta=np.pi / 6, toggledebug=True)
    # coverage = pcdu.cal_coverage(pcd_i, pcd_gt, voxel_size=.001, tor=.002, toggledebug=True)
    # base.run()

    path = 'D:/nbv_mesh/'
    cat = 'bspl'
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        rot = np.eye(3)
        rot_center = [0, 0, 0]
        o3dpcd = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], rot, rot_center, resolusion=(1280, 720),
                                  rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6),
                                  occ_vt_ratio=1, noise_vt_ratio=1, noise_cnt=random.randint(0, 5),
                                  add_noise=False, add_vt_occ=False, add_rnd_occ=True, add_noise_pts=True,
                                  toggledebug=False)
        pcd_res = pcn.inference_sgl(np.asarray(o3dpcd.points), model_name, load_model, toggledebug=False)
        # o3dpcd_res = du.nparray2o3dpcd(pcd_res)
        # o3dpcd.paint_uniform_color(COLOR[0])
        # o3dpcd_res.paint_uniform_color(COLOR[2])
        # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_res])
        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(np.asarray(o3dpcd.points), np.asarray(o3dpcd.points),
                                                         theta=np.pi / 6, toggledebug=True)
        base.run()
