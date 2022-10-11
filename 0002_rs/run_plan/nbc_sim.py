import os
import h5py
import numpy as np
import open3d as o3d

import basis.robot_math as rm
import config
# import localenv.envloader as el
import modeling.geometric_model as gm
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd
import basis.o3dhelper as o3dh


def show_pcn_res_pytorch(result_path, test_path):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    for i in range(10, len(test_f['complete_pcds'])):
        o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][i]))
        o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][i]))
        o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][i]))
        o3dpcd_gt.paint_uniform_color([0, 1, 0])
        o3dpcd_i.paint_uniform_color([0, 0, 1])
        o3dpcd_o.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
        o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])


def read_pcn_res_pytorch(result_path, test_path, id):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][id]))
    o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][id]))
    o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][id]))
    o3dpcd_gt.paint_uniform_color([0, 1, 0])
    o3dpcd_i.paint_uniform_color([0, 0, 1])
    o3dpcd_o.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
    o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])
    return np.asarray(test_f['complete_pcds'][id]), \
           np.asarray(test_f['incomplete_pcds'][id]), \
           np.asarray(res_f['results'][id])


if __name__ == '__main__':
    import math
    cam_pos = [.5, .5, .5]
    fo = 'D:/liu/MVP_Benchmark/completion'

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    # rbt = el.loadXarm(showrbt=False)
    # m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    #
    # seedjntagls = m_planner.get_armjnts()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    result_path = f'{fo}/log/pcn_cd_cubic/results.h5'
    test_path = f'{fo}/data_2048/test.h5'
    pcd_gt, pcd_i, pcd_o = read_pcn_res_pytorch(result_path, test_path, 10)

    pcdu.show_pcd(pcd_gt, rgba=(1, 0, 0, 1))
    pcdu.show_pcd(pcd_i, rgba=(1, 0, 0, 1))

    # pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_i, pcd_o, theta=np.pi / 6, toggledebug=True)
    base.run()
