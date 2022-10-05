import copy
import math
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import interpolate
from sklearn.neighbors import KDTree

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import modeling.geometric_model as gm

import datagenerator.utils as utl
import visualization.panda.world as wd

if __name__ == '__main__':
    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    width = .005
    thickness = .0015
    path = './tst'
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    pseq = utl.cubic_inp(pseq=np.asarray([[0, 0, 0], [.018, .03, .02], [.06, .06, 0], [.12, 0, 0]]))
    pseq = utl.uni_length(pseq, goal_len=.2)
    rotseq = utl.get_rotseq_by_pseq(pseq)

    objcm = utl.gen_swap(pseq, rotseq, cross_sec)
    # objcm.set_rgba((1, 1, 0, 1))
    # objcm.attach_to(base)
    # base.run()

    '''
    gen data
    '''
    cnt = 0
    obj_id = 0
    rot_center = (0, 0, 0)
    homomat4_dict = dict()
    homomat4_dict[str(obj_id)] = {}

    # icomats = rm.gen_icorotmats(rotation_interval=np.radians(90))
    # for i, mats in enumerate(icomats):
    #     for j, rot in enumerate(mats):
    #         utl.get_objpcd_partial_o3d(objcm, rot, rot_center, path=path,
    #                                    f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
    #                                    occ_vt_ratio=random.uniform(.5, 1), noise_vt_ration=random.uniform(.5, 1),
    #                                    add_noise=True, add_occ=True, toggledebug=True)
    #         homomat4_dict[str(obj_id)][str(cnt).zfill(3)] = rm.homomat_from_posrot(rot_center, rot)
    #         cnt += 1
    #         pickle.dump(homomat4_dict, open(f'{path}/homomat4_dict.pkl', 'wb'))
    # cammat4_seq = []
    # for i, mats in enumerate(icomats):
    #     for j, rot in enumerate(mats):
    #         cammat4_seq.append(rm.homomat_from_posrot())
    # utl.get_objpcd_partial_o3d_vctrl(objcm, path=path,
    #                                  f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
    #                                  occ_vt_ratio=random.uniform(.5, 1), noise_vt_ration=random.uniform(.5, 1),
    #                                  add_noise=True, add_occ=True, toggledebug=True)
    utl.get_objpcd_partial_o3d(objcm, np.eye(3), rot_center, path=path, resolusion=(550, 550),
                               f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
                               occ_vt_ratio=random.uniform(.5, 1), noise_vt_ration=random.uniform(.5, 1),
                               add_noise=True, add_occ=True, toggledebug=True)

    # '''
    # show data
    # '''
    # for f in sorted(os.listdir(folder_name)):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{folder_name}/{f}")
    #         gm.gen_pointcloud(o3dpcd.points).attach_to(base)
    # base.run()
    #
    # '''
    # show key point
    # '''
    # objpcd_narry = utl.get_objpcd_full_sample(objcm)
    # gm.gen_pointcloud(objpcd_narry, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # utl.get_kpts_gmm(objpcd_narry, rgba=(1, 0, 0, 1))
    #
    # o3dmesh = o3d.io.read_triangle_mesh(f"tst/0_0_0.ply")
    # o3dpcd = o3d.io.read_point_cloud(f"tst/0_0_0.pcd")
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 0, .5]]).attach_to(base)
    # utl.get_kpts_gmm(o3dpcd.points, rgba=(0, 1, 0, 1))
    #
    # base.run()
    #
    # o3dpcd = o3d.io.read_point_cloud(f"tst/0_0_0.pcd")
    # o3dpcd_occ = utl.add_random_occ(o3dpcd)
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # gm.gen_pointcloud(o3dpcd_occ.points, pntsize=5, rgbas=[[1, 0, 0, .5]]).attach_to(base)
    # base.run()
