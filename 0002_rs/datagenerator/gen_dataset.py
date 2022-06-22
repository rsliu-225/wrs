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


def gen_sgl_curve(pseq, step=.001, toggledebug=False):
    length = np.sum(np.linalg.norm(np.diff(np.asarray(pseq), axis=0), axis=1))
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='cubic')
    inp_z = interpolate.interp1d(pseq[:, 0], pseq[:, 2], kind='cubic')
    x = np.linspace(0, pseq[-1][0], int(length / step))
    y = inp(x)
    z = inp_z(x)
    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
        ax.scatter3D(x, y, z, color='green')
        plt.show()

    return np.asarray(list(zip(x, y, z)))


def gen_swap(pseq, rotseq, cross_sec, toggledebug=False):
    vertices = []
    faces = []
    cross_sec.append(cross_sec[0])
    for i, p in enumerate(pseq):
        for n in cross_sec:
            vertices.append(p + rotseq[i][:, 0] * n[0] + rotseq[i][:, 2] * n[1])
    for i in range(len(cross_sec) - 3):
        faces.append([0, i + 1, i + 2])
    for i in range((len(cross_sec)) * (len(pseq) - 1)):
        if i % (len(cross_sec)) == 0:
            for v in range(i, i + len(cross_sec) - 1):
                faces.extend([[v, v + len(cross_sec), v + len(cross_sec) + 1],
                              [v, v + len(cross_sec) + 1, v + 1]])
    for i in range(len(cross_sec) - 3):
        faces.append([len(vertices) - 1, len(vertices) - 2 - i, len(vertices) - 3 - i])
    if toggledebug:
        utl.show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
        utl.show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj')


if __name__ == '__main__':
    import visualization.panda.world as wd

    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    width = .005
    thickness = .0015
    folder_name = 'tst'

    pseq = gen_sgl_curve(pseq=np.asarray([[0, 0, 0], [.018, .03, .02], [.06, .06, 0], [.12, 0, 0]]))
    # pseq = gen_sgl_curve(pseq=np.asarray([[0, 0, 0], [.018, .03, 0], [.06, .06, 0], [.12, 0, 0]]))
    rotseq = utl.get_rotseq_by_pseq(pseq)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    objcm = gen_swap(pseq, rotseq, cross_sec)
    objcm.attach_to(base)

    '''
    gen data
    '''
    cnt = 0
    obj_id = 0
    rot_center = (0, 0, 0)
    homomat4_dict = dict()
    homomat4_dict[str(obj_id)] = {}

    icomats = rm.gen_icorotmats(rotation_interval=np.radians(90))
    for i, mats in enumerate(icomats):
        for j, rot in enumerate(mats):
            utl.get_objpcd_partial_o3d(objcm, rot, rot_center, path=folder_name,
                                       f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
                                       add_noise=False, add_occ=True, toggledebug=False)
            homomat4_dict[str(obj_id)][str(cnt).zfill(3)] = rm.homomat_from_posrot(rot_center, rot)
            cnt += 1
            pickle.dump(homomat4_dict, open(f'{folder_name}/homomat4_dict.pkl', 'wb'))

    # '''
    # show data
    # '''
    # for f in sorted(os.listdir(folder_name)):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{folder_name}/{f}")
    #         gm.gen_pointcloud(o3dpcd.points).attach_to(base)
    # base.run()

    # '''
    # show key point
    # '''
    # objpcd_narry = get_objpcd_full_sample(objcm)
    # gm.gen_pointcloud(objpcd_narry, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # get_kpts_gmm(objpcd_narry, rgba=(1, 0, 0, 1))
    #
    # o3dmesh = o3d.io.read_triangle_mesh(f"tst/0_0_0.ply")
    # o3dpcd = o3d.io.read_point_cloud(f"tst/0_0_0.pcd")
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 0, .5]]).attach_to(base)
    # get_kpts_gmm(o3dpcd.points, rgba=(0, 1, 0, 1))
    #
    # base.run()

    # o3dpcd = point_clouo3d.io.read_d(f"tst/0_0_0.pcd")
    # o3dpcd_occ = add_random_occ(o3dpcd)
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # gm.gen_pointcloud(o3dpcd_occ.points, pntsize=5, rgbas=[[1, 0, 0, .5]]).attach_to(base)
    # base.run()
