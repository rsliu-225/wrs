import copy
import os
import pickle
import random

import cv2
import numpy as np
import open3d as o3d
from cv2 import aruco as aruco

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import utils.recons_utils as rcu
import visualization.panda.world as wd
import localenv.envloader as el
import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
from sklearn.mixture import GaussianMixture


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    kpts = gmix.means_
    if show:
        for p in kpts:
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)

    kdt, _ = pcdu.get_kdt(objpcd)
    kpts_rotseq = []
    pre_n = None
    for i, p in enumerate(kpts[1:]):
        knn = pcdu.get_knn(p, kdt, k=50)
        pcv, pcaxmat = rm.compute_pca(knn)
        inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
        y_v = kpts[i - 1] - kpts[i]
        z_v = pcaxmat[:, inx[0]]
        if pre_n is not None:
            if rm.angle_between_vectors(pre_n, z_v):
                z_v = -z_v
        x_v = np.cross(y_v, z_v)
        pcaxmat = np.asarray([rm.unit_vector(x_v), rm.unit_vector(y_v), rm.unit_vector(z_v)]).T
        kpts_rotseq.append(pcaxmat)
        pre_n = z_v
    kpts_rotseq = [kpts_rotseq[0]] + kpts_rotseq

    return kpts, np.asarray(kpts_rotseq)


if __name__ == '__main__':
    import bendplanner.bend_utils as bu

    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    fo = 'nbc_pcn/plate_a_cubic'
    # fo = 'nbc/plate_a_cubic'
    # fo = 'opti/plate_a_quadratic'
    # fo = 'seq/plate_a_quadratic'
    gm.gen_frame().attach_to(base)

    width = .005
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, .0155)

    x_range = (.065, .215)
    y_range = (-.15, .15)
    z_range = (.0165, .2)
    # z_range = (-.2, -.0155)
    # gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=True, icp=False)
    pts = []
    for pcd in pcd_cropped_list:
        pts.extend(pcd)

    kpts, kpts_rotseq = get_kpts_gmm(pts, rgba=(1, 1, 0, 1))
    sort_ids = []
    seed = np.asarray([0, 0, 0])
    while len(sort_ids) < len(kpts):
        dist_list = np.linalg.norm(kpts - seed, axis=1)
        sort_ids_tmp = np.argsort(dist_list)
        min_id = list(dist_list).index(min(dist_list))
        for i in sort_ids_tmp:
            if i not in sort_ids:
                sort_ids.append(i)
                break
        seed = kpts[i]
    kpts = kpts[sort_ids]
    kpts_rotseq = kpts_rotseq[sort_ids]
    # kpts = bu.linear_inp3d_by_step(kpts)

    # kpts, kpts_rotseq = bu.inp_rotp_by_step(kpts, kpts_rotseq)

    for i, rot in enumerate(kpts_rotseq):
        gm.gen_frame(kpts[i], kpts_rotseq[i], thickness=.001, length=.05).attach_to(base)
    objcm = bu.gen_swap(kpts, kpts_rotseq, cross_sec)
    objcm.attach_to(base)

    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1]]
    for i, pcd in enumerate(pcd_cropped_list):
        pcdu.show_pcd(pcd, rgba=rgba_list[i])
    # pcdu.show_pcd(pts, rgba_list[0])
    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)

    base.run()
