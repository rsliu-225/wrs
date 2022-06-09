import pickle

import numpy as np

import config
import os
import basis.o3dhelper as o3h
import utils.vision_utils as vu
import cv2
import utils.pcd_utils as pcdu
from sklearn.cluster import DBSCAN
import random
from cv2 import aruco as aruco
import modeling.geometric_model as gm
import basis.robot_math as rm


def load_frame_seq(folder_name=None, root_path=os.path.join(config.ROOT, 'img/phoxi/seq/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    depthimg_list = []
    rgbimg_list = []
    pcd_list = []
    for f in sorted(os.listdir(path)):
        print(f)
        if f[-3:] != 'pkl':
            continue
        tmp = pickle.load(open(os.path.join(path, f), 'rb'))
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            rgbimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            rgbimg_list.append(tmp[1])
        if len(tmp) == 3:
            pcd_list.append(tmp[2])
    return [depthimg_list, rgbimg_list, pcd_list]


def get_max_cluster(pts, eps=.003, min_samples=2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    res = []
    mask = []
    max_len = 0

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            if len(cluster) > max_len:
                max_len = len(cluster)
                res = cluster
                mask = [class_member_mask & core_samples_mask]

    return np.asarray(res), mask


def crop_maker(img, pcd):
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    print(ids)
    if ids is None:
        return None
    ps = []
    for corner in corners[0][0]:
        p = np.asarray(vu.map_grayp2pcdp(corner, img, pcd))
        ps.append(p[0])
        # gm.gen_sphere(p[0], radius=.005).attach_to(base)
    center = np.mean(np.asarray(ps), axis=0)
    x = rm.unit_vector(ps[0] - ps[1])
    y = rm.unit_vector(ps[2] - ps[1])
    z = rm.unit_vector(np.cross(x, y))
    rotmat = np.asarray([x, y, z]).T
    pcd_trans = pcdu.trans_pcd(pcd, np.linalg.inv(rm.homomat_from_posrot(center, rotmat)))
    # gm.gen_frame(pos=center, rotmat=rotmat).attach_to(base)
    # pcdu.show_pcd(pcd_trans, rgba=(1, 1, 0, 1))
    gm.gen_frame().attach_to(base)
    return pcdu.crop_pcd(pcd_trans, x_range=(-.4, .2), y_range=(-.08, .5), z_range=(-.1, .2))


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])
    grayimg_list, depthimg_list, pcd_list = load_frame_seq(folder_name='plate_as')
    pcd_cropped_list = []
    trans = np.eye(4)
    for i in range(len(grayimg_list)):
        # img = vu.crop(((200, 900), (800, 1500)), depthimg_list[i])
        # pcd_cropped = np.asarray(vu.map_gray2pcd(img, pcd_list[i])) / 1000
        # pcd_cropped, _ = get_max_cluster(pcd_cropped, eps=.01, min_samples=50)
        # pcd_cropped_list.append(pcd_cropped)
        # cv2.imshow('crop', img)
        # cv2.waitKey(0)
        pcd = np.asarray(pcd_list[i]) / 1000
        pcd_cropped = crop_maker(grayimg_list[i], pcd)
        if pcd_cropped is not None:
            # pcd_cropped, _ = get_max_cluster(pcd_cropped, eps=.01, min_samples=50)
            pcd_cropped_list.append(pcd_cropped)

    for i in range(1, len(pcd_cropped_list)):
        # plate_pcd = pcdu.crop_pcd(pcd_cropped_list[i - 1], x_range=(-1, 1), y_range=(.1, .5), z_range=(-.02, 1))
        print(len(pcd_cropped_list[i - 1]))
        # pcdu.show_pcd(plate_pcd,
        #               rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))
        pcdu.show_pcd(pcd_cropped_list[i - 1],
                      rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))
        # _, _, trans_tmp = o3h.registration_ptpt(pcd_cropped_list[i], pcd_cropped_list[i - 1],
        #                                         downsampling_voxelsize=.005,
        #                                         toggledebug=False)
        # trans = trans_tmp.dot(trans)
        # print(trans)
        # pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans),
        #               rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))

    base.run()
