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


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, .2])

    depthimg_list, txtimg_list, pcd_list = load_frame_seq(folder_name='plate')
    pcd_crop_list = []
    trans = np.eye(4)
    for i in range(len(txtimg_list)):
        img = vu.crop(((300, 1000), (600, 1500)), txtimg_list[i])
        pcd = np.asarray(vu.map_gray2pcd(img, pcd_list[i])) / 1000
        pcd, _ = get_max_cluster(pcd, eps=.01, min_samples=50)
        pcd_crop_list.append(pcd)
        # cv2.imshow('crop', img)
        # cv2.waitKey(0)
        if i > 0:
            _, _, trans_tmp = o3h.registration_ptpt(pcd, pcd_crop_list[i - 1], downsampling_voxelsize=.005,
                                                    toggledebug=False)
            trans = trans_tmp.dot(trans)
            pcdu.show_pcd(pcdu.trans_pcd(pcd, trans),
                          rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))

    base.run()
