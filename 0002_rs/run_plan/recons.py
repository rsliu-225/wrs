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


def get_center_frame(corners, id, img):
    ps = []
    if id == 1:
        seq = [1, 0, 0, 3]
        relpos = np.asarray([0, -.025, -.05124])
        relrot = np.eye(3)
    elif id == 2:
        seq = [1, 0, 0, 3]
        relpos = np.asarray([0, .025, -.05124])
        relrot = np.eye(3)
    elif id == 3:
        seq = [2, 3, 0, 3]
        relpos = np.asarray([0, -.025, -.03776])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi)
    elif id == 4:
        seq = [2, 3, 0, 3]
        relpos = np.asarray([0, .025, -.03776])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi)
    elif id == 5:
        seq = [0, 3, 3, 2]
        relpos = np.asarray([0, 0, -.072])
        relrot = rm.rotmat_from_axangle((1, 0, 0), -np.pi / 2)
    else:
        seq = [1, 2, 3, 2]
        relpos = np.asarray([0, 0, -.072])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi / 2)
    for i, corner in enumerate(corners[0]):
        p = np.asarray(vu.map_grayp2pcdp(corner, img, pcd))[0]
        if all(np.equal(p, np.asarray([0, 0, 0]))):
            break
        ps.append(p)
        gm.gen_sphere(p, radius=.005, rgba=(1, 0, 0, i * .25)).attach_to(base)
    if len(ps) == 4:
        center = np.mean(np.asarray(ps), axis=0)
        x = rm.unit_vector(ps[seq[0]] - ps[seq[1]])
        y = rm.unit_vector(ps[seq[2]] - ps[seq[3]])
        z = rm.unit_vector(np.cross(x, y))
        rotmat = np.asarray([x, y, z]).T
        marker_mat4 = rm.homomat_from_posrot(center, rotmat)
        relmat4 = rm.homomat_from_posrot(relpos, relrot)
        origin_mat4 = np.dot(marker_mat4, relmat4)

        gm.gen_frame(np.linalg.inv(relmat4)[:3, 3], np.linalg.inv(relmat4)[:3, :3], thickness=.005,
                     length=.05, rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        # gm.gen_frame(origin_mat4[:3, 3], origin_mat4[:3, :3]).attach_to(base)
        # gm.gen_frame(marker_mat4[:3, 3], marker_mat4[:3, :3]).attach_to(base)
        return origin_mat4
    else:
        return None


def crop_maker(img, pcd):
    tgt_id = 1
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    ids = [v[0] for v in ids]
    print(ids)
    if ids is None:
        return None
    if tgt_id not in ids:
        return None
    # pcdu.show_pcd(pcd, rgba=(1, 1, 1, 1))
    gripperframe = get_center_frame(corners[ids.index(tgt_id)], tgt_id, img)
    # gripperframe = get_center_frame(corners[0], ids[0], img)
    if gripperframe is None:
        return None
    # gm.gen_frame(pos=center, rotmat=rotmat).attach_to(base)
    pcd_trans = pcdu.trans_pcd(pcd, np.linalg.inv(gripperframe))
    # pcdu.show_pcd(pcd_trans, rgba=(1, 1, 1, .1))
    gm.gen_frame().attach_to(base)
    return pcdu.crop_pcd(pcd_trans, x_range=(-.05, .215), y_range=(-.2, .2), z_range=(-.2, .2))


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    grayimg_list, depthimg_list, pcd_list = load_frame_seq(folder_name='plate')
    pcd_cropped_list = []
    trans = np.eye(4)

    for i in range(len(grayimg_list)):
        pcd = np.asarray(pcd_list[i]) / 1000
        pcd_cropped = crop_maker(grayimg_list[i], pcd)
        if pcd_cropped is not None:
            # pcd_cropped, _ = get_max_cluster(pcd_cropped, eps=.01, min_samples=50)
            pcd_cropped_list.append(pcd_cropped)

    for i in range(1, len(pcd_cropped_list)):
        print(len(pcd_cropped_list[i - 1]))
        pcdu.show_pcd(pcd_cropped_list[i - 1],
                      rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))
        # _, _, trans_tmp = o3h.registration_ptpt(pcd_cropped_list[i], pcd_cropped_list[i - 1],
        #                                         downsampling_voxelsize=.005,
        #                                         toggledebug=True)
        # trans = trans_tmp.dot(trans)
        # print(trans)
        # pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans),
        #               rgba=(random.choice([.5, 1]), random.choice([0, .5]), random.choice([0, 1])))

    base.run()
