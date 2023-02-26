import itertools
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
from cv2 import aruco as aruco

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import motionplanner.nbc_solver as nbcs
import utils.pcd_utils as pcdu
import utils.vision_utils as vu
from sklearn.mixture import GaussianMixture


def load_frame_seq(fo=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, fo)
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


def load_frame_seq_withf(fo=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, fo)
    depthimg_list = []
    textureimg_list = []
    pcd_list = []
    fname_list = []
    for f in sorted(os.listdir(path)):
        if f[-3:] != 'pkl' or '_' in f:
            continue
        fname_list.append(f[:-4])
        tmp = pickle.load(open(os.path.join(path, f), 'rb'))
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            textureimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            textureimg_list.append(tmp[1])
        if len(tmp) == 3:
            pcd_list.append(np.asarray(tmp[2]) / 1000)
    return [fname_list, depthimg_list, textureimg_list, pcd_list]


def load_frame(folder_name, f_name, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    tmp = pickle.load(open(os.path.join(path, f_name), 'rb'))
    if tmp[0].shape[-1] == 3:
        depthimg = tmp[1]
        textureimg = tmp[0]
    else:
        depthimg = tmp[0]
        textureimg = tmp[1]
    pcd = tmp[2]
    return depthimg, textureimg, pcd


def load_opti_rigidbody_seq(fo=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, fo)
    opti_list = []
    for f in sorted(os.listdir(path)):
        if 'opti' not in f:
            continue
        opti_list.append(pickle.load(open(os.path.join(path, f), 'rb')).rigidbody_set_dict)
    return opti_list


def load_opti_rigidbody(fo, f_name, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, fo)
    opti_data = pickle.load(open(os.path.join(path, f_name), 'rb'))

    return opti_data.rigidbody_set_dict


def load_opti_markers_seq(fo=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, fo)
    opti_list = []
    for f in sorted(os.listdir(path)):
        if 'opti' not in f:
            continue
        opti_list.append(pickle.load(open(os.path.join(path, f), 'rb')))
    return opti_list


def get_center_frame(corners, id, img, pcd, colors=None, show_frame=False):
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

    ps = _map_corners_in_pcd(img, pcd, corners)
    if ps is None:
        return None
    center = np.mean(np.asarray(ps), axis=0)
    x = rm.unit_vector(ps[seq[0]] - ps[seq[1]])
    y = rm.unit_vector(ps[seq[2]] - ps[seq[3]])
    z = rm.unit_vector(np.cross(x, y))
    rotmat = np.asarray([x, y, z]).T
    marker_mat4 = rm.homomat_from_posrot(center, rotmat)
    relmat4 = rm.homomat_from_posrot(relpos, relrot)
    origin_mat4 = np.dot(marker_mat4, relmat4)
    if show_frame:
        gm.gen_frame(np.linalg.inv(relmat4)[:3, 3], np.linalg.inv(relmat4)[:3, :3], thickness=.005,
                     length=.05, rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        if colors is not None:
            gm.gen_sphere(np.linalg.inv(relmat4)[:3, 3], rgba=colors[id], radius=.007).attach_to(base)
        # gm.gen_frame(origin_mat4[:3, 3], origin_mat4[:3, :3]).attach_to(base)
        # gm.gen_frame(marker_mat4[:3, 3], marker_mat4[:3, :3]).attach_to(base)
    return origin_mat4


def _get_corners(img):
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if ids is None:
        return corners, ids
    ids = [v[0] for v in ids]
    return corners, ids


def _map_corners_in_pcd(img, pcd, corners):
    pts = []
    for i, corner in enumerate(corners[0]):
        p = np.asarray(vu.map_grayp2pcdp(corner, img, pcd))[0]
        if all(np.equal(p, np.asarray([0, 0, 0]))):
            break
        pts.append(p)
    if len(pts) < 4:
        return None
    return pts


def trans_by_armaker(img, pcd, x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155), tgt_id=None,
                     show_frame=False):
    corners, ids = _get_corners(img)
    if ids is None:
        return None, None, None, None
    print('AR marker ids in image:', ids)
    if tgt_id is not None:
        if tgt_id not in ids:
            return None, None, None, None
        gripperframe = get_center_frame(corners[ids.index(tgt_id)], tgt_id, img, pcd, show_frame=show_frame)
    else:
        gripperframe = get_center_frame(corners[0], ids[0], img, pcd, show_frame=show_frame)
    if gripperframe is None:
        return None, None, None, None
    pcd_trans = pcdu.trans_pcd(pcd, np.linalg.inv(gripperframe))
    # pcdu.show_pcd(pcd_trans, rgba=(1, 1, 1, .1))
    if show_frame:
        gm.gen_stick(np.asarray((x_range[0], y_range[0], z_range[0])), np.asarray((x_range[1], y_range[0], z_range[0])),
                     rgba=(1, 0, 0, 1)).attach_to(base)
        gm.gen_stick(np.asarray((x_range[0], y_range[0], z_range[0])), np.asarray((x_range[0], y_range[1], z_range[0])),
                     rgba=(0, 1, 0, 1)).attach_to(base)
        gm.gen_stick(np.asarray((x_range[0], y_range[0], z_range[0])), np.asarray((x_range[0], y_range[0], z_range[1])),
                     rgba=(0, 0, 1, 1)).attach_to(base)
        gm.gen_frame().attach_to(base)
    return ids[0], gripperframe, pcd_trans, \
           pcdu.crop_pcd(pcd_trans, x_range=x_range, y_range=y_range, z_range=z_range)


def opti_in_armarker(corners_lft, corners_rgt):
    center_lft = np.mean(corners_lft, axis=0)
    z_lft = np.cross(corners_lft[0] - corners_lft[1], corners_lft[1] - corners_lft[2])
    center_rgt = np.mean(corners_rgt, axis=0)
    z_rgt = np.cross(corners_rgt[0] - corners_rgt[1], corners_rgt[1] - corners_rgt[2])
    y = center_lft - center_rgt
    z = (z_lft + z_rgt) / 2
    if rm.angle_between_vectors(z, np.asarray((0, 0, 1))) > np.pi / 2:
        z = -z
    x = np.cross(y, z)
    rot = np.asarray([rm.unit_vector(x), rm.unit_vector(y), rm.unit_vector(z)]).T
    pos = (center_lft + center_rgt) / 2
    return rm.homomat_from_posrot(pos, rot)


def reg_armarker(fo, seed=(.116, 0, -.1), center=(.116, 0, -.0155), icp=False, to_zero=True,
                 x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155), toggledebug=False):
    if not os.path.exists(os.path.join(config.ROOT, 'recons_data', fo)):
        os.mkdir(os.path.join(config.ROOT, 'recons_data', fo))
    fnlist, grayimg_list, depthimg_list, pcd_list = load_frame_seq_withf(fo=fo)
    pcd_cropped_list = []
    inx_list = []
    trans = np.eye(4)
    if toggledebug:
        gm.gen_frame(center, np.eye(3)).attach_to(base)
    for i in range(len(grayimg_list)):
        # cv2.imshow('', grayimg_list[i])
        # cv2.waitKey(0)
        pcd = np.asarray(pcd_list[i])
        inx, gripperframe, pcd, pcd_cropped, = \
            trans_by_armaker(grayimg_list[i], pcd, x_range=x_range, y_range=y_range, z_range=z_range,
                             show_frame=toggledebug)
        if pcd_cropped is not None:
            # pcdu.show_pcd(pcd)
            # pcd_cropped, _ = pcdu.get_nearest_cluster(pcd_cropped, seed=seed, eps=.02, min_samples=200)
            # seed = np.mean(pcd_cropped, axis=0)
            # pcd_cropped = pcdu.remove_outliers(pcd_cropped, nb_points=16, toggledebug=True)
            gm.gen_sphere(seed, rgba=(1, 1, 0, 1)).attach_to(base)
            print('Num. of points in cropped pcd:', len(pcd_cropped))
            if len(pcd_cropped) > 0:
                if to_zero:
                    pcd_cropped = pcd_cropped - np.asarray(center)
                o3dpcd = o3dh.nparray2o3dpcd(pcd_cropped)
                o3d.io.write_point_cloud(os.path.join(config.ROOT, 'recons_data', fo, f'{fnlist[i]}' + '.pcd'), o3dpcd)
                # o3d.visualization.draw_geometries([o3dpcd])
                pcd_cropped_list.append(pcd_cropped)
                inx_list.append(inx)

    colors = [(1, 0, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (0, 0, 1, 1)]
    for i in range(1, len(pcd_cropped_list)):
        print(len(pcd_cropped_list[i - 1]))
        if icp:
            _, _, trans_tmp = \
                o3dh.registration_ptpt(pcd_list[i], pcd_list[i - 1], downsampling_voxelsize=.001, toggledebug=False)
            trans = trans_tmp.dot(trans)
            pcd_cropped_list[i] = pcdu.trans_pcd(pcd_cropped_list[i], trans)
            pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans), rgba=colors[inx_list[i] - 1])
        else:
            pcdu.show_pcd(pcd_cropped_list[i - 1], rgba=colors[inx_list[i] - 1])
    # if toggledebug:
    #     o3dpcd_list = []
    #     for i in range(len(pcd_cropped_list)):
    #         o3dpcd = o3dh.nparray2o3dpcd(np.asarray(pcd_cropped_list[i]))
    #         o3dpcd_list.append(o3dpcd)
    #         o3dpcd.paint_uniform_color(list(colors[inx_list[i] - 1][:3]))
    #     o3d.visualization.draw_geometries(o3dpcd_list)
    return pcd_cropped_list


def opti_rigidbody2homomat4(seg):
    if not any([seg.x, seg.z, seg.y]):
        return None
    rot = rm.rotmat_from_axangle((0, 0, 1), seg.qz) \
        .dot(rm.rotmat_from_axangle((0, 1, 0), seg.qy)) \
        .dot(rm.rotmat_from_axangle((1, 0, 0), seg.qx))
    # homomat = rm.homomat_from_posrot([seg.x, seg.z, seg.y], rot)
    homomat = rm.homomat_from_posrot([0, 0, 0], rot)
    return homomat


def opti_markers2homomat4(pts, premat4=None):
    pts_pair = itertools.combinations(range(4), r=2)
    pos = np.mean(pts, axis=0)
    z = np.cross(pts[0] - pts[1], pts[1] - pts[2])
    if rm.angle_between_vectors(z, np.asarray([0, 0, -1])) > np.pi / 2:
        z = -z
    y = None
    for pair in pts_pair:
        gm.gen_stick(pts[pair[0]], pts[pair[1]], thickness=.001).attach_to(base)
        dist = np.linalg.norm(pts[pair[0]] - pts[pair[1]])
        print(dist)
        if abs(dist - .164) <= .008:
            y = pts[pair[0]] - pts[pair[1]]
            if premat4 is not None:
                if rm.angle_between_vectors(y, premat4[:3, 1]) > np.pi / 2:
                    y = -y
    x = np.cross(y, z)
    # if rm.angle_between_vectors(x, np.asarray([1, 0, 0])) > np.pi / 2:
    #     x = -x
    rot = np.asarray([rm.unit_vector(x), rm.unit_vector(y), rm.unit_vector(z)]).T
    gm.gen_frame(pos, rot, thickness=.001).attach_to(base)
    return rm.homomat_from_posrot(pos, rot)


def reg_opti(fo, seed=(.116, 0, -.1), center=(.116, 0, -.0155), icp=False,
             x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155), toggledebug=True):
    if not os.path.exists(os.path.join(config.ROOT, 'recons_data', fo)):
        os.mkdir(os.path.join(config.ROOT, 'recons_data', fo))
    fnlist, grayimg_list, depthimg_list, pcd_list = load_frame_seq_withf(fo=fo)
    optidata_list = load_opti_markers_seq(fo=fo)

    pcd_cropped_list = []
    inx_list = []
    trans = np.eye(4)
    # gm.gen_frame(center, np.eye(3)).attach_to(base)
    homomat4_in_opti_pre = None

    for i in range(len(grayimg_list)):
        pcd = np.asarray(pcd_list[i])
        corners, ids = _get_corners(grayimg_list[i])
        if ids is None:
            continue
        if len(optidata_list[i]) < 4:
            continue
        if 1 not in ids or 2 not in ids:
            continue
        corners_lft = _map_corners_in_pcd(grayimg_list[i], pcd, corners[ids.index(1)])
        corners_rgt = _map_corners_in_pcd(grayimg_list[i], pcd, corners[ids.index(2)])
        if corners_lft is None or corners_rgt is None:
            continue
        homomat4_in_opti = opti_markers2homomat4(optidata_list[i], homomat4_in_opti_pre)
        homomat4_in_phoxi = opti_in_armarker(corners_lft, corners_rgt)

        pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
        gm.gen_frame(homomat4_in_opti[:3, 3], homomat4_in_opti[:3, :3]).attach_to(base)
        gm.gen_frame(homomat4_in_phoxi[:3, 3], homomat4_in_phoxi[:3, :3]).attach_to(base)

        inx, gripperframe, pcd, pcd_cropped, = \
            trans_by_armaker(grayimg_list[i], pcd, x_range=x_range, y_range=y_range, z_range=z_range)
        if pcd_cropped is not None:
            # pcd_cropped, _ = pcdu.get_nearest_cluster(pcd_cropped, seed=seed, eps=.01, min_samples=200)
            pcd_cropped_org = pcdu.trans_pcd(pcd_cropped, gripperframe)
            pcdu.show_pcd(pcd_cropped_org, rgba=(1, 0, 0, 1))

            print('Num. of points in cropped pcd:', len(pcd_cropped))
            tmp_trans = np.eye(4)
            if len(pcd_cropped) > 0:
                seed = np.mean(pcd_cropped, axis=0)
                pcd_cropped = pcd_cropped - np.asarray(center)
                o3dpcd = o3dh.nparray2o3dpcd(pcd_cropped)
                # cl, ind = o3dpcd.remove_radius_outlier(nb_points=16, radius=0.005)
                # display_inlier_outlier(o3dpcd, ind)
                o3d.io.write_point_cloud(os.path.join(config.ROOT, 'recons_data', fo, f'{fnlist[i]}' + '.pcd'),
                                         o3dpcd)
                pcd_cropped_list.append(pcd_cropped)
                pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_org, np.linalg.inv(homomat4_in_phoxi)), rgba=(1, 0, 0, 1))
                pcd_cropped_opti = pcdu.trans_pcd(pcd_cropped_org,
                                                  homomat4_in_opti.dot(np.linalg.inv(homomat4_in_phoxi)))
                pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_opti, homomat4_in_opti),
                              rgba=(0, 1, 1, 1))
                inx_list.append(inx)
        homomat4_in_opti_pre = homomat4_in_opti

    colors = [(1, 0, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (0, 0, 1, 1)]
    # for i in range(1, len(pcd_cropped_list)):
    #     print(len(pcd_cropped_list[i - 1]))
    #     if icp:
    #         _, _, trans_tmp = o3dh.registration_ptpt(pcd_cropped_list[i], pcd_cropped_list[i - 1],
    #                                                  downsampling_voxelsize=.005, toggledebug=True)
    #         trans = trans_tmp.dot(trans)
    #         pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans), rgba=colors[inx_list[i] - 1])
    #     else:
    #         pcdu.show_pcd(pcd_cropped_list[i - 1], rgba=colors[inx_list[i] - 1])
    if toggledebug:
        o3dpcd_list = []
        for i in range(len(pcd_cropped_list)):
            o3dpcd = o3dh.nparray2o3dpcd(np.asarray(pcd_cropped_list[i]))
            o3dpcd_list.append(o3dpcd)
            o3dpcd.paint_uniform_color(list(colors[inx_list[i] - 1][:3]))
        o3d.visualization.draw_geometries(o3dpcd_list)
    return pcd_cropped_list


def extract_roi_by_armarker(textureimg, pcd, seed,
                            x_range=(.06, .215), y_range=(-.15, .15), z_range=(-.2, -.0155), toggledebug=False):
    _, gripperframe, pcd_trans, pcd_roi = \
        trans_by_armaker(textureimg, pcd, x_range=x_range, y_range=y_range, z_range=z_range, show_frame=toggledebug)
    if pcd_roi is None:
        print('No marker detected!')
        return None, None, None

    # pcd_roi, _ = pcdu.get_nearest_cluster(pcd_roi, seed=seed, eps=.02, min_samples=200)
    pcd_roi = pcdu.remove_outliers(pcd_roi, nb_points=50, radius=0.005, toggledebug=False)

    print('Num. of points in extracted pcd:', len(pcd_roi))
    if len(pcd) < 0:
        return None, None, None
    if toggledebug:
        gm.gen_sphere(seed).attach_to(base)
    return pcd_roi, pcd_trans, gripperframe


def cal_nbc(pcd, gripperframe, rbt, seedjntagls, gl_transmat4=np.eye(4),
            theta=np.pi / 3, max_a=np.pi / 18, max_dist=1, toggledebug=True, show_cam=True):
    arrow_len = .04
    cam_pos = np.linalg.inv(gripperframe)[:3, 3]

    pts, nrmls, confs = pcdu.cal_conf(pcd, voxel_size=.005, campos=cam_pos, theta=theta)
    pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs, toggledebug=toggledebug)

    print('Num. of NBV:', len(pts_nbv))

    pcd = pcdu.trans_pcd(pcd, gl_transmat4)
    pts_nbv = pcdu.trans_pcd(pts_nbv, gl_transmat4)
    nrmls_nbv = pcdu.trans_pcd(nrmls_nbv, gl_transmat4)
    cam_pos = pcdu.trans_pcd([cam_pos], gl_transmat4)[0]

    if show_cam:
        pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))
    if toggledebug:
        for i in range(len(pts_nbv)):
            gm.gen_arrow(pts_nbv[i], pts_nbv[i] + np.linalg.norm(nrmls_nbv[i]) * arrow_len,
                         rgba=(confs_nbv[i], 0, 1 - confs_nbv[i], 1)).attach_to(base)
        gm.gen_stick(cam_pos, pts_nbv[0], rgba=(1, 1, 0, 1)).attach_to(base)

    nbc_solver = nbcs.NBCOptimizerVec(rbt, max_a=max_a, max_dist=max_dist)
    jnts, transmat4, _ = nbc_solver.solve(seedjntagls, pts_nbv[0], nrmls_nbv[0], cam_pos)
    pcd_cropped_new = pcdu.trans_pcd(pcd, transmat4)
    n_new = pcdu.trans_pcd(nrmls_nbv, transmat4)[0]
    p_new = pcdu.trans_pcd(pts_nbv, transmat4)[0]
    pcdu.show_pcd(pcd_cropped_new, rgba=(0, 1, 0, 1))

    gm.gen_arrow(p_new, p_new + np.linalg.norm(n_new) * arrow_len, rgba=(0, 1, 0, 1)).attach_to(base)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + np.linalg.norm(nrmls_nbv[0]) * arrow_len, rgba=(0, 0, 1, 1)).attach_to(base)
    gm.gen_stick(cam_pos, p_new, rgba=(1, 1, 0, 1)).attach_to(base)

    return pts_nbv, nrmls_nbv, jnts


def cal_nbc_pcn(pcd, pcd_pcn, gripperframe, rbt, seedjntagls, gl_transmat4=np.eye(4),
                theta=np.pi / 3, max_a=np.pi / 18, max_dist=1, toggledebug=False, show_cam=True):
    arrow_len = .04
    cam_pos = np.linalg.inv(gripperframe)[:3, 3]
    gm.gen_frame().attach_to(base)
    pcd_pcn = pcdu.crop_pcd(pcd_pcn, x_range=(-1, 1), y_range=(-1, 1), z_range=(.0155, 1))

    pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd, pcd_pcn, theta=theta, toggledebug=toggledebug)
    print(confs_nbv)
    print('Num. of NBV:', len(pts_nbv))

    pcd = pcdu.trans_pcd(pcd, gl_transmat4)
    pts_nbv = pcdu.trans_pcd(pts_nbv, gl_transmat4)
    nrmls_nbv = pcdu.trans_pcd(nrmls_nbv, gl_transmat4)
    cam_pos = pcdu.trans_pcd([cam_pos], gl_transmat4)[0]

    if show_cam:
        pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))
    if toggledebug:
        for i in range(len(pts_nbv)):
            gm.gen_arrow(pts_nbv[i], pts_nbv[i] + np.linalg.norm(nrmls_nbv[i]) * arrow_len,
                         rgba=(confs_nbv[i], 0, 1 - confs_nbv[i], 1)).attach_to(base)
        gm.gen_stick(cam_pos, pts_nbv[0], rgba=(1, 1, 0, 1)).attach_to(base)
    # base.run()

    nbc_solver = nbcs.NBCOptimizerVec(rbt, max_a=max_a, max_dist=max_dist)
    jnts, transmat4, _ = nbc_solver.solve(seedjntagls, pts_nbv[0], nrmls_nbv[0], cam_pos)
    pcd_cropped_new = pcdu.trans_pcd(pcd, transmat4)
    n_new = pcdu.trans_pcd([nrmls_nbv[0]], transmat4)[0]
    p_new = pcdu.trans_pcd([pts_nbv[0]], transmat4)[0]
    pcdu.show_pcd(pcd_cropped_new, rgba=(0, 1, 0, 1))

    gm.gen_arrow(p_new, p_new + np.linalg.norm(n_new) * arrow_len, rgba=(0, 1, 0, 1)).attach_to(base)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + np.linalg.norm(nrmls_nbv[0]) * arrow_len, rgba=(0, 0, 1, 1)).attach_to(base)
    gm.gen_stick(cam_pos, p_new, rgba=(1, 1, 0, 1)).attach_to(base)

    return pts_nbv, nrmls_nbv, jnts


def sort_kpts(kpts, seed):
    sort_ids = []
    while len(sort_ids) < len(kpts):
        dist_list = np.linalg.norm(kpts - seed, axis=1)
        sort_ids_tmp = np.argsort(dist_list)
        for i in sort_ids_tmp:
            if i not in sort_ids:
                sort_ids.append(i)
                break
        seed = kpts[sort_ids[-1]]
    return kpts[sort_ids]


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    print(len(objpcd))
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    kpts = sort_kpts(gmix.means_, seed=np.asarray([0, 0, 0]))

    if show:
        for i, p in enumerate(kpts[1:]):
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)

    kdt, _ = pcdu.get_kdt(objpcd)
    kpts_rotseq = []
    for i, p in enumerate(kpts[:-1]):
        knn = pcdu.get_knn(kpts[i], kdt, k=int(len(objpcd) / n_components))
        pcv, pcaxmat = rm.compute_pca(knn)
        y_v = kpts[i + 1] - kpts[i]
        x_v = pcaxmat[:, np.argmin(pcv)]
        if len(kpts_rotseq) != 0:
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 0], x_v) > np.pi / 2:
                x_v = -x_v
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 1], y_v) > np.pi / 2:
                y_v = -y_v
        z_v = np.cross(x_v, y_v)

        rot = np.asarray([rm.unit_vector(x_v), rm.unit_vector(y_v), rm.unit_vector(z_v)]).T
        kpts_rotseq.append(rot)
    kpts_rotseq.append(kpts_rotseq[-1])

    return kpts, np.asarray(kpts_rotseq)
