import copy
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
import modeling.collision_model as cm
import utils.pcd_utils as pcdu
import utils.vision_utils as vu
import motionplanner.nbc_solver as nbcs


def load_frame_seq(folder_name=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
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


def load_frame_seq_withf(folder_name=None, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    depthimg_list = []
    textureimg_list = []
    pcd_list = []
    fname_list = []
    for f in sorted(os.listdir(path)):
        if f[-3:] != 'pkl':
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
            pcd_list.append(tmp[2])
    return [fname_list, depthimg_list, textureimg_list, pcd_list]


def load_frame(folder_name, f_name, root_path=os.path.join(config.ROOT, 'img/phoxi/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    tmp = pickle.load(open(os.path.join(path, f_name), 'rb'))
    print(tmp[0].shape, tmp[1].shape)
    if tmp[0].shape[-1] == 3:
        depthimg = tmp[1]
        textureimg = tmp[0]
    else:
        depthimg = tmp[0]
        textureimg = tmp[1]
    pcd = tmp[2]
    return depthimg, textureimg, pcd


def get_center_frame(corners, id, img, pcd, colors=None, show_frame=True):
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
        # gm.gen_sphere(p, radius=.005, rgba=(1, 0, 0, i * .25)).attach_to(base)
    if len(ps) == 4:
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
    else:
        return None


def crop_maker(img, pcd, x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155), tgt_id=None, show_frame=True):
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if ids is None:
        return None, None
    ids = [v[0] for v in ids]
    print(ids)
    if tgt_id is not None:
        if tgt_id not in ids:
            return None, None
        gripperframe = get_center_frame(corners[ids.index(tgt_id)], tgt_id, img, pcd)
    else:
        gripperframe = get_center_frame(corners[0], ids[0], img, pcd)
    if gripperframe is None:
        return None, None
    pcd_trans = pcdu.trans_pcd(pcd, np.linalg.inv(gripperframe))
    pcdu.show_pcd(pcd_trans, rgba=(1, 1, 1, .1))
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


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def reg_plate(folder_name, seed=(.116, 0, -.1), center=(.116, 0, -.0155), icp=False,
              x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155), toggledebug=True):
    if not os.path.exists(os.path.join(config.ROOT, 'recons_data', folder_name)):
        os.mkdir(os.path.join(config.ROOT, 'recons_data', folder_name))
    fnlist, grayimg_list, depthimg_list, pcd_list = load_frame_seq_withf(folder_name=folder_name)
    pcd_cropped_list = []
    inx_list = []
    trans = np.eye(4)
    # gm.gen_frame(center, np.eye(3)).attach_to(base)
    for i in range(len(grayimg_list)):
        pcd = np.asarray(pcd_list[i]) / 1000
        inx, gripperframe, pcd, pcd_cropped, = \
            crop_maker(grayimg_list[i], pcd, x_range=x_range, y_range=y_range, z_range=z_range)
        if pcd_cropped is not None:
            pcd_cropped, _ = pcdu.get_nearest_cluster(pcd_cropped, seed=seed, eps=.01, min_samples=200)
            seed = np.mean(pcd_cropped, axis=0)
            print('Num. of points in cropped pcd:', len(pcd_cropped))
            if len(pcd_cropped) > 0:
                pcd_cropped = pcd_cropped - np.asarray(center)
                o3dpcd = o3dh.nparray2o3dpcd(pcd_cropped)
                # cl, ind = o3dpcd.remove_radius_outlier(nb_points=16, radius=0.005)
                # display_inlier_outlier(o3dpcd, ind)
                o3d.io.write_point_cloud(os.path.join(config.ROOT, 'recons_data', folder_name, f'{fnlist[i]}' + '.pcd'),
                                         o3dpcd)
                pcd_cropped_list.append(pcd_cropped)
                inx_list.append(inx)

    colors = [(1, 0, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1), (0, 0, 1, 1)]
    for i in range(1, len(pcd_cropped_list)):
        print(len(pcd_cropped_list[i - 1]))
        if icp:
            _, _, trans_tmp = o3dh.registration_ptpt(pcd_cropped_list[i], pcd_cropped_list[i - 1],
                                                     downsampling_voxelsize=.005,
                                                     toggledebug=False)
            trans = trans_tmp.dot(trans)
            pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans), rgba=colors[inx_list[i] - 1])
        else:
            pcdu.show_pcd(pcd_cropped_list[i - 1], rgba=colors[inx_list[i] - 1])
    if toggledebug:
        o3dpcd_list = []
        for i in range(len(pcd_cropped_list)):
            o3dpcd = o3dh.nparray2o3dpcd(np.asarray(pcd_cropped_list[i]))
            o3dpcd_list.append(o3dpcd)
            o3dpcd.paint_uniform_color(list(colors[inx_list[i] - 1][:3]))
        o3d.visualization.draw_geometries(o3dpcd_list)
    return pcd_cropped_list


def cal_nbc(textureimg, pcd, rbt, seedjntagls, seed=(.116, 0, -.1), gl_transmat4=np.eye(4), theta=np.pi / 3,
            x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155),
            toggledebug=True, show_cam=True):
    pcd = np.asarray(pcd) / 1000
    inx, gripperframe, pcd, pcd_cropped = \
        crop_maker(textureimg, pcd, x_range=x_range, y_range=y_range, z_range=z_range)
    cam_pos = np.linalg.inv(gripperframe)[:3, 3]

    if pcd_cropped is None:
        print('No marker detected!')
        return None, None
    pcd_cropped, _ = pcdu.get_nearest_cluster(pcd_cropped, seed=seed, eps=.01, min_samples=200)
    gm.gen_sphere(seed).attach_to(base)
    print('Num. of points in cropped pcd:', len(pcd_cropped))

    if len(pcd_cropped) < 0:
        return None, None

    pts, nrmls, confs = \
        pcdu.cal_conf(pcd_cropped, voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=theta)
    nbv_pts, nbv_nrmls, nbv_confs = \
        pcdu.cal_nbv(pts, nrmls, confs, cam_pos=np.linalg.inv(gripperframe)[:3, 3])
    print('Num. of NBV:', len(nbv_pts))

    pcd = pcdu.trans_pcd(pcd, gl_transmat4)
    pcd_cropped = pcdu.trans_pcd(pcd_cropped, gl_transmat4)
    nbv_pts = pcdu.trans_pcd(nbv_pts, gl_transmat4)
    nbv_nrmls = pcdu.trans_pcd(nbv_nrmls, gl_transmat4)
    cam_pos = pcdu.trans_pcd([cam_pos], gl_transmat4)[0]

    if show_cam:
        pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=np.dot(rm.rotmat_from_axangle((0, 0, 1), np.pi / 2),
                                                                 rm.rotmat_from_axangle((1, 0, 0), -np.pi / 3))))
    if toggledebug:
        pcdu.show_pcd(pcd, rgba=(1, 0, 0, 1))
        for i in range(len(nbv_pts)):
            p = np.asarray(nbv_pts[i])
            n = np.asarray(nbv_nrmls[i])
            gm.gen_arrow(p, p + n * .05, thickness=.002,
                         rgba=(nbv_confs[i], 0, 1 - nbv_confs[i], 1)).attach_to(base)
            gm.gen_stick(cam_pos, p, rgba=(1, 1, 0, 1)).attach_to(base)

    nbc_solver = nbcs.NBCOptimizer(rbt, max_a=np.pi / 18)
    jnts, transmat4, _ = nbc_solver.solve(seedjntagls, nbv_pts, nbv_nrmls, cam_pos)
    pcd_cropped_new = pcdu.trans_pcd(pcd_cropped, transmat4)
    n_new = pcdu.trans_pcd([nbv_nrmls[0]], transmat4)[0]
    p_new = pcdu.trans_pcd([nbv_pts[0]], transmat4)[0]
    pcdu.show_pcd(pcd_cropped_new, rgba=(0, 1, 0, 1))
    gm.gen_arrow(p_new, p_new + n_new * .05, rgba=(0, 1, 0, 1)).attach_to(base)
    gm.gen_arrow(nbv_pts[0], nbv_pts[0] + nbv_nrmls[0] * .05, rgba=(1, 0, 0, 1)).attach_to(base)
    gm.gen_stick(cam_pos, p_new, rgba=(0, 1, 1, 1)).attach_to(base)

    return pcd_cropped, nbv_pts, nbv_nrmls, jnts
