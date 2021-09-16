import chainer.functions as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sknw
from shapely.geometry import Polygon
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

from localenv import envloader as el
import visualization.panda.world as wd
import utils.pcd_utils as pcdu
import local_vis.realsense.realsense as rs
import utils.vision_utils as vu
import basis.robot_math as rm
import basis.o3dhelper as o3dhelper
from mask_rcnn_seg.inference import MaskRcnnPredictor
import detection_utils as du
import open3d as o3d
import pickle
import config_LfD as config
import os
from tqdm import tqdm
import copy


def remove_by_label(depthimg_list, rgbimg_list, folder_name=None, label=0, toggledebug=False, dilation=False):
    if folder_name is not None:
        if not os.path.exists(f'{config.DATA_PATH}/seg_result/{folder_name}/'):
            os.makedirs(f'{config.DATA_PATH}/seg_result/{folder_name}/')
        if not os.path.exists(f'{config.DATA_PATH}/inf_result/{folder_name}/'):
            os.makedirs(f'{config.DATA_PATH}/inf_result/{folder_name}/')

    bgmask_list = du.flist_remove_bg(depthimg_list, rgbimg_list, threshold=100, toggledebug=False)
    depthimg_list_res, rgbimg_list_res = [], []
    predictor = MaskRcnnPredictor()
    realsense = rs.RealSense()

    for i, im in tqdm(enumerate(rgbimg_list)):
        if i < 86:
            continue
        # pcdu.show_pcd(realsense.depth2pcd(depthimg_list[i]), rgba=(1, 1, 1, .1))
        # base.run()

        print(f'---------------image {i}---------------')
        hnd_prediction = predictor.predict(im, label)
        hnd_mask_list = hnd_prediction.get('pred_masks').numpy()
        if len(hnd_mask_list) == 0:
            print('no target mask!')
            continue

        visualized_pred = predictor.visualize_prediction(im, hnd_prediction)
        depthimg = depthimg_list[i]
        # print(depthimg)

        hnd_mask = np.zeros(depthimg.shape)
        for m in hnd_mask_list:
            if dilation:
                kernel = np.ones((3, 3), np.uint8)
                if toggledebug:
                    cv2.imshow('mask_org', m.astype(np.uint8) * 255)
                    cv2.waitKey(0)
                m = cv2.dilate(np.float32(m), kernel, iterations=5)
            hnd_mask = np.logical_or(hnd_mask, m)

        mask = np.logical_and(~bgmask_list[i], ~hnd_mask)
        depthimg[~mask] = 0
        rgbimg = rgbimg_list[i] * np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        if toggledebug:
            cv2.imshow('prediction', np.hstack((rgbimg, visualized_pred)))
            cv2.imshow('mask', np.hstack(((~bgmask_list[i]).astype(np.uint8) * 255,
                                          (~hnd_mask).astype(np.uint8) * 255,
                                          mask.astype(np.uint8) * 255)))
            cv2.waitKey(0)
            pcd_o3d = realsense.rgbd2pcd(depthimg, rgbimg, toggledebug=False)
            pcd, colors = np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.colors)
            pcdu.show_pcd_withrgb(pcd, rgbas=np.hstack((colors,
                                                        np.repeat(1, [len(pcd)]).reshape((len(pcd), 1)))))
            base.run()
        depthimg_list_res.append(depthimg)
        rgbimg_list_res.append(rgbimg)
        if folder_name is not None:
            pickle.dump([depthimg, rgbimg],
                        open(f'{config.DATA_PATH}/seg_result/{folder_name}/{str(i).zfill(4)}.pkl', 'wb'))
            cv2.imwrite(f'{config.DATA_PATH}/inf_result/{folder_name}/{str(i).zfill(4)}.png', visualized_pred)
    return depthimg_list_res, rgbimg_list_res


def extract_conponent_o3d(depthimg_list, rgbimg_list, folder_name=None, seed=None, toggledebug=False):
    if folder_name is not None:
        if not os.path.exists(f'{config.DATA_PATH}/seg_pcd/{folder_name}/'):
            os.makedirs(f'{config.DATA_PATH}/seg_pcd/{folder_name}/')
    cluster_o3d_list = []
    pcd_list = []
    realsense = rs.RealSense()
    for i in range(len(depthimg_list)):
        pcd_o3d = realsense.rgbd2pcd(depthimg_list[i], rgbimg_list[i])
        pcd_o3d = o3dhelper.removeoutlier_o3d(pcd_o3d)
        if seed is None:
            du.cluster_dbscan(pcd_o3d.points)
            cluster_pts, mask = du.get_max_cluster(pcd_o3d.points)
        else:
            cluster_pts, mask = du.get_closest_cluster(pcd_o3d.points, seed=seed, min_pts=100, eps=.005)
        if len(mask) == 0:
            print(f'No cluster found around {seed}!')
            continue
        pcd_list.append(pcd_o3d.points)
        cluster_o3d = o3d.geometry.PointCloud()
        cluster_o3d.points = o3d.utility.Vector3dVector(cluster_pts)
        cluster_o3d.colors = o3d.utility.Vector3dVector(np.array(pcd_o3d.colors)[tuple(mask)])
        cluster_o3d_list.append(cluster_o3d)
        seed = np.mean(cluster_pts, axis=0)
        print(seed, len(cluster_pts))
        if folder_name is not None:
            o3d.io.write_point_cloud(f'{config.DATA_PATH}/seg_pcd/{folder_name}/{str(i).zfill(4)}.pcd', cluster_o3d)

    if toggledebug:
        pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in cluster_o3d_list],
                                 rgbasseq=[o3d.colors for o3d in cluster_o3d_list], time_sleep=.5)
        pcdu.show_pcdseq(pcd_list, rgba=(1, 1, 1, .1), time_sleep=.5)
        base.run()
    return cluster_o3d_list


def registration_rgbd(source, target, current_transformation=np.identity(4)):
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    result_icp = None
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        try:
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=iter))
            current_transformation = result_icp.transformation
            print(result_icp)
        except:
            continue
    if result_icp is None:
        return None, None, None
    return [result_icp.inlier_rmse, result_icp.fitness, result_icp.transformation]


def merge_o3dpcd_seq_acc(o3dpcd_seq):
    res_o3dpcd = o3d.geometry.PointCloud()
    pts = []
    colors = []
    for i in range(len(o3dpcd_seq) - 1):
        source = o3dpcd_seq[i + 1]
        if len(pts) != 0:
            target_narray = np.asarray(pts)
            target = res_o3dpcd
        else:
            target_narray = np.asarray(o3dpcd_seq[i].points)
            target = o3dpcd_seq[i]

        print(len(np.asarray(source.points)), len(np.asarray(target.points)))
        # rmse, fitness, trans = registration_rgbd(source, target, current_transformation = np.identity(4))
        rmse, fitness, trans = o3dhelper.registration_ptpt(np.asarray(source.points), target_narray)

        if trans is None:
            continue
        print(i, trans)
        source_trans = source.transform(trans)
        pts.extend(np.asarray(source_trans.points))
        colors.extend(np.asarray(source_trans.colors))
        res_o3dpcd = o3d.geometry.PointCloud()
        res_o3dpcd.points = o3d.utility.Vector3dVector(np.asarray(pts))
        res_o3dpcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return res_o3dpcd


def merge_o3dpcd_seq(o3dpcd_seq, win_size=4):
    res_o3dpcd_seq = [o3dpcd_seq[0]]
    pts_list = [np.asarray(o3dpcd_seq[0].points)]
    colors_list = [np.asarray(o3dpcd_seq[0].colors)]
    trans_list = [np.eye(4)]
    rmse_list = [0]
    fitness_list = [1]
    for i in range(len(o3dpcd_seq) - 1):
        back_inx = int(max([0, i - win_size]))
        target_list = res_o3dpcd_seq[back_inx:i][::-1]
        inithomomat = trans_list[-1]
        source = o3dpcd_seq[i + 1]

        min_rmse = np.inf
        max_fitness = -1
        trans = None

        for target in target_list:
            print(len(np.asarray(source.points)), len(np.asarray(target.points)))
            # rmse, fitness, tmp_trans = registration_rgbd(source, target, current_transformation=inithomomat)
            # rmse, fitness, tmp_trans = \
            #     o3dhelper.registration_ptpt(np.asarray(source.points), np.asarray(target.points),
            #                                 downsampling_voxelsize=.002, toggledebug=False)
            rmse, fitness, tmp_trans = \
                o3dhelper.registration_icp_ptpt(np.asarray(source.points), np.asarray(target.points),
                                                maxcorrdist=.002, inithomomat=inithomomat, toggledebug=False)

            if tmp_trans is None:
                continue
            print(i, rmse, fitness)
            # if rmse < min_rmse:
            if fitness > max_fitness:
                min_rmse = rmse
                max_fitness = fitness
                trans = tmp_trans

            # pcdu.show_pcd(np.asarray(target.points), rgba=(1, 0, 0, 1))
            # pcdu.show_pcd(np.asarray(source.points), rgba=(0, 1, 0, 1))
            # source_trans_narray = pcdu.trans_pcd(np.asarray(source.points), trans)
            # pcdu.show_pcd(source_trans_narray, rgba=(0, 0, 1, 1))
            # pcdu.show_pcd(np.asarray(source_trans.points), rgba=(0, 0, 1, 1))
            # base.run()
        if trans is None or max_fitness < .9:
            continue
        rmse_list.append(min_rmse)
        fitness_list.append(max_fitness)
        trans_list.append(trans)
        source_trans = source.transform(trans)
        pts_list.append(np.asarray(source_trans.points))
        colors_list.append(np.asarray(source_trans.colors))
        res_o3dpcd = o3d.geometry.PointCloud()
        res_o3dpcd.points = o3d.utility.Vector3dVector(np.asarray(pts_list[-1]))
        res_o3dpcd.colors = o3d.utility.Vector3dVector(np.asarray(colors_list[-1]))
        res_o3dpcd_seq.append(res_o3dpcd)
    # plt.plot(rmse_list)
    # plt.plot(fitness_list)
    # plt.show()
    return res_o3dpcd_seq


if __name__ == '__main__':
    folder_name = 'bunny'
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])

    # depthimg_list, rgbimg_list = du.load_frame_seq(folder_name)
    # du.show_rgbdseq(depthimg_list, rgbimg_list)
    # depthimg_list_res, rgbimg_list_res = remove_by_label(depthimg_list, rgbimg_list, folder_name, toggledebug=False,
    #                                                      dilation=True)
    # pcdu.show_pcd(pcd, rgba=(1, 1, 1, 1))

    # depthimg_list, rgbimg_list = \
    #     du.load_frame_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'seg_result/'))
    # seed = np.asarray([0, .1, -.25])
    # cluster_o3d_list = extract_conponent_o3d(depthimg_list, rgbimg_list, folder_name=folder_name,
    #                                          seed=seed, toggledebug=True)

    # o3dpcd_list = du.load_o3dpcd_seq(folder_name)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in o3dpcd_list],
    #                          rgbasseq=[o3d.colors for o3d in o3dpcd_list], time_sleep=.1)
    # base.run()

    o3dpcd_list = du.load_o3dpcd_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'seg_pcd/'))
    o3dpcd_list = o3dpcd_list[20:200]

    res_o3dpcd_list = merge_o3dpcd_seq(o3dpcd_list)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in res_o3dpcd_list],
    #                          rgbasseq=[o3d.colors for o3d in res_o3dpcd_list], time_sleep=.5)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in res_o3dpcd_list[1:]],
    #                          rgbasseq=[o3d.colors for o3d in res_o3dpcd_list[1:]], time_sleep=.5)
    # base.run()
    for res_o3dpcd in res_o3dpcd_list:
        pcdu.show_pcd_withrgb(res_o3dpcd.points, res_o3dpcd.colors, show_percentage=1/len(res_o3dpcd_list)*2)
    # pcdu.show_pcd(pcd)
    base.run()
