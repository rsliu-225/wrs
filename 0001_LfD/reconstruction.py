import cv2
import matplotlib.pyplot as plt
import numpy as np
import visualization.panda.world as wd
import utils.pcd_utils as pcdu
import local_vis.realsense.realsense as rs
import local_vis.knt_azure.knt_azure as k4a
import basis.o3dhelper as o3dhelper
import detection_utils as du
import utils.vision_utils as vu
import open3d as o3d
import config_LfD as config
import os
import time
import pickle
import copy


def extract_component_o3d(camera, depthimg_list, rgbimg_list, folder_name=None, seed=None, toggledebug=False):
    if folder_name is not None:
        du.create_path(f'{config.DATA_PATH}/seg_result/{folder_name}/pcd/')
    cluster_o3d_list = []
    pcd_list = []

    for i in range(len(depthimg_list)):
        pcd_o3d = camera.rgbd2pcd(depthimg_list[i], rgbimg_list[i])
        pcd_o3d = o3dhelper.removeoutlier_o3d(pcd_o3d)
        if len(np.asarray(pcd_o3d.points)) < 500:
            continue
        # print(np.asarray(pcd_o3d.points))
        # du.cluster_meanshift(pcd_o3d.points, toggledebug=True)
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
            o3d.io.write_point_cloud(f'{config.DATA_PATH}/seg_result/{folder_name}/pcd/{str(i).zfill(4)}.pcd',
                                     cluster_o3d)

    if toggledebug:
        pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in cluster_o3d_list],
                                 rgbasseq=[o3d.colors for o3d in cluster_o3d_list], time_sleep=.5)
        pcdu.show_pcdseq(pcd_list, rgba=(1, 1, 1, .4), time_sleep=.5)
        base.run()
    return cluster_o3d_list


def extract_component_rg(depthimg_list, rgbimg_list, folder_name=None, seed=None, toggledebug=False):
    if folder_name is not None:
        du.create_path(f'{config.DATA_PATH}/seg_result/{folder_name}/rg/')
        du.create_path(f'{config.DATA_PATH}/mask/{folder_name}/obj/')
    res_depthimg_list = []
    res_rgbimg_list = []
    for i in range(len(depthimg_list)):
        try:
            components_list = du.get_dp_components(depthimg_list[i], toggledebug=False)
        except:
            continue
        if len(components_list) == 0:
            continue
        if seed is None:
            mask, seed = du.find_largest_dpcomponent(components_list)
        else:
            mask, seed = du.find_closest_dpcomponent(components_list, seed)
        if mask is None:
            print('skip')
            continue
        mask[mask == 255] = 1
        depthimg = copy.deepcopy(depthimg_list[i])
        rgbimg = copy.deepcopy(rgbimg_list[i])
        depthimg[mask == 0] = 0
        rgbimg = rgbimg * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        res_depthimg_list.append(depthimg)
        res_rgbimg_list.append(rgbimg)
        # cv2.imshow('result', mask)
        # cv2.imshow('depth', du.scale_depth_img(depthimg))
        # cv2.imshow('rgb', rgbimg)
        # cv2.waitKey(0)
        if folder_name is not None:
            pickle.dump([depthimg, rgbimg],
                        open(f'{config.DATA_PATH}/seg_result/{folder_name}/rg/{str(i).zfill(4)}.pkl', 'wb'))
            cv2.imwrite(f'{config.DATA_PATH}/mask/{folder_name}/obj/{str(i).zfill(4)}.png', mask * 255)

    return res_depthimg_list, res_rgbimg_list


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


def graphcut(image, init_mask, itercont=10):
    if len(image.shape) == 2:
        image = vu.gray23channel(du.scale_depth_img(image))
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")
    saliency_mask = cv2.threshold(saliency_map.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Saliency Mask", saliency_mask)
    cv2.waitKey(0)

    # mask = np.where((init_mask == 1) & (saliency_mask == 255), 2, init_mask)
    # mask = np.where((mask == 0) & (saliency_mask == 255), 2, mask)
    mask = init_mask
    mask[init_mask == 0] = 3
    mask[init_mask == 1] = 0
    cv2.imshow("Mask", mask * 100)
    cv2.waitKey(0)

    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    start = time.time()
    mask, bgModel, fgModel = cv2.grabCut(image, mask, None, bgModel, fgModel, iterCount=itercont,
                                         mode=cv2.GC_INIT_WITH_MASK)
    end = time.time()
    print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))
    values = (
        ("Definite Background", cv2.GC_BGD),
        ("Probable Background", cv2.GC_PR_BGD),
        ("Definite Foreground", cv2.GC_FGD),
        ("Probable Foreground", cv2.GC_PR_FGD),
    )
    # loop over the possible GrabCut mask values
    for (name, value) in values:
        # construct a mask that for the current value
        print("[INFO] showing mask for '{}'".format(name))
        valueMask = (mask == value).astype("uint8") * 255
        # display the mask so we can visualize it
        cv2.imshow(name, valueMask)
        cv2.waitKey(0)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)

    cv2.imshow("Input", image)
    cv2.imshow("GrabCut Mask", outputMask)
    cv2.imshow("GrabCut Output", output)
    cv2.waitKey(0)


if __name__ == '__main__':
    folder_name = 'templ'
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    # camera = rs.RealSense()
    camera = k4a.KinectAzura(online=False)

    '''
    remove background and hand
    '''
    # depthimg_list, rgbimg_list, _ = du.load_frame_seq(folder_name,
    #                                                   root_path=os.path.join(config.DATA_PATH, 'raw_img/k4a/seq/'))
    # # # camera.show_rgbdseq(depthimg_list, rgbimg_list)
    # # bgmask_list = du.get_bg_maskseq(depthimg_list, bg_inxs=range(20), threshold=100, folder_name=folder_name,
    # #                                 toggledebug=False)
    # # hndmask_list = du.get_tgt_maskseq(rgbimg_list, start_id=50, folder_name=folder_name, toggledebug=False)
    # # depthimg_list, rgbimg_list, _ = du.load_frame_seq(folder_name,
    # #                                                   root_path=os.path.join(config.DATA_PATH, 'filter_result/'))
    # # camera.show_frameseq(depthimg_list, rgbimg_list)
    #
    # bgmask_list, bg_id_list = du.load_mask(folder_name, 'bg')
    # hndmask_list, hnd_id_list = du.load_mask(folder_name, 'hand')
    # ls, id_list = du.get_list_inersection_by_id([bgmask_list, hndmask_list], [bg_id_list, hnd_id_list])
    # bgmask_list, hndmask_list = ls
    # hndmask_list = [np.logical_and(~bgmask_list[i], hndmask_list[i]) for i in range(len(id_list))]
    # objmask_list = [np.logical_and(~bgmask_list[i], ~hndmask_list[i]) for i in range(len(id_list))]
    # obj_depthimg_list, obj_rgbimg_list = \
    #     du.filter_by_maskseq(objmask_list, depthimg_list, rgbimg_list, id_list, folder_name=folder_name, exclude=False,
    #                          toggledebug=False)
    # camera.show_frameseq(obj_depthimg_list, obj_rgbimg_list)
    # # hnd_depthimg_list, hnd_rgbimg_list = \
    # #     du.filter_by_maskseq(hndmask_list, depthimg_list, rgbimg_list, id_list, exclude=False, toggledebug=False)
    # # camera.show_frameseq(hnd_depthimg_list, hnd_rgbimg_list)
    # # front_depthimg_list, front_rgbimg_list = \
    # #     du.filter_by_maskseq(bgmask_list, depthimg_list, rgbimg_list, id_list, exclude=True, toggledebug=False)
    # # camera.show_frameseq(front_depthimg_list, front_rgbimg_list)
    #
    # for i, f_name in enumerate(hnd_id_list):
    #     # init_mask = np.asarray(np.logical_not(bgmask_list[i]), dtype='uint8')
    #     init_mask = np.asarray(hndmask_list[i], dtype='uint8')
    #     graphcut(front_rgbimg_list[int(f_name)], init_mask)
    #     # graphcut(front_depthimg_list[int(f_name)], init_mask)
    #
    # # pcd = camera.rgbd2pcd(hnd_depthimg_list[100], hnd_rgbimg_list[100], toggledebug=True)
    # # pcd2 = camera.rgbd2pcd(front_depthimg_list[100], front_rgbimg_list[100], toggledebug=True)
    # # du.oneclasssvm_pcd(pcd, pcd2)
    #
    # res_depthimg_list, res_rgbimg_list = \
    #     du.filter_by_maskseq([np.logical_and(~bgmask_list[i], ~hndmask_list[i]) for i in range(len(id_list))],
    #                          depthimg_list, rgbimg_list, id_list, exclude=False, toggledebug=False)
    # # camera.show_rgbdseq(hnd_depthimg_list, hnd_rgbimg_list, win_name='hnd')
    # camera.show_rgbdseq(res_depthimg_list, res_rgbimg_list, win_name='rest')

    '''
    extract main cluster
    '''
    # depthimg_list, rgbimg_list, _ = \
    #     du.load_frame_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'filter_result/'))
    # camera.show_frameseq(depthimg_list, rgbimg_list)
    # camera.show_rgbdseq(depthimg_list, rgbimg_list)

    # seed = np.asarray([0, .1, -.25])
    # cluster_o3d_list = extract_component_o3d(camera, depthimg_list, rgbimg_list, folder_name=folder_name,
    #                                          seed=None, toggledebug=True)
    # o3dpcd_list = du.load_o3dpcd_seq(folder_name)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in o3dpcd_list],
    #                          rgbasseq=[o3d.colors for o3d in o3dpcd_list], time_sleep=.1)
    # base.run()

    # comp_depthimg_list, comp_rgbimg_list = \
    #     extract_component_rg(depthimg_list, rgbimg_list, folder_name=folder_name, toggledebug=True)
    comp_depthimg_list, comp_rgbimg_list, _ = \
        du.load_frame_seq(path=os.path.join(config.DATA_PATH, 'seg_result', folder_name, 'rg'))
    # camera.show_rgbdseq(comp_depthimg_list, comp_rgbimg_list, win_name='hnd')
    camera.show_rgbdseq_p3d(comp_depthimg_list, comp_rgbimg_list)

    '''
    icp
    '''
    # o3dpcd_list = du.load_o3dpcd_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'seg_pcd/'))
    # o3dpcd_list = o3dpcd_list[20:200]
    #
    # res_o3dpcd_list = merge_o3dpcd_seq(o3dpcd_list)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in res_o3dpcd_list],
    #                          rgbasseq=[o3d.colors for o3d in res_o3dpcd_list], time_sleep=.5)
    # pcdu.show_pcdseq_withrgb(pcdseq=[o3d.points for o3d in res_o3dpcd_list[1:]],
    #                          rgbasseq=[o3d.colors for o3d in res_o3dpcd_list[1:]], time_sleep=.5)
    # base.run()
    # for res_o3dpcd in res_o3dpcd_list:
    #     pcdu.show_pcd_withrgb(res_o3dpcd.points, res_o3dpcd.colors, show_percentage=1 / len(res_o3dpcd_list) * 2)
    # # pcdu.show_pcd(pcd)
    # base.run()
