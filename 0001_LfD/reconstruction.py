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
import geojson


def remove_by_label(depthimg_list, rgbimg_list, label=0, write_f=True):
    bgmask_list = du.flist_remove_bg(depthimg_list, rgbimg_list, toggledebug=False)
    depthimg_list_res, rgbimg_list_res = [], []
    predictor = MaskRcnnPredictor()

    for i, im in tqdm(enumerate(rgbimg_list)):
        if i < 85:
            continue
        print(f"---------------image {i}---------------")
        hnd_prediction = predictor.predict(im, label)
        hnd_mask_list = hnd_prediction.get("pred_masks").numpy()
        if len(hnd_mask_list) == 0:
            print("no target mask!")
            continue

        visualized_pred = predictor.visualize_prediction(im, hnd_prediction)
        # cv2.imshow("prediction", visualized_pred)
        # cv2.waitKey(0)
        depthimg = depthimg_list[i]
        hnd_mask = np.zeros(depthimg.shape)
        for m in hnd_mask_list:
            hnd_mask = np.logical_or(hnd_mask, m)

        mask = np.logical_and(~bgmask_list[i], ~hnd_mask)
        # cv2.imshow('foreground', (~bgmask_list[i]).astype(np.uint8) * 255)
        # cv2.imshow('hand mask', (~hnd_mask).astype(np.uint8) * 255)
        # cv2.imshow('final mask', mask.astype(np.uint8) * 255)
        # cv2.waitKey(0)
        depthimg[~mask] = 0
        rgbimg = rgbimg_list[i] * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # cv2.imshow('rgb', rgbimg)
        # cv2.waitKey(0)

        pcd = realsense.rgbd2pcd(rgbimg, depthimg, toggledebug=True)
        pcdu.show_pcd(pcd)
        base.run()

        depthimg_list_res.append(depthimg)
        rgbimg_list_res.append(rgbimg)
        print(depthimg)
        if write_f:
            pickle.dump([depthimg, rgbimg], open(f"./seg_result/{folder_name}/{str(i).zfill(4)}.pkl", "wb"))
            cv2.imwrite(f'./mask_rcnn_seg/inference_results/{folder_name}/{str(i).zfill(4)}.png', visualized_pred)
    return depthimg_list_res, rgbimg_list_res


if __name__ == '__main__':
    folder_name = "recons"
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])

    realsense = rs.RealSense()
    depthimg_list, rgbimg_list = realsense.load_frame_seq(folder_name, root_path=os.path.join(
        "/home/rsliu/PycharmProjects/wrs/0001_LfD/seg_result/"))

    depthimg_list_res, rgbimg_list_res = remove_by_label(depthimg_list, rgbimg_list)
    # pcdu.show_pcd(pcd, rgba=(1, 1, 1, 1))
    for i in range(len(depthimg_list)):
        pcd = realsense.depth2pcd(depthimg_list[i])
        cluster_pts = du.get_max_cluster(pcd)
        pcdu.show_pcd(cluster_pts, rgba=(1, 0, 0, 1))
        base.run()

    realsense.show_frameseq(folder_name, root_path=os.path.join("/home/rsliu/PycharmProjects/wrs/0001_LfD/seg_result/"))
