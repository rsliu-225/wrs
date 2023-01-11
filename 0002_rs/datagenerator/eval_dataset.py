import pickle
import random

import h5py
import os
import open3d as o3d
import numpy as np
import basis.o3dhelper as o3dh
import collections
import modeling.geometric_model as gm
import scipy.spatial.kdtree as kdt
import matplotlib.pyplot as plt
from multiprocessing import Process
import data_utils as du

ROOT = os.path.abspath('./')

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def query_ball_tree():
    from scipy.spatial import KDTree
    rng = np.random.default_rng()
    points1 = rng.random((15, 2))
    points2 = rng.random((15, 2))
    plt.figure(figsize=(6, 6))
    plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
    plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
    kd_tree1 = KDTree(points1)
    kd_tree2 = KDTree(points2)
    indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
    for i in range(len(indexes)):
        for j in indexes[i]:
            plt.plot([points1[i, 0], points2[j, 0]],
                     [points1[i, 1], points2[j, 1]], "-r")
    plt.show()


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def remove_files(path, cat='plat'):
    cnt = 0
    for fo in os.listdir(path):
        if fo != cat:
            continue
        for f in os.listdir(os.path.join(path, fo, 'complete')):
            cnt += 1
            if cnt % 200 == 0:
                print(du.printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                                          prefix=f'Progress({cat}):',
                                          suffix='Complete', length=100), "\r")
            if f[-3:] != 'pcd':
                continue
            o3dpcd_i = o3d.io.read_point_cloud(os.path.join(path, fo, 'partial', f))
            labels = np.array(o3dpcd_i.cluster_dbscan(eps=0.15, min_points=5))
            max_label = labels.max()
            if max_label >= 1:
                print(cat, f)
                # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                # colors[labels < 0] = 0
                # o3dpcd_i.colors = o3d.utility.Vector3dVector(colors[:, :3])
                # o3d.visualization.draw_geometries([o3dpcd_i])
                os.remove(os.path.join(path, fo, 'complete', f))
                os.remove(os.path.join(path, fo, 'partial', f))
                if os.path.exists(os.path.join(path, fo, 'kpts', f[:-4] + '.pkl')):
                    os.remove(os.path.join(path, fo, 'kpts', f[:-4] + '.pkl'))
                if os.path.exists(os.path.join(path, fo, 'conf', f[:-4] + '.pkl')):
                    os.remove(os.path.join(path, fo, 'conf', f[:-4] + '.pkl'))
        print(du.printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                                  prefix=f'Progress({cat}):',
                                  suffix='Finished!', length=100), "\r")


# def remove_files(path, cat='plat'):
#     for fo in os.listdir(path):
#         if fo != cat:
#             continue
#         for f in os.listdir(os.path.join(path, fo, 'kpts')):
#             if not os.path.exists(os.path.join(path, fo, 'partial', f[:-4] + '.pcd')):
#                 print(cat, f)
#                 os.remove(os.path.join(path, fo, 'kpts', f))


def show_dataset_o3d(path, cat='plat'):
    for fo in os.listdir(path):
        if fo != cat:
            continue
        for f in os.listdir(os.path.join(path, fo, 'complete')):
            if f[-3:] != 'pcd':
                continue
            print(f)
            o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(path, fo, 'complete', f))
            o3dpcd_i = o3d.io.read_point_cloud(os.path.join(path, fo, 'partial', f))
            # gm.gen_pointcloud(np.asarray(o3dpcd_gt.points)).attach_to(base)
            o3dpcd_i.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
            o3d.visualization.draw_geometries([o3dpcd_i])
            o3d.visualization.draw_geometries([o3dpcd_gt])


def show_multiview(path='./', cat='bspl'):
    random_f = random.choices(sorted(os.listdir(os.path.join(path, cat, 'complete'))), k=10)
    for f in random_f:
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(path, cat, 'complete', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[0, 1, 0, 1]]).attach_to(base)
            o3dpcd = o3d.io.read_point_cloud(os.path.join(path, cat, 'partial', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    path = 'E:/liu/org_data/dataset/'

    # remove_files(path, cat='plat')
    cat_list = []
    for fo in os.listdir(path):
        cat_list.append(fo)
    print(cat_list)
    cat_list = ['rand']
    proc = []
    for cat in cat_list:
        p = Process(target=remove_files, args=(path, cat))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
