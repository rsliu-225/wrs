import copy
import itertools
import os
import pickle
import random

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

import datagenerator.data_utils as utl
import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as wd
from multiprocessing import Process

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


def gen_kpts(cat, tor=.6, num_kpts=16, radius=.005, overwrite=False, path='', toggledebug=False):
    if not os.path.exists(os.path.join(path, cat, 'kpts')):
        os.mkdir(os.path.join(path, cat, 'kpts'))
    cnt = 0
    f_list = os.listdir(os.path.join(path, cat, 'partial'))
    # random.shuffle(f_list)
    for f in f_list:
        cnt += 1
        colors = []
        if cnt % 100 == 0:
            print(printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                                   prefix=f'Progress({cat}):',
                                   suffix='Complete', length=100), "\r")
        if f[-3:] != 'pcd':
            continue
        if os.path.exists(os.path.join(path, cat, 'kpts', f'{f.split(".pcd")[0]}.pkl')) and not overwrite:
            continue
        o3dpcd_i = o3d.io.read_point_cloud(f"{path}/{cat}/partial/{f}")
        o3dpcd_gt = o3d.io.read_point_cloud(f"{path}/{cat}/complete/{f}")
        pcd_i = np.asarray(o3dpcd_i.points)
        pcd_gt = np.asarray(o3dpcd_gt.points)
        kpts, kpts_rotseq = utl.get_kpts_gmm(pcd_gt, n_components=num_kpts, show=False)

        conf = []
        kdt_i = o3d.geometry.KDTreeFlann(o3dpcd_i)
        for p in np.asarray(kpts):
            k, _, _ = kdt_i.search_radius_vector_3d(p, radius)
            # print(k, len(pcd_i) / len(kpts))
            if k < (len(pcd_i) / len(kpts)) * tor:
                conf.append(0)
                colors.append([1, 0, 0, .3])
            else:
                conf.append(1)
                colors.append([0, 1, 0, .3])

        if toggledebug:
            # gm.gen_pointcloud(pcd_gt).attach_to(base)
            gm.gen_pointcloud(pcd_i, rgbas=[[1, 0, 0, 1]]).attach_to(base)
            for i, p in enumerate(kpts):
                gm.gen_sphere(p, radius=radius, rgba=colors[i]).attach_to(base)
                gm.gen_frame(p, rotmat=kpts_rotseq[i], length=.02, thickness=.001).attach_to(base)
            base.run()
        pickle.dump([kpts, kpts_rotseq, conf],
                    open(os.path.join(path, cat, 'kpts', f'{f.split(".pcd")[0]}.pkl'), 'wb'))
    print(printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                           prefix=f'Progress({cat}):',
                           suffix='Finished!', length=100), "\r")


def show(fo='./', cat='bspl'):
    random_f = random.choices(sorted(os.listdir(os.path.join(fo, cat, 'complete'))), k=10)
    for f in random_f:
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(fo, cat, 'complete', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[0, 1, 0, 1]]).attach_to(base)
            o3dpcd = o3d.io.read_point_cloud(os.path.join(fo, cat, 'partial', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()


if __name__ == '__main__':
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    path = 'E:/liu/org_data/dataset_prim'
    overwrite = False
    num_kpts = 16
    radius = .1 / 16
    # gen_kpts('quad', tor=.5, num_kpts=16, radius=.1 / 16, path=path, toggledebug=True)

    cat_list = []
    for fo in os.listdir(path):
        cat_list.append(fo)
    print(cat_list)
    cat_list = ['quad', 'bspl']
    proc = []
    for cat in cat_list:
        if cat in ['plat', 'tmpl']:
            tor = .5
        elif cat in ['multiview']:
            tor = .5
        else:
            tor = .5
        p = Process(target=gen_kpts, args=(cat, tor, num_kpts, radius, overwrite, path, False))
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

    # show(path, cat='multiview')
