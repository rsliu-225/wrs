import pickle
import random

import h5py
import os
import open3d as o3d
import numpy as np
import basis.o3dhelper as o3dh
import collections
import modeling.geometric_model as gm

ROOT = os.path.abspath('./')

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def show_dataset_h5(goal_path, f_name, label=4):
    f = h5py.File(f'{goal_path}/{f_name}.h5', 'r')
    print(f.name, f.keys())
    for i in range(len(f['complete_pcds'])):
        if f['labels'][i] == label:
            o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(f['complete_pcds'][i]))
            o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(f['incomplete_pcds'][i]))
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            o3dpcd_i.paint_uniform_color(COLOR[0])
            # o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_gt])
            o3d.visualization.draw_geometries([o3dpcd_i])
            o3d.visualization.draw_geometries([o3dpcd_gt])


def show_dataset_o3d(path, cat=['plat']):
    for fo in os.listdir(path):
        if fo not in cat:
            continue
        f_list = os.listdir(os.path.join(path, fo, 'complete'))
        random.shuffle(f_list)
        for f in f_list:
            if f[-3:] != 'pcd':
                continue
            # if int(f.split('_')[0]) < 1600:
            #     continue
            print(f)
            o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(path, fo, 'complete', f))
            o3dpcd_i = o3d.io.read_point_cloud(os.path.join(path, fo, 'partial', f))
            # gm.gen_pointcloud(np.asarray(o3dpcd_gt.points)).attach_to(base)
            o3dpcd_i.paint_uniform_color(COLOR[0])
            o3dpcd_gt.paint_uniform_color(COLOR[1])
            # o3d.visualization.draw_geometries([o3dpcd_i])
            # o3d.visualization.draw_geometries([o3dpcd_gt])
            draw_geometry_with_rotation([o3dpcd_i])


def show_conf(path, cat=['plat'], toggledebug=False):
    coverage_list = []
    for fo in os.listdir(path):
        if fo not in cat:
            continue
        for f in os.listdir(os.path.join(path, fo, 'complete')):
            if f[-3:] != 'pcd':
                continue
            # if int(f.split('_')[0]) != 802:
            #     continue
            conf = pickle.load(open(os.path.join(path, fo, 'conf', f[:-3] + 'pkl'), 'rb'))
            coverage_list.append(collections.Counter(conf)[1] / 2048)
            if toggledebug:
                if coverage_list[-1] < 1:
                    print(f, coverage_list[-1])
                    o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(path, fo, 'complete', f))
                    colors = []
                    for v in conf:
                        if v == 1:
                            colors.append(COLOR[1])
                        else:
                            colors.append(COLOR[2])
                    o3dpcd_gt.colors = o3d.utility.Vector3dVector(colors)
                    o3dpcd_i = o3d.io.read_point_cloud(os.path.join(path, fo, 'partial', f))
                    # gm.gen_pointcloud(np.asarray(o3dpcd_gt.points)).attach_to(base)
                    o3dpcd_i.paint_uniform_color(COLOR[0])
                    # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
                    o3d.visualization.draw_geometries([o3dpcd_i])
                    o3dpcd_i.paint_uniform_color((.7, .7, .7))
                    o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
    print('coverage', round(np.asarray(coverage_list).mean(), 2), round(np.asarray(coverage_list).std(), 2))


def show_kpts(path, cat):
    for fo in os.listdir(path):
        if fo not in cat:
            continue
        print(fo)
        f_list = os.listdir(os.path.join(path, fo, 'complete'))
        random.shuffle(f_list)
        for f in f_list:
            print(f)
            if f[-3:] != 'pcd':
                continue
            kpts, kpts_rotseq, conf = pickle.load(open(os.path.join(path, fo, 'kpts', f[:-3] + 'pkl'), 'rb'))
            o3dpcd_i = o3d.io.read_point_cloud(os.path.join(path, fo, 'partial', f))
            pcd_i = np.asarray(o3dpcd_i.points)
            # gm.gen_pointcloud(pcd_gt).attach_to(base)
            gm.gen_pointcloud(pcd_i, rgbas=[[1, 0, 0, 1]]).attach_to(base)
            for i, p in enumerate(kpts):
                if conf[i] == 1:
                    gm.gen_sphere(p, radius=.005, rgba=[0, 1, 0, .4]).attach_to(base)
                else:
                    gm.gen_sphere(p, radius=.005, rgba=[1, 0, 0, .4]).attach_to(base)

                gm.gen_frame(p, rotmat=kpts_rotseq[i], length=.02, thickness=.001).attach_to(base)
            base.run()


def show_multiview(path='./', cat='bspl'):
    random_f = random.choices(sorted(os.listdir(os.path.join(path, cat, 'complete'))), k=10)
    for f in random_f:
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(path, cat, 'complete', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[0, 1, 0, 1]]).attach_to(base)
            o3dpcd = o3d.io.read_point_cloud(os.path.join(path, cat, 'partial', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()


def draw_geometry_with_rotation(o3d_geos):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(o3d_geos, rotate_view)


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    goal_path = 'E:/liu/h5_data/data_prim/'

    # show_dataset_h5(goal_path, 'train', label=3)
    # show_dataset_h5(goal_path, 'test', label=2)
    path = 'E:/liu/org_data/dataset/'
    # show_dataset_o3d(path, cat=['bspl', 'quad'])
    # show_dataset_o3d(path, cat=['plat'])
    # show_dataset_o3d(path, cat=['tmpl'])
    # show_dataset_o3d(path, cat=['multiview'])

    # path = 'E:/liu/org_data/dataset/'
    # show_conf(path, cat=['bspl'], toggledebug=False)
    show_conf(path, cat=['rlen'], toggledebug=False)
    # show_conf(path, cat=['plat'], toggledebug=False)
    # show_conf(path, cat=['tmpl'], toggledebug=False)
    show_conf(path, cat=['multiview'], toggledebug=False)
    # show_conf(path, cat=['multiview_true'], toggledebug=False)
    # show_kpts('E:/liu/org_data/dataset_kpts/', cat=['plat'])
    # show_dataset_h5('val')
    # show_dataset_o3d(cat='quad')
    # base.run()
