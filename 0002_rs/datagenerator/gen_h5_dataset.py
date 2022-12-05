import os
import pickle

import h5py
import numpy as np
import open3d as o3d

ROOT = os.path.abspath('./')

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


def gen_h5(f_name, org_path, goal_path, multiview=True):
    if not os.path.exists(goal_path):
        os.mkdir(goal_path)
    complete_pcds = []
    incomplete_pcds = []
    labels = []
    for fo in os.listdir(org_path):
        if fo == 'quad':
            label = 1
        elif fo == 'bspl':
            label = 2
        elif fo == 'plat':
            label = 3
        elif fo == 'tmpl':
            label = 4
        elif fo == 'sprl':
            label = 5
        else:
            continue
        # id_list = [int(f.split('_')[0]) for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'complete'))]
        print('category:', fo)

        id_list = [int(f.split(fo)[1].split('.pcd')[0]) for f in
                   os.listdir(os.path.join(org_path, fo, 'complete'))]
        if f_name == 'train':
            id_range = range(0, int(np.floor(.8 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.8 * max(id_list))), int(np.floor(.9 * max(id_list))))
        elif f_name == 'test':
            id_range = range(int(np.floor(.9 * max(id_list))), max(id_list) + 1)
        else:
            id_range = (0, max(id_list) + 1)
        print(id_range)

        for f in os.listdir(os.path.join(org_path, fo, 'complete')):
            if f[-3:] == 'pcd' and int(f.split(fo)[1].split('.pcd')[0]) in id_range:
                if not os.path.exists(os.path.join(org_path, fo, 'partial', f)):
                    print(f)
                    continue
                o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'complete', f))
                o3dpcd_i = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'partial', f))
                incomplete_pcds.append(np.asarray(o3dpcd_i.points, dtype='<f4'))
                complete_pcds.append(np.asarray(o3dpcd_gt.points, dtype='<f4'))
                # o3dpcd_gt.paint_uniform_color([0, 1, 0])
                # o3dpcd_i.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
                labels.append(label)
                if multiview:
                    if not os.path.exists(os.path.join(org_path, 'multiview', 'complete', f)) or \
                            not os.path.exists(os.path.join(org_path, 'multiview', 'partial', f)):
                        print(f)
                        continue
                    o3dpcd_mv_gt = o3d.io.read_point_cloud(os.path.join(org_path, 'multiview', 'complete', f))
                    o3dpcd_mv_i = o3d.io.read_point_cloud(os.path.join(org_path, 'multiview', 'partial', f))
                    incomplete_pcds.append(np.asarray(o3dpcd_mv_i.points, dtype='<f4'))
                    complete_pcds.append(np.asarray(o3dpcd_mv_gt.points, dtype='<f4'))
                    # o3dpcd_mv_gt.paint_uniform_color([0, 1, 0])
                    # o3dpcd_mv_i.paint_uniform_color([1, 0, 0])
                    # o3d.visualization.draw_geometries([o3dpcd_mv_gt, o3dpcd_mv_i])
                    labels.append(-label)

    print('complete pcd shape:', np.asarray(complete_pcds).shape)
    print('incomplete pcd shape:', np.asarray(incomplete_pcds).shape)
    print('label pcd shape:', np.asarray(labels).shape)
    with h5py.File(f"{goal_path}/{f_name}.h5", "w") as f:
        f.create_dataset("complete_pcds", data=np.asarray(complete_pcds, dtype='<f4'))
        f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds, dtype='<f4'))
        f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        print(f.keys())


def gen_h5_new(f_name, org_path, goal_path, multiview=True):
    if not os.path.exists(goal_path):
        os.mkdir(goal_path)
    complete_pcds = []
    incomplete_pcds = []
    labels = []
    for fo in os.listdir(org_path):
        if fo == 'quad':
            label = 1
        elif fo == 'bspl':
            label = 2
        elif fo == 'plat':
            label = 3
        elif fo == 'tmpl':
            label = 4
        else:
            continue
        print('category:', fo)
        id_list = [int(f.split('_')[0]) for f in os.listdir(os.path.join(org_path, fo, 'complete'))]
        if f_name == 'train':
            id_range = range(0, int(np.floor(.7 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.7 * max(id_list))), int(np.floor(.9 * max(id_list))))
        elif f_name == 'test':
            id_range = range(int(np.floor(.9 * max(id_list))), max(id_list) + 1)
        else:
            id_range = (0, max(id_list) + 1)
        print(id_range)

        for f in os.listdir(os.path.join(org_path, fo, 'complete')):
            if f[-3:] == 'pcd' and int(f.split('_')[0]) in id_range:
                o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'complete', f))
                o3dpcd_i = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'partial', f))
                incomplete_pcds.append(np.asarray(o3dpcd_i.points, dtype='<f4'))
                complete_pcds.append(np.asarray(o3dpcd_gt.points, dtype='<f4'))
                # o3dpcd_gt.paint_uniform_color([0, 1, 0])
                # o3dpcd_i.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
                labels.append(label)
                if multiview:
                    o3dpcd_mv_gt = o3d.io.read_point_cloud(
                        os.path.join(org_path, 'multiview', 'complete', f'{fo}_{f}'))
                    o3dpcd_mv_i = o3d.io.read_point_cloud(
                        os.path.join(org_path, 'multiview', 'partial', f'{fo}_{f}'))
                    incomplete_pcds.append(np.asarray(o3dpcd_mv_i.points, dtype='<f4'))
                    complete_pcds.append(np.asarray(o3dpcd_mv_gt.points, dtype='<f4'))
                    # o3dpcd_mv_gt.paint_uniform_color([0, 1, 0])
                    # o3dpcd_mv_i.paint_uniform_color([1, 0, 0])
                    # o3d.visualization.draw_geometries([o3dpcd_mv_gt, o3dpcd_mv_i])
                    labels.append(-label)

    print('complete pcd shape:', np.asarray(complete_pcds).shape)
    print('incomplete pcd shape:', np.asarray(incomplete_pcds).shape)
    print('label pcd shape:', np.asarray(labels).shape)
    with h5py.File(f"{goal_path}/{f_name}.h5", "w") as f:
        f.create_dataset("complete_pcds", data=np.asarray(complete_pcds, dtype='<f4'))
        f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds, dtype='<f4'))
        f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        print(f.keys())
    print('-----------------------------')


from collections import Counter


def gen_h5_conf(f_name, org_path, goal_path, multiview_fo=None):
    if not os.path.exists(goal_path):
        os.mkdir(goal_path)
    pcds_gt = []
    pcds_i = []
    labels = []
    coverages = []
    confs = []
    for fo in os.listdir(org_path):
        if fo == 'quad':
            label = 1
        elif fo == 'bspl':
            label = 2
        elif fo == 'plat':
            label = 3
        elif fo == 'tmpl':
            label = 4
        else:
            continue
        print('category:', fo)
        id_list = [int(f.split('_')[0]) for f in os.listdir(os.path.join(org_path, fo, 'complete'))]
        if f_name == 'train':
            id_range = range(0, int(np.floor(.7 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.7 * max(id_list))), int(np.floor(.9 * max(id_list))))
        elif f_name == 'test':
            id_range = range(int(np.floor(.9 * max(id_list))), max(id_list) + 1)
        else:
            id_range = (0, max(id_list) + 1)
        print(id_range)

        for f in os.listdir(os.path.join(org_path, fo, 'complete')):
            if f[-3:] == 'pcd' and int(f.split('_')[0]) in id_range:
                o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'complete', f))
                conf = pickle.load(open(os.path.join(org_path, fo, 'conf', f[:-3] + 'pkl'), 'rb'))
                o3dpcd_i = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'partial', f))
                # gt = np.hstack((np.asarray(o3dpcd_gt.points), np.asarray(conf).reshape(2048, 1)))
                pcds_i.append(np.asarray(o3dpcd_i.points, dtype='<f4'))
                pcds_gt.append(np.asarray(o3dpcd_gt.points, dtype='<f4'))
                # o3dpcd_gt.paint_uniform_color(COLOR[2])
                # o3dpcd_i.paint_uniform_color(COLOR[1])
                # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
                labels.append(label)
                coverages.append(Counter(conf)[1] / 2048)
                confs.append(conf)
                if multiview_fo:
                    o3dpcd_mv_gt = o3d.io.read_point_cloud(
                        os.path.join(org_path, multiview_fo, 'complete', f'{fo}_{f}'))
                    o3dpcd_mv_i = o3d.io.read_point_cloud(
                        os.path.join(org_path, multiview_fo, 'partial', f'{fo}_{f}'))
                    mv_conf = pickle.load(open(
                        os.path.join(org_path, multiview_fo, 'conf', f'{fo}_{f[:-3]}pkl'), 'rb'))
                    coverages.append(Counter(mv_conf)[1] / 2048)
                    confs.append(mv_conf)
                    # mv_gt = np.hstack((np.asarray(o3dpcd_mv_gt.points), np.asarray(mv_conf).reshape(2048, 1)))
                    pcds_i.append(np.asarray(o3dpcd_mv_i.points, dtype='<f4'))
                    pcds_gt.append(np.asarray(o3dpcd_mv_gt.points, dtype='<f4'))
                    # o3dpcd_mv_gt.paint_uniform_color(COLOR[1])
                    # o3dpcd_mv_i.paint_uniform_color(COLOR[2])
                    # o3d.visualization.draw_geometries([o3dpcd_mv_gt, o3dpcd_mv_i])
                    labels.append(-label)

    print(f'-------------{f_name}----------------')
    print('complete pcd shape:', np.asarray(pcds_gt).shape)
    print('incomplete pcd shape:', np.asarray(pcds_i).shape)
    print('label shape:', np.asarray(labels).shape)
    print('coverages shape:', np.asarray(coverages).shape)
    print('conf shape:', np.asarray(confs).shape)
    with h5py.File(f"{goal_path}/{f_name}.h5", "w") as f:
        f.create_dataset("complete_pcds", data=np.asarray(pcds_gt, dtype='<f4'))
        f.create_dataset("incomplete_pcds", data=np.asarray(pcds_i, dtype='<f4'))
        f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        f.create_dataset("coverages", data=np.asarray(coverages, dtype='<f4'))
        f.create_dataset("confs", data=np.asarray(confs, dtype='<f4'))
        print(f.keys())


def gen_h5_kpts(f_name, org_path, goal_path, multiview=True):
    if not os.path.exists(goal_path):
        os.mkdir(goal_path)
    pcds_gt = []
    pcds_i = []
    kpts_list = []
    knrmls_list = []
    conf_list = []
    labels = []
    for fo in os.listdir(org_path):
        if fo == 'quad':
            label = 1
        elif fo == 'bspl':
            label = 2
        elif fo == 'plat':
            label = 3
        elif fo == 'tmpl':
            label = 4
        else:
            continue
        print('category:', fo)
        id_list = [int(f.split('_')[0]) for f in os.listdir(os.path.join(org_path, fo, 'complete'))]
        if f_name == 'train':
            id_range = range(0, int(np.floor(.7 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.7 * max(id_list))), int(np.floor(.9 * max(id_list))))
        elif f_name == 'test':
            id_range = range(int(np.floor(.9 * max(id_list))), max(id_list) + 1)
        else:
            id_range = (0, max(id_list) + 1)
        print(id_range)

        for f in os.listdir(os.path.join(org_path, fo, 'complete')):
            if f[-3:] == 'pcd' and int(f.split('_')[0]) in id_range:
                o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'complete', f))
                kpts, krots, conf = pickle.load(open(os.path.join(org_path, fo, 'kpts', f[:-3] + 'pkl'), 'rb'))
                o3dpcd_i = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'partial', f))
                # gt = np.hstack((np.asarray(o3dpcd_gt.points), np.asarray(conf).reshape(2048, 1)))

                pcds_i.append(np.asarray(o3dpcd_i.points, dtype='<f4'))
                pcds_gt.append(np.asarray(o3dpcd_gt.points, dtype='<f4'))
                labels.append(label)
                kpts_list.append(np.asarray(kpts, dtype='<f4'))
                knrmls_list.append(np.asarray(krots, dtype='<f4')[:, :, 0])
                conf_list.append(np.asarray(conf, dtype='<f4'))

                # o3dpcd_gt.paint_uniform_color([0, 1, 0])
                # o3dpcd_i.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])
                if multiview:
                    o3dpcd_mv_gt = o3d.io.read_point_cloud(
                        os.path.join(org_path, 'multiview', 'complete', f'{fo}_{f}'))
                    o3dpcd_mv_i = o3d.io.read_point_cloud(
                        os.path.join(org_path, 'multiview', 'partial', f'{fo}_{f}'))

                    pcds_i.append(np.asarray(o3dpcd_mv_i.points, dtype='<f4'))
                    pcds_gt.append(np.asarray(o3dpcd_mv_gt.points, dtype='<f4'))
                    # o3dpcd_mv_gt.paint_uniform_color([0, 1, 0])
                    # o3dpcd_mv_i.paint_uniform_color([1, 0, 0])
                    # o3d.visualization.draw_geometries([o3dpcd_mv_gt, o3dpcd_mv_i])
                    labels.append(-label)
                    kpts_list.append(np.asarray(kpts, dtype='<f4'))
                    knrmls_list.append(np.asarray(krots, dtype='<f4')[:, :, 0])
                    conf_list.append(np.asarray(conf, dtype='<f4'))

    print('complete pcd shape:', np.asarray(pcds_gt).shape)
    print('incomplete pcd shape:', np.asarray(pcds_i).shape)
    print('kpts shape:', np.asarray(kpts_list).shape)
    print('knrmls shape:', np.asarray(knrmls_list).shape)
    print('conf shape:', np.asarray(conf_list).shape)
    print('label pcd shape:', np.asarray(labels).shape)
    with h5py.File(f"{goal_path}/{f_name}.h5", "w") as f:
        f.create_dataset("complete_pcds", data=np.asarray(pcds_gt, dtype='<f4'))
        f.create_dataset("incomplete_pcds", data=np.asarray(pcds_i, dtype='<f4'))
        f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        f.create_dataset("kpts", data=np.asarray(kpts_list, dtype='<f4'))
        f.create_dataset("knrmls", data=np.asarray(knrmls_list, dtype='<f4'))
        f.create_dataset("conf", data=np.asarray(conf_list, dtype='<f4'))
        print(f.keys())
    print('-----------------------------')


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    # gen_h5('train', org_path, goal_path, multiview=False)
    # gen_h5('test', org_path, goal_path, multiview=False)
    # gen_h5('val', org_path, goal_path, multiview=False)

    org_path = 'E:/liu/org_data/dataset/'
    # goal_path = 'E:/liu/h5_data/data_rec/'

    # gen_h5_new('train', org_path, goal_path, multiview=True)
    # gen_h5_new('test', org_path, goal_path, multiview=True)
    # gen_h5_new('val', org_path, goal_path, multiview=True)

    # goal_path = 'E:/liu/h5_data/data_prim_womv/'
    # gen_h5_new('train', org_path, goal_path, multiview=False)
    # gen_h5_new('test', org_path, goal_path, multiview=False)
    # gen_h5_new('val', org_path, goal_path, multiview=False)

    goal_path = 'E:/liu/h5_data/data_conf/'
    gen_h5_conf('train', org_path, goal_path, multiview_fo='multiview')
    gen_h5_conf('test', org_path, goal_path, multiview_fo='multiview')
    gen_h5_conf('val', org_path, goal_path, multiview_fo='multiview')

    goal_path = 'E:/liu/h5_data/data_conf_wodiff/'
    gen_h5_conf('train', org_path, goal_path, multiview_fo='multiview_true')
    gen_h5_conf('test', org_path, goal_path, multiview_fo='multiview_true')
    gen_h5_conf('val', org_path, goal_path, multiview_fo='multiview_true')

    # org_path = 'E:/liu/org_data/dataset_kpts/'
    # goal_path = 'E:/liu/h5_data/data_kpts/'
    # gen_h5_kpts('train', org_path, goal_path, multiview=False)
    # gen_h5_kpts('test', org_path, goal_path, multiview=False)
    # gen_h5_kpts('val', org_path, goal_path, multiview=False)
