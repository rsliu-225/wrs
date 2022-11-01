import h5py
import os
import open3d as o3d
import numpy as np

ROOT = os.path.abspath('./')

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255


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
        dset = f.create_dataset("complete_pcds", data=np.asarray(complete_pcds, dtype='<f4'))
        dset = f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds, dtype='<f4'))
        dset = f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        print(f.keys())


def gen_h5_new(f_name, org_path, goal_path, multiview=True):
    if not os.path.exists(goal_path):
        os.mkdir(goal_path)
    complete_pcds = []
    incomplete_pcds = []
    labels = []
    for fo in os.listdir(org_path):
        # elif fo == 'linear':
        #     label = 0
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
            id_range = range(0, int(np.floor(.8 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.8 * max(id_list))), int(np.floor(.9 * max(id_list))))
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
        dset = f.create_dataset("complete_pcds", data=np.asarray(complete_pcds, dtype='<f4'))
        dset = f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds, dtype='<f4'))
        dset = f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        print(f.keys())
    print('-----------------------------')




if __name__ == '__main__':
    import modeling.geometric_model as gm
    import visualization.panda.world as wd

    org_path = 'E:/liu/dataset_2048_prim_v10/'
    goal_path = 'D:/liu/MVP_Benchmark/completion/data_2048_prim_v10_mv/'
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    gen_h5_new('train', org_path, goal_path, multiview=True)
    gen_h5_new('test', org_path, goal_path, multiview=True)
    gen_h5_new('val', org_path, goal_path, multiview=True)

    # gen_h5('train', org_path, goal_path, multiview=False)
    # gen_h5('test', org_path, goal_path, multiview=False)
    # gen_h5('val', org_path, goal_path, multiview=False)


