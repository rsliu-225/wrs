import h5py
import os
import open3d as o3d
import numpy as np

ROOT = os.path.abspath('./')
ORG_DATA_PATH = 'E:/liu/dataset_v3/'


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


if __name__ == '__main__':
    complete_pcds = []
    incomplete_pcds = []
    labels = []
    for fo in os.listdir(ORG_DATA_PATH):
        print(fo)
        if fo == 'cubic':
            label = 2
        elif fo == 'quad':
            # label = 1
            break
        elif fo == 'linear':
            # label = 0
            break
        else:
            break

        for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'complete')):
            if f[-3:] == 'pcd':
                o3dpcd = o3d.io.read_point_cloud(os.path.join(ORG_DATA_PATH, fo, 'complete', f))
                complete_pcds.append(np.asarray(o3dpcd.points, dtype='<f4'))
                # o3d.visualization.draw_geometries([o3dpcd])
                labels.append(label)
                # print(len(complete_pcds[-1]))

        for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'partial')):
            if f[-3:] == 'pcd':
                o3dpcd = o3d.io.read_point_cloud(os.path.join(ORG_DATA_PATH, fo, 'partial', f))
                incomplete_pcds.append(np.asarray(o3dpcd.points, dtype='<f4'))
                # print(len(incomplete_pcds[-1]))
                # o3d.visualization.draw_geometries([o3dpcd])

    print(np.asarray(complete_pcds).shape)
    print(np.asarray(incomplete_pcds).shape)
    print(np.asarray(labels).shape)
    with h5py.File(f"{ROOT}/data/Train.h5", "w") as f:
        dset = f.create_dataset("dataset", data=np.asarray(complete_pcds))
        # dset = f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds))
        dset = f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))

        # dset['complete_pcds'] = complete_pcds
        # dset['incomplete_pcds'] = incomplete_pcds
        # dset['labels'] = labels
        print(f.keys())

        train_f = h5py.File(f'{os.path.abspath("./")}/data/Train.h5', 'r')

        print(train_f.name, train_f.keys())
        for k in train_f.keys():
            print(k)
        for pcd in train_f[k]:
            o3dpcd = nparray2o3dpcd(np.asarray(pcd))
        o3d.visualization.draw_geometries([o3dpcd])
