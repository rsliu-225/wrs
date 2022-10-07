import h5py
import os
import open3d as o3d
import numpy as np

ROOT = os.path.abspath('./')
ORG_DATA_PATH = 'E:/liu/dataset_2048/'
GOAL_DATA_PATH = 'D:/liu/MVP_Benchmark/completion/data/'


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


def gen_h5(f_name):
    complete_pcds = []
    incomplete_pcds = []
    labels = []
    for fo in os.listdir(ORG_DATA_PATH):
        print(fo)
        if fo == 'cubic':
            label = 2
        elif fo == 'quad':
            label = 1
        elif fo == 'linear':
            label = 0
        else:
            break

        id_list = [int(f.split('_')[0]) for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'complete'))]
        if f_name == 'train':
            id_range = range(0, int(np.floor(.8 * max(id_list))))
        elif f_name == 'val':
            id_range = range(int(np.floor(.8 * max(id_list))), int(np.floor(.9 * max(id_list))))
        elif f_name == 'test':
            id_range = range(int(np.floor(.9 * max(id_list))), max(id_list)+1)
        else:
            id_range = (0, max(id_list) + 1)
        print(id_range)
        print([i for i in id_range])

        for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'complete')):
            if f[-3:] == 'pcd' and int(f.split('_')[0]) in id_range:
                o3dpcd = o3d.io.read_point_cloud(os.path.join(ORG_DATA_PATH, fo, 'complete', f))
                complete_pcds.append(np.asarray(o3dpcd.points, dtype='<f4'))
                # o3d.visualization.draw_geometries([o3dpcd])
                labels.append(label)
                # print(len(complete_pcds[-1]))

        for f in os.listdir(os.path.join(ORG_DATA_PATH, fo, 'partial')):
            if f[-3:] == 'pcd' and int(f.split('_')[0]) in id_range:
                o3dpcd = o3d.io.read_point_cloud(os.path.join(ORG_DATA_PATH, fo, 'partial', f))
                incomplete_pcds.append(np.asarray(o3dpcd.points, dtype='<f4'))
                # print(len(incomplete_pcds[-1]))
                # o3d.visualization.draw_geometries([o3dpcd])

    print(np.asarray(complete_pcds).shape)
    print(np.asarray(incomplete_pcds).shape)
    print(np.asarray(labels).shape)
    with h5py.File(f"{GOAL_DATA_PATH}/{f_name}.h5", "w") as f:
        dset = f.create_dataset("complete_pcds", data=np.asarray(complete_pcds, dtype='<f4'))
        dset = f.create_dataset("incomplete_pcds", data=np.asarray(incomplete_pcds, dtype='<f4'))
        dset = f.create_dataset("labels", data=np.asarray(labels, dtype='<f4'))
        print(f.keys())

def show_dataset(f_name):
    f = h5py.File(f'{GOAL_DATA_PATH}/{f_name}.h5', 'r')
    print(f.name, f.keys())
    for k in f.keys():
        print(k)
        for pcd in f[k][:2]:
            try:
                o3dpcd = nparray2o3dpcd(np.asarray(pcd))
                o3d.visualization.draw_geometries([o3dpcd])
            except:
                break

if __name__ == '__main__':
    gen_h5('train')
    gen_h5('test')
    gen_h5('val')
    show_dataset('train')
    show_dataset('test')
    show_dataset('val')

