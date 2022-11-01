import h5py
import os
import open3d as o3d
import numpy as np
import basis.o3dhelper as o3dh

ROOT = os.path.abspath('./')

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255


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


def show_dataset_o3d(cat='plat'):
    for fo in os.listdir(org_path):
        if fo == cat:
            for f in os.listdir(os.path.join(org_path, fo, 'complete')):
                if f[-3:] == 'pcd':
                    print(f)
                    o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'complete', f))
                    o3dpcd_i = o3d.io.read_point_cloud(os.path.join(org_path, fo, 'partial', f))
                    # gm.gen_pointcloud(np.asarray(o3dpcd_gt.points)).attach_to(base)
                    o3dpcd_i.paint_uniform_color(COLOR[0])
                    o3dpcd_gt.paint_uniform_color(COLOR[1])
                    o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_i])


if __name__ == '__main__':
    import visualization.panda.world as wd

    org_path = 'E:/liu/dataset_2048_prim/'
    goal_path = 'D:/liu/MVP_Benchmark/completion/data_2048_prim/'

    show_dataset_h5(goal_path, 'train', label=3)
    # show_dataset_h5('test')
    # show_dataset_h5('val')
    # show_dataset_o3d(cat='multiview')
    # base.run()
