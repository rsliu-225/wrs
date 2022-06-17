import os
import open3d as o3d


def show_sgl(folder_name='plate_a_cubic'):
    for f in sorted(os.listdir(folder_name)):
        if f[-3:] != 'pcd':
            continue
        o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, f))
        o3d.visualization.draw_geometries([o3dpcd])


def show_tstngt(tst_folder_name='./test output(model-300000)/output(model-300000)/cubic',
                gt_folder_name='./test output(model-300000)/cubic',
                f_name=None):
    for f in sorted(os.listdir(tst_folder_name)):
        if f[-3:] != 'pcd':
            continue
        if f_name is not None:
            if f != f_name:
                continue
        o3dpcd = o3d.io.read_point_cloud(os.path.join(tst_folder_name, f))
        o3dpcd.paint_uniform_color([0, 0.706, 1])
        # o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(gt_folder_name, 'complete', f))
        # o3dpcd_gt.paint_uniform_color([1, 0.706, 0])
        o3dpcd_input = o3d.io.read_point_cloud(os.path.join(gt_folder_name, 'partial', f))
        o3dpcd_input.paint_uniform_color([1, 0.706, 0])

        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_input])


if __name__ == '__main__':
    show_tstngt(tst_folder_name='./test output(model-300000)/output(model-300000)/cubic',
                gt_folder_name='./test output(model-300000)/cubic',
                f_name=None)
