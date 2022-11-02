import os
import open3d as o3d
import modeling.geometric_model as gm


def show_sgl(folder_name='plate_a_cubic'):
    for f in sorted(os.listdir(folder_name)):
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, f))
            print(len(o3dpcd.points))
            o3d.visualization.draw_geometries([o3dpcd])


def show_recons(folder_name='plate_a_cubic'):
    for f in sorted(os.listdir(folder_name)):
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, f))
            o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30))
            o3d.visualization.draw_geometries([o3dpcd])
            # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3dpcd, depth=5, width=0)
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3dpcd,
                                                                                   o3d.utility.DoubleVector(radii))
            o3d.visualization.draw_geometries([mesh, o3dpcd])


def show_sgl_p3d(folder_name='plate_a_cubic', fname=None):
    gm.gen_frame(length=.5).attach_to(base)
    for f in sorted(os.listdir(folder_name)):
        if f == fname or fname is None:
            o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, f))
            gm.gen_pointcloud(o3dpcd.points).attach_to(base)


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
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])
    # show_sgl(folder_name='seq/plate_cubic_nbc')
    show_recons(folder_name='seq/plate_cubic_nbc')
    # show_sgl_p3d(folder_name='./test output(model-300000)/cubic/partial', fname=None)
    # base.run()
    # show_tstngt(tst_folder_name='./test output(model-300000)/output(model-300000)/cubic',
    #             gt_folder_name='./test output(model-300000)/cubic',
    #             f_name=None)
