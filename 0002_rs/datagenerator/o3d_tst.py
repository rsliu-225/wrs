import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    vis.get_render_option().load_from_json("./TestData/renderoption.json")
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(geo):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([geo], rotate_view)


def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = \
        o3d.io.read_pinhole_camera_trajectory("./TestData/camera_trajectory.json")
    print(custom_draw_geometry_with_camera_trajectory.trajectory)
    print(custom_draw_geometry_with_camera_trajectory.trajectory.parameters[0].intrinsic.intrinsic_matrix)
    print(custom_draw_geometry_with_camera_trajectory.trajectory.parameters[1].intrinsic.intrinsic_matrix)

    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    if not os.path.exists("./TestData/image/"):
        os.makedirs("./TestData/image/")
    if not os.path.exists("./TestData/depth/"):
        os.makedirs("./TestData/depth/")

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave("./TestData/depth/{:05d}.png".format(glb.index), np.asarray(depth), dpi=1)
            plt.imsave("./TestData/image/{:05d}.png".format(glb.index), np.asarray(image), dpi=1)
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("./TestData/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    import config
    import utils as utl

    width = .005
    thickness = .0015
    path = './tst'
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    pseq = utl.cubic_inp(pseq=np.asarray([[0, 0, 0], [.018, .03, .02], [.06, .06, 0], [.12, 0, 0]]))
    pseq = utl.uni_length(pseq, goal_len=.2)
    rotseq = utl.get_rotseq_by_pseq(pseq)

    objcm = utl.gen_swap(pseq, rotseq, cross_sec)
    o3dmesh = utl.cm2o3dmesh(objcm)

    pcd = o3d.io.read_point_cloud(config.ROOT + "/recons_data/opti/plate_a_cubic/000.pcd")

    custom_draw_geometry_with_custom_fov(o3dmesh, -90)
    # custom_draw_geometry_with_rotation(o3dmesh)
    # custom_draw_geometry_with_camera_trajectory(pcd)
