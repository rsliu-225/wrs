import os
import cv2

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


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
    import utils as utl
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])

    width = .005
    thickness = .0015
    path = './tst'

    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    resolusion = (1280, 720)

    pseq = utl.cubic_inp(pseq=np.asarray([[0, 0, 0], [.018, .03, .02], [.06, .06, 0], [.12, 0, 0]]))
    pseq = utl.uni_length(pseq, goal_len=.2)
    rotseq = utl.get_rotseq_by_pseq(pseq)

    objcm = utl.gen_swap(pseq, rotseq, cross_sec)
    o3dmesh = utl.cm2o3dmesh(objcm)
    objcm.attach_to(base)

    vis = o3d.visualization.Visualizer()
    ctr = o3d.visualization.ViewControl()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    vis.add_geometry(o3dmesh)

    # vis.get_render_option().load_from_json("./renderoption.json")
    init_param = ctr.convert_to_pinhole_camera_parameters()
    print(init_param.intrinsic)
    print('extrinsic', init_param.extrinsic)
    # w, h = 4000, 3000
    # K = np.asarray([[0.744375, 0.0, 0.0],
    #                 [0.0, 0.744375, 0.0],
    #                 [0.4255, 0.2395, 1.0]])
    # fx = K[0, 0]
    # fy = K[1, 1]
    # cx = K[0, 2]
    # cy = K[1, 2]
    # init_param.intrinsic.width = w
    # init_param.intrinsic.height = h
    # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    init_param.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(init_param)
    vis.poll_events()

    ctr.rotate(10, 0)
    image = vis.capture_screen_float_buffer()
    cv2.imshow('', cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    vis.capture_depth_point_cloud(os.path.join(path, 'tst_partial_org.pcd'), do_render=False,
                                  convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f'tst_partial_org.pcd'))

    gm.gen_pointcloud(np.asarray(o3dpcd.points)).attach_to(base)
    base.run()
    # custom_draw_geometry_with_custom_fov(o3dmesh, -90)
    # custom_draw_geometry_with_rotation(o3dmesh)
    # custom_draw_geometry_with_camera_trajectory(pcd)
