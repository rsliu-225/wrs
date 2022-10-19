import os
import random

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json


def custom_draw_geometry_with_camera_trajectory(pcd, path='./tst', f_name=''):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = \
        o3d.io.read_pinhole_camera_trajectory("./param/camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    if not os.path.exists(f"{path}/rgbimg/"):
        os.makedirs(f"{path}/rgbimg")
    if not os.path.exists(f"{path}/depthimg/"):
        os.makedirs(f"{path}/depthimg")

    def move_forward(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capture image {:03d}".format(glb.index))
            vis.capture_depth_point_cloud(os.path.join(path, f"{str(glb.index).zfill(3)}.pcd"), do_render=False,
                                          convert_to_world_coordinate=True)
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(path, 'depthimg', f"{str(glb.index).zfill(3)}.png"), np.asarray(depth), dpi=1)
            plt.imsave(os.path.join(path, 'rgbimg', f"{str(glb.index).zfill(3)}.png"), np.asarray(image), dpi=1)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("./param/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()


if __name__ == "__main__":
    import modeling.geometric_model as gm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    # o3dmesh = o3d.io.read_triangle_mesh('./tst/mesh/0_000.ply')
    # custom_draw_geometry_with_camera_trajectory(o3dmesh)
    for f in os.listdir('./tst/'):
        if f[-3:] == 'pcd':
            pcd = o3d.io.read_point_cloud(f"./tst/{f}")
            pcd_cm = gm.gen_pointcloud(np.asarray(pcd.points),
                                       rgbas=[[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1]])
            pcd_cm.attach_to(base)

    base.run()
