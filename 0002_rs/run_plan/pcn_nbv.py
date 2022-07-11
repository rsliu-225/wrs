import os

import numpy as np
import open3d as o3d

import config
import utils.pcd_utils as pcdu
import visualization.panda.world as wd

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])

fo = 'plate_a_cubic_2'
cam_pos = [1, 1, 1]

for f in os.listdir(os.path.join(config.ROOT, 'pcd_output', fo)):
    print(f)
    if f[-3:] != 'pcd':
        continue
    op = o3d.io.read_point_cloud(os.path.join(config.ROOT, 'pcd_output', fo, f))
    ip = o3d.io.read_point_cloud(os.path.join(config.ROOT, 'recons_data', fo, f'{f.split("_")[0]}.pcd'))

    pts, nrmls, confs = \
        pcdu.cal_conf(np.asarray(op.points), voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=np.pi / 6,
                      toggledebug=True)
    nbv_pts, nbv_nrmls, nbv_confs = \
        pcdu.cal_nbv(pts, nrmls, confs, cam_pos=cam_pos, toggledebug=True)

    pts, nrmls, confs = \
        pcdu.cal_conf(np.asarray(ip.points), voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=np.pi / 6,
                      toggledebug=True)
    nbv_pts, nbv_nrmls, nbv_confs = \
        pcdu.cal_nbv(pts, nrmls, confs, cam_pos=cam_pos, toggledebug=True)

    op.paint_uniform_color([1, 1, 0])
    ip.paint_uniform_color([0, 1, 1])
    # o3d.visualization.draw_geometries([ip,op])
    base.run()
