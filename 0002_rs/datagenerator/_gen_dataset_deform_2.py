import os
import pickle
import random

import numpy as np
import open3d as o3d

import basis.robot_math as rm
import basis.trimesh as trm
import datagenerator.data_utils as utl
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd
import math
import utils.pcd_utils as pcdu

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def save_stl(objcm, path):
    from stl import mesh
    vertices = np.asarray(objcm.objtrm.vertices) * 1000
    faces = np.asarray(objcm.objtrm.faces)
    objmesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            objmesh.vectors[i][j] = vertices[f[j], :]
    objmesh.save(path)


if __name__ == '__main__':
    import basis.o3dhelper as o3dh

    base = wd.World(cam_pos=[.1, .25, .25], lookat_pos=[.1, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    gm.gen_frame(thickness=.002, length=.05).attach_to(base)

    objcm = cm.CollisionModel('../obstacles/plate.stl')
    # objcm = cm.gen_box(np.asarray([.2, .01, .002]))

    fo = 'tst_plate'
    objcm.attach_to(base)
    objcm.set_rgba((.7, .7, .7, 1))

    # base.run()
    # o3dcm = o3dh.cm2o3dmesh(objcm)
    # o3dcm.compute_vertex_normals()
    # r = rm.rotmat_from_axangle(axis=(1, 0, 0), angle=-np.radians(50))
    # o3dcm.rotate(r)
    # o3d.visualization.draw_geometries([o3dcm])

    deformed_objcm, objcm_gt, _, _ = utl.gen_seed_deform(objcm, num_kpts=4, max=.02, n=200, toggledebug=False)
    deformed_objcm.attach_to(base)
    deformed_objcm.set_rgba((.7, .7, 0, 1))
    base.run()
    '''
    gen data
    '''
    rot_center = (0, 0, 0)
    icomats = rm.gen_icorotmats(rotation_interval=np.radians(360 / 60))
    cnt = 0
    obj_id = 0
    # rot = np.eye(3)
    rot = rm.rotmat_from_axangle((1, 0, 0), np.pi / 3)

    o3dpcd_1 = utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, rot, rot_center, path=fo,
                                          f_name=f'{obj_id}_{str(cnt).zfill(3)}',
                                          visible_threshold=np.radians(60),
                                          rnd_occ_ratio_rng=(0, .2), occ_vt_ratio=random.uniform(.05, .1),
                                          noise_vt_ratio=random.uniform(.2, .5), noise_cnt=3,
                                          add_occ=True, add_noise=True, add_rnd_occ=False, add_noise_pts=True,
                                          savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)
    o3dpcd_2 = utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, icomats[0][10], rot_center, path=fo,
                                          f_name=f'{obj_id}_{str(cnt).zfill(3)}',
                                          visible_threshold=np.radians(60),
                                          rnd_occ_ratio_rng=(0, .2), occ_vt_ratio=random.uniform(.05, .1),
                                          noise_vt_ratio=random.uniform(.2, .5), noise_cnt=3,
                                          add_occ=True, add_noise=True, add_rnd_occ=False, add_noise_pts=True,
                                          savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)

    homomat4 = rm.homomat_from_posrot((0, 0, 0), icomats[0][10])
    pcd = np.asarray(o3dpcd_2.points)
    pcd = utl.trans_pcd(pcd, np.dot(rm.homomat_from_posrot((0, 0, 0), rot), np.linalg.inv(homomat4)))
    random_homomat4 = utl.gen_random_homomat4((.001, .001, .001), np.radians((1, 1, 1)))
    pcd = utl.trans_pcd(pcd, random_homomat4)

    o3dpcd_2 = utl.nparray2o3dpcd(pcd)
    o3dpcd_1.paint_uniform_color(COLOR[0])
    o3dpcd_2.paint_uniform_color((.7, .7, .7))
    o3d.visualization.draw_geometries([o3dpcd_1, o3dpcd_2])

    # '''
    # show data
    # '''
    # for f in sorted(os.listdir(f"{fo}/complete")):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{fo}/complete/{f}")
    #         gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 1, 0, 1]]).attach_to(base)
    # for f in sorted(os.listdir(f"{fo}/partial")):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{fo}/partial/{f}")
    #         gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    # base.run()
