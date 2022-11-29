import os
import pickle
import random

import numpy as np
import open3d as o3d

import basis.robot_math as rm
import basis.trimesh as trm
import datagenerator.data_utils as du
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd
import math
import utils.pcd_utils as pcdu

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
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    gm.gen_frame(thickness=.002, length=.05).attach_to(base)

    width = .008
    thickness = 0
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    objcm = cm.CollisionModel('../obstacles/plate.stl')
    fo = 'tst_plate'
    # objcm.attach_to(base)
    # objcm.set_rgba((.7, .7, .7, .7))

    vs = objcm.objtrm.vertices

    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.04 + random.uniform(-.02, .01), 0, random.uniform(-.01, .01)],
    #                         [.08 + random.uniform(-.01, .01), 0, random.uniform(-.02, .02)],
    #                         [.12 + random.uniform(-.01, .02), 0, random.uniform(-.01, .01)],
    #                         [.16, 0, 0]])
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.04, 0, -.005],
    #                         [.08, 0, .01],
    #                         [.12, 0, .005],
    #                         [.16, 0, 0]])
    goal_pseq = np.asarray([[0, 0, 0],
                            [.04 + random.uniform(-.02, .04), 0, random.uniform(-.03, .03)],
                            [.12 + random.uniform(-.04, .02), 0, random.uniform(-.03, .03)],
                            [.16, 0, 0]])
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.04 + random.uniform(-.02, .04), 0, .03],
    #                         [.12 + random.uniform(-.04, .02), 0, -.03],
    #                         [.16, 0, 0]])
    print(goal_pseq)
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.08 + random.uniform(-.02, .02), 0,
    #                          random.uniform(.03, .04) * random.choice([-1, 1])],
    #                         [.16, 0, 0]])
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.08 + random.uniform(-.02, .02), 0,
    #                          random.uniform(.03, .04) * random.choice([-1, 1])],
    #                         [.16, 0, 0]])
    # icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    # icomats = [x for row in icomats for x in row]
    # rot = random.choice(icomats)

    rot_axial, rot_radial = du.random_rot_radians(len(goal_pseq))
    deformed_objcm, objcm_gt, _, _ = du.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, show=True)

    '''
    gen data
    '''
    rot_center = (0, 0, 0)
    icomats = rm.gen_icorotmats(rotation_interval=np.radians(90))
    cnt = 0
    obj_id = 0
    # rot = np.eye(3)
    rot = rm.rotmat_from_axangle((1, 0, 0), np.pi / 3)

    du.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, rot, rot_center, path=fo,
                              f_name=f'{obj_id}_{str(cnt).zfill(3)}',
                              visible_threshold=np.radians(60),
                              rnd_occ_ratio_rng=(.2, .5), occ_vt_ratio=random.uniform(.05, .1),
                              noise_vt_ratio=random.uniform(.2, 5), noise_cnt=3,
                              add_occ=True, add_noise=True, add_rnd_occ=False, add_noise_pts=True,
                              savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)

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
