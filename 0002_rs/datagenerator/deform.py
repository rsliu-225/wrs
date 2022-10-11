import os
import pickle
import random

import numpy as np
import open3d as o3d
from pygem import RBF

import basis.robot_math as rm
import basis.trimesh as trm
import datagenerator.utils as utl
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd


def gen_plate_ctr_pts(pts, goal_pseq, edge=0.0):
    org_len = np.linalg.norm(pts[:, 0].max() - pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_diff_uni = org_len * goal_diff / goal_diff.sum()
    # x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), num)
    x = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min()) - edge
    y_max = float(pts[:, 1].max()) + edge
    z_min = float(pts[:, 2].min()) - edge
    z_max = float(pts[:, 2].max()) + edge
    ctr_pts = [[x, y_min, z_min],
               [x, y_max, z_min],
               [x, y_max, z_max],
               [x, y_min, z_max]]
    for i in range(len(goal_diff_uni)):
        x += goal_diff_uni[i]
        ctr_pts.extend([[x, y_min, z_min],
                        [x, y_max, z_min],
                        [x, y_max, z_max],
                        [x, y_min, z_max]])

    return np.asarray(ctr_pts)


def gen_deformed_ctr_pts(ctr_pts, goal_pseq, rot_diff=None):
    goal_rotseq = utl.get_rotseq_by_pseq_1d(goal_pseq)
    org_len = np.linalg.norm(ctr_pts[:, 0].max() - ctr_pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_pseq = org_len * goal_pseq / goal_diff.sum()

    org_kpts = ctr_pts.reshape((int(len(ctr_pts) / 4), 4, 3)).mean(axis=1)
    deformed_ctr_pts = []
    if len(ctr_pts) != len(goal_pseq) * 4:
        print('Wrong goal_diff size!', ctr_pts.shape, goal_pseq.shape)
        return None
    for i in range(len(goal_pseq)):
        gm.gen_frame(goal_pseq[i], goal_rotseq[i], length=.01, thickness=.001).attach_to(base)
        gm.gen_frame(org_kpts[i], np.eye(3), length=.01, thickness=.001,
                     rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        transmat4 = np.dot(rm.homomat_from_posrot(goal_pseq[i], goal_rotseq[i]),
                           np.linalg.inv(rm.homomat_from_posrot(org_kpts[i], np.eye(3))))
        deformed_ctr_pts.extend(utl.trans_pcd(ctr_pts[i * 4:(i + 1) * 4], transmat4))

    if rot_diff is not None:
        deformed_ctr_pts_rot = []
        for i in range(len(rot_diff)):
            deformed_ctr_pts_rot.extend(utl.rot_new_orgin(deformed_ctr_pts[i * 4:(i + 1) * 4],
                                                          goal_pseq[i],
                                                          rm.rotmat_from_axangle(goal_rotseq[i][:, 0], rot_diff[i])))
        deformed_ctr_pts = np.copy(deformed_ctr_pts_rot)

    for p in ctr_pts:
        gm.gen_sphere(p, radius=.001, rgba=(1, 1, 0, 1)).attach_to(base)
    for p in deformed_ctr_pts:
        gm.gen_sphere(p, radius=.001, rgba=(1, 0, 0, 1)).attach_to(base)
    return np.asarray(deformed_ctr_pts)


if __name__ == '__main__':
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])

    objcm = cm.CollisionModel('../obstacles/plate.stl')
    folder_name = 'tst_plate'
    objcm.attach_to(base)
    # objcm.set_rgba((.7, .7, .7, .7))
    # gm.gen_frame(length=.02, thickness=.001).attach_to(base)

    vs = objcm.objtrm.vertices
    radius = .05
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.04 + random.uniform(-.005, .005), 0, random.uniform(0, .005)],
    #                         [.08 + random.uniform(-.005, .005), 0, random.uniform(-.005, .005)],
    #                         [.12 + random.uniform(-.005, .005), 0, random.uniform(-.005, 0)],
    #                         [.16, 0, 0]])
    goal_pseq = np.asarray([[0, 0, 0],
                            [.04 + random.uniform(-.01, .01), 0, random.uniform(0, .003)],
                            [.08 + random.uniform(-.005, .005), 0, random.uniform(.004, .01)],
                            [.12 + random.uniform(-.01, .01), 0, random.uniform(0, .003)],
                            [.16, 0, 0]])
    # goal_pseq = np.asarray([[0, 0, 0],
    #                         [.08 + random.uniform(-.02, .02), 0, random.uniform(.004, .01)],
    #                         [.16, 0, 0]])
    goal_pseq = utl.uni_length(goal_pseq, goal_len=.16)
    original_ctr_pts = gen_plate_ctr_pts(vs, goal_pseq)

    # rot_diff = np.radians([0, 5, 10, -10, 0])
    rot_diff = None

    deformed_ctr_pts = gen_deformed_ctr_pts(original_ctr_pts, goal_pseq, rot_diff)
    rbf = RBF(original_control_points=original_ctr_pts, deformed_control_points=deformed_ctr_pts, radius=radius)

    new_vs = rbf(vs)
    deformed_objtrm = trm.Trimesh(vertices=np.asarray(new_vs), faces=objcm.objtrm.faces)
    deformed_objcm = cm.CollisionModel(initor=deformed_objtrm, btwosided=True, name='plate_deform')
    deformed_objcm.set_rgba((.7, .7, 0, 1))
    deformed_objcm.attach_to(base)
    gm.gen_pointcloud(new_vs).attach_to(base)

    base.run()

    '''
    gen data
    '''
    rot_center = (0, 0, 0)
    icomats = rm.gen_icorotmats(rotation_interval=np.radians(90))
    cnt = 0
    obj_id = 0

    homomat4_dict = {}
    homomat4_dict[str(obj_id)] = {}
    for i, mats in enumerate(icomats):
        for j, rot in enumerate(mats):
            utl.get_objpcd_partial_o3d(deformed_objcm, rot, rot_center, path=folder_name,
                                       f_name=f'{obj_id}_{str(cnt).zfill(3)}',
                                       occ_vt_ratio=random.uniform(.05, .1), noise_vt_ratio=random.uniform(.5, 1),
                                       add_noise=True, add_occ=True, toggledebug=True)
            homomat4_dict[str(obj_id)][str(cnt).zfill(3)] = rm.homomat_from_posrot(rot_center, rot)
            cnt += 1
            pickle.dump(homomat4_dict, open(f'{folder_name}/homomat4_dict.pkl', 'wb'))

    # '''
    # show data
    # '''
    # for f in sorted(os.listdir(folder_name)):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{folder_name}/{f}")
    #         gm.gen_pointcloud(o3dpcd.points).attach_to(base)
    # base.run()
