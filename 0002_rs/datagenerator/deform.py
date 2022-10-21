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
import bendplanner.bend_utils as bu
import utils.recons_utils as ru


def random_rot_radians(n=3):
    rot_axial = []
    rot_radial = []
    for i in range(n):
        rot_axial.append(random.randint(10, 30) * random.choice([1, -1]))
        rot_radial.append(random.randint(0, 1) * random.choice([1, -1]))
    return np.radians(rot_axial), np.radians(rot_radial)


if __name__ == '__main__':

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    gm.gen_frame(thickness=.002, length=.05).attach_to(base)

    width = .008
    thickness = 0
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    objcm = cm.CollisionModel('../obstacles/template.stl')
    fo = 'tst_plate'
    # objcm.attach_to(base)
    # objcm.set_rgba((.7, .7, .7, .7))

    vs = objcm.objtrm.vertices

    for i in range(3):
        # goal_pseq = np.asarray([[0, 0, 0],
        #                         [.04 + random.uniform(-.02, .01), 0, random.uniform(-.005, .005)],
        #                         [.08 + random.uniform(-.01, .01), 0, random.uniform(-.01, .01)],
        #                         [.12 + random.uniform(-.01, .02), 0, random.uniform(-.005, .005)],
        #                         [.16, 0, 0]])
        # goal_pseq = np.asarray([[0, 0, 0],
        #                         [.04 + random.uniform(-.02, .04), 0, random.uniform(-.015, .015)],
        #                         [.12 + random.uniform(-.04, .02), 0, random.uniform(-.015, .015)],
        #                         [.16, 0, 0]])
        goal_pseq = np.asarray([[0, 0, 0],
                                [.08 + random.uniform(-.02, .02), 0,
                                 random.uniform(.015, .02) * random.choice([-1, 1])],
                                [.16, 0, 0]])
        # goal_pseq = du.uni_length(goal_pseq, goal_len=.128)
        # goal_pseq = du.uni_length(goal_pseq, goal_len=.143)
        rot_axial, rot_radial = random_rot_radians(len(goal_pseq))

        deformed_objcm, objcm_gt = du.deform_cm(objcm, goal_pseq, rot_axial, rot_radial)

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
            du.get_objpcd_partial_o3d(deformed_objcm, objcm, rot, rot_center, path=fo,
                                      f_name=f'{obj_id}_{str(cnt).zfill(3)}',
                                      occ_vt_ratio=random.uniform(.05, .1), noise_vt_ratio=random.uniform(.5, 1),
                                      add_noise=True, add_occ=True, toggledebug=True)
            homomat4_dict[str(obj_id)][str(cnt).zfill(3)] = rm.homomat_from_posrot(rot_center, rot)
            cnt += 1
            pickle.dump(homomat4_dict, open(f'{fo}/homomat4_dict.pkl', 'wb'))

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
