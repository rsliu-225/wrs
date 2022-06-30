import copy
import os
import pickle

import cv2
import numpy as np
import open3d as o3d
from cv2 import aruco as aruco

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import utils.recons_utils as rcu
import visualization.panda.world as wd
import localenv.envloader as el
import motionplanner.motion_planner as mp

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    fo = 'plate_cubic_2'

    rbt = el.loadXarm(showrbt=False)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    m_planner.ah.show_armjnts(rgba=(1, 0, 0, .5))

    gm.gen_frame(rbt.arm.lnks[-1]['gl_pos'] + np.asarray([.03466 + .087, 0, 0]),
                 rbt.arm.lnks[-1]['gl_rotmat']).attach_to(base)

    gl_transrot = rm.rotmat_from_axangle((1, 0, 0), np.pi)
    gl_transpos = rbt.arm.lnks[-1]['gl_pos'] + np.asarray([.03466 + .087, 0, 0])
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.0155)

    pcd_cropped_list = rcu.reg_plate(fo, seed, center)
    base.run()

    textureimg, depthimg, pcd = rcu.load_frame(fo, f_name='000.pkl')
    cv2.imshow("grayimg", textureimg)
    cv2.waitKey(0)

    seedjntagls = rbt.get_jnt_values('arm')
    jnts = rcu.cal_nbc(textureimg, pcd, rbt=rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4)
    m_planner.ah.show_armjnts(armjnts=jnts)
    path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    m_planner.ah.show_ani(path)
    base.run()

    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)
    # base.run()
