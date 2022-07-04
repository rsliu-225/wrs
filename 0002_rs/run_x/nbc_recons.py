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
import utils.phoxi as phoxi
import visualization.panda.world as wd
import localenv.envloader as el
import motionplanner.motion_planner as mp
import motionplanner.rbtx_motion_planner as mpx

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    folder_name = 'plate_a_cubic'

    rbt = el.loadXarm(showrbt=False)
    rbtx = el.loadXarmx(ip='10.2.0.201')
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname='arm')
    m_planner_x = mpx.MotionPlannerRbtX(env=None, rbt=rbt, rbtx=rbtx, armname='arm')

    # seedjntagls = m_planner.get_armjnts()

    icp = False

    seed = (.116, 0, .08)
    center = (.116, 0, -.0155)

    x_range = (.06, .215)
    y_range = (-.15, .15)
    z_range = (.0155, .2)
    # z_range = (-.2, -.0155)
    theta = np.pi / 6
    rbtx.arm_jaw_to(0)

    f_path = 'phoxi/nbc/plate_cubic_2/'
    i = 0
    while i < 2:
        seedjntagls = m_planner_x.get_armjnts()
        pickle.dump(seedjntagls, open(config.ROOT + '/img/' + f_path + f'{str(i).zfill(3)}_armjnts.pkl', 'wb'))

        m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
        tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)

        relrot = np.asarray([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).T
        gl_transrot = np.dot(tcprot, relrot)
        gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
        gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)
        gm.gen_frame(gl_transpos, tcprot).attach_to(base)

        textureimg, depthimg, pcd = phxi.dumpalldata(f_name='img/' + f_path + f'{str(i).zfill(3)}.pkl')
        cv2.imshow('grayimg', textureimg)
        cv2.waitKey(0)

        jnts = rcu.cal_nbc(textureimg, pcd, rbt=rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4, seed=seed,
                           theta=theta, x_range=x_range, y_range=y_range, z_range=z_range)
        m_planner.ah.show_armjnts(armjnts=jnts)
        path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
        m_planner.ah.show_ani(path)
        m_planner_x.movepath(path)

        i += 1

    base.run()
    # textureimg, depthimg, pcd = phxi.dumpalldata(f_name='img/' + path + f'{str(i).zfill(3)}.pkl')
    # cv2.imshow('grayimg', textureimg)
    # cv2.waitKey(0)
    # textureimg, depthimg, pcd = phxi.loadalldata(f_name='img/' + path + f'{str(1).zfill(3)}.pkl')
    # cv2.imshow('grayimg', textureimg)
    # cv2.waitKey(0)
