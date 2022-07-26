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
    import utils.pcd_utils as pcdu

    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    # fo = 'nbc/plate_a_cubic'
    fo = 'opti/plate_a_cubic'

    rbt = el.loadXarm(showrbt=False)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    # m_planner.ah.show_armjnts(rgba=(1, 0, 0, .5))

    seedjntagls = m_planner.get_armjnts()
    # m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    # gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    # relrot = np.asarray([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.0155)

    # x_range = (.06, .215)
    x_range = (0, .215)
    y_range = (-.15, .15)
    z_range = (.0155, .2)
    # z_range = (-.2, -.0155)

    # textureimg, depthimg, pcd = rcu.load_frame(fo, f_name='000.pkl')
    # opti_data = rcu.load_opti(fo, f_name='000_opti.pkl')
    # seg = opti_data[1]
    # if any([seg.x, seg.z, seg.y]):
    #     rot = rm.rotmat_from_axangle((1, 0, 0), seg.qx) \
    #         .dot(rm.rotmat_from_axangle((0, 1, 0), seg.qy)) \
    #         .dot(rm.rotmat_from_axangle((0, 0, 1), seg.qz))
    #     homomat = rm.homomat_from_posrot([seg.x, seg.z, seg.y], rot)
    #     print(homomat)
    # cv2.imshow("grayimg", textureimg)
    # cv2.waitKey(0)

    pcd_cropped_list = rcu.reg_opti(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                    toggledebug=False, icp=False)
    gm.gen_frame().attach_to(base)
    # pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
    #                                     toggledebug=True, icp=False)

    # rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1]]
    # for i, pcd in enumerate(pcd_cropped_list):
    #     pcdu.show_pcd(pcd, rgba=rgba_list[i%2])
    base.run()

    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)
    # base.run()
