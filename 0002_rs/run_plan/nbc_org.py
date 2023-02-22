import os
import pickle

import numpy as np
import open3d as o3d

import basis.robot_math as rm
import config
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    # fo = 'nbc/plate_a_cubic'
    fo = 'nbc_pcn/plate_a_cubic'
    # fo = 'opti/plate_a_quadratic'
    # fo = 'seq/plate_a_quadratic'

    rbt = el.loadXarm(showrbt=False)
    # gm.gen_frame().attach_to(base)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    seedjntagls = pickle.load(open(config.ROOT + '/img/phoxi/nbc/plate_a_cubic/000_armjnts.pkl', 'rb'))

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    gl_relrot = np.asarray([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).T
    # gl_relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.0155)

    x_range = (.08, .215)
    y_range = (-.15, .15)
    z_range = (.0155, .2)
    # z_range = (-.2, -.0155)

    theta = np.pi / 6
    max_a = np.pi / 90

    textureimg, depthimg, pcd = rcu.load_frame(fo, f_name='000.pkl')
    pcd = np.asarray(pcd) / 1000

    # cv2.imshow("grayimg", textureimg)
    # cv2.waitKey(0)

    pcd_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    pcdu.show_pcd(pcd_gl, rgba=(1, 0, 0, 1))
    pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

    cam_pos = np.linalg.inv(gripperframe)[:3, 3]
    # pts, nrmls, confs = pcdu.cal_conf(pcd_roi, voxel_size=.005, radius=.005, cam_pos=cam_pos, theta=np.pi/6,
    #                                   toggledebug=True)
    # pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs, cam_pos=np.linalg.inv(gripperframe)[:3, 3],
    #                                              toggledebug=True)
    nbv_pts, nbv_nrmls, jnts = \
        rcu.cal_nbc(pcd_roi, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
                    theta=theta, max_a=max_a, show_cam=True, toggledebug=True)

    m_planner.ah.show_armjnts(armjnts=seedjntagls)
    m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))
    # path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    # m_planner.ah.show_ani(path)
    base.run()

    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)
    # base.run()
