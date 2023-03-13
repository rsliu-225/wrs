import pickle
import os

import numpy as np

import basis.robot_math as rm
import config
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd
import nbv.nbv_utils as nu

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    fo = 'nbc_opt/extrude_1'

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.016)

    x_range = (.1, .2)
    y_range = (-.15, .02)
    z_range = (-.1, -.02)

    theta = None
    max_a = np.pi / 18
    arrow_len = .04

    rbt = el.loadXarm(showrbt=False)
    # gm.gen_frame().attach_to(base)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    seedjntagls = rbt.get_jnt_values()

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    gm.gen_frame().attach_to(base)
    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    # gl_relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    textureimg, depthimg, pcd = rcu.load_frame(fo, f_name='000.pkl')
    # pcd, _, textureimg = pickle.load(open(os.path.join(config.ROOT, 'img/zivid', fo, '000.pkl'), 'rb'))

    # cv2.imshow("grayimg", textureimg)
    # cv2.waitKey(0)

    pcd_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    cam_mat4 = np.linalg.inv(gripperframe)
    cam_mat4 = np.dot(gl_transmat4, cam_mat4)
    cam_mat4 = np.dot(cam_mat4, rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((1, 0, 0), np.pi / 2)))
    cam_pos = cam_mat4[:3, 3]
    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    pcdu.show_pcd(pcd_gl, rgba=(.5, .5, .5, .5))
    pcdu.show_pcd(pcd_roi, rgba=(1, 1, 0, 1))
    pcdu.show_cam(cam_mat4)

    # pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, _ = \
    #     rcu.cal_nbc(pcd_roi, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
    #                 theta=theta, max_a=max_a, toggledebug_p3d=True, toggledebug=False)
    # pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
    #     rcu.cal_nbc_pcn(pcd_roi, gripperframe, rbt, center=center, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
    #                     theta=theta, max_a=max_a, toggledebug_p3d=True, toggledebug=False)
    # pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
    #     rcu.cal_nbc_pcn_opt(pcd_roi, gripperframe, rbt, center=center, seedjntagls=seedjntagls,
    #                         gl_transmat4=gl_transmat4, theta=theta, toggledebug=False)
    # pickle.dump([pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn], open(os.path.join(f'tmp_res.pkl'), 'wb'))

    pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = pickle.load(open(os.path.join(f'tmp_res.pkl'), 'rb'))
    m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 1, 0, .5))
    m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))

    path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    m_planner.ah.show_ani(path)
    base.run()
