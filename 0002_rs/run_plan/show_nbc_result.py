import pickle
import cv2

import numpy as np

import basis.robot_math as rm
import config
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as mp
import nbv.nbv_utils as nu
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd

import basis.o3dhelper as o3dh
import open3d as o3d


def show_nbv_conf(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len):
    # nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + nrmls_nbv[0] / np.linalg.norm(nrmls_nbv[0]) * arrow_len,
                 rgba=(0, 0, 1, 1), thickness=0.005).attach_to(base)
    gm.gen_dashstick(cam_pos, pts_nbv[0], rgba=(.7, .7, 0, .5), thickness=.005).attach_to(base)
    # for i in range(len(confs_nbv)):
    #     gm.gen_sphere(pts_nbv[i], radius=.01, rgba=[confs_nbv[i], 0, 1 - confs_nbv[i], .2]).attach_to(base)


def show_nbv(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len,thickness=0.005):
    # nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + nrmls_nbv[0] / np.linalg.norm(nrmls_nbv[0]) * arrow_len,
                 rgba=(0, 0, 1, 1), thickness=thickness).attach_to(base)
    gm.gen_dashstick(cam_pos, pts_nbv[0], rgba=(.7, .7, 0, .5), thickness=thickness).attach_to(base)


if __name__ == '__main__':
    base = wd.World(cam_pos=[.8, 1, 1.4], lookat_pos=[.5, 0, 1])
    # base = wd.World(cam_pos=[.1, -.5, -.05], lookat_pos=[.1, 0, -.05])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    method = ''

    fo = 'nbc_pcn/extrude_1_woef'

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.016)

    x_range = (.1, .2)
    y_range = (-.15, .02)
    z_range = (-.1, -.02)

    theta = None
    max_a = np.pi / 90
    arrow_len = .06

    rbt = el.loadXarm(showrbt=False)
    gm.gen_frame().attach_to(base)
    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    seedjntagls = rbt.get_jnt_values()
    pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pts_pcn = \
        pickle.load(open(config.ROOT + f'/img/phoxi/{fo}/001_res.pkl', 'rb'))

    textureimg, _, pcd = rcu.load_frame(fo, f_name='000.pkl')
    textureimg_nxt, _, pcd_nxt = rcu.load_frame(fo, f_name='001.pkl')

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    # gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)
    # cv2.imshow("grayimg", textureimg)
    # cv2.waitKey(0)

    pts_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    pts_roi_nxt, pcd_trans_nxt, gripperframe_nxt = \
        rcu.extract_roi_by_armarker(textureimg_nxt, pcd_nxt, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    cam_mat4 = np.dot(gl_transmat4, np.linalg.inv(gripperframe))
    cam_pos = cam_mat4[:3, 3]

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
    pts_nbv_origin = pcdu.trans_pcd(pts_nbv, np.linalg.inv(gripperframe))
    nrmls_nbv_origin = pcdu.trans_pcd(nrmls_nbv, np.linalg.inv(gripperframe))

    # if method == 'pcn':
    #     pcdu.cal_nbv_pcn(pts_roi, pts_pcn, cam_pos, radius=.01, toggledebug=True)
    # else:
    #     nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.02)
    #     pts_roi = pcdu.trans_pcd(pts_roi, gl_transmat4)
    #     pcdu.show_pcd(pts_roi, rgba=(nu.COLOR[0][0], nu.COLOR[0][1], nu.COLOR[0][2], 1))
    # base.run()

    pcdu.show_cam(cam_mat4)
    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    pcdu.show_pcd(pcd_gl, rgba=(.7, .7, .7, .5))
    m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(.7, .7, .7, .5))
    # path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    # m_planner.ah.show_ani(path)

    show_nbv(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.06)
    # base.run()
    pcd_cropped_new = pcdu.trans_pcd(pcd_gl, transmat4)
    pts_nbv_new = pcdu.trans_pcd(pts_nbv, transmat4)
    nrmls_nbv_new = [transmat4[:3, :3].dot(n) for n in nrmls_nbv]
    pcd_gl_nxt = pcdu.trans_pcd(pcd_trans_nxt, np.dot(transmat4, gl_transmat4))
    pcdu.show_pcd(pcd_gl_nxt, rgba=(0, .7, 0, .5))
    m_planner.ah.show_armjnts(armjnts=jnts)
    show_nbv(pts_nbv_new[:1], nrmls_nbv_new[:1], confs_nbv, cam_pos,  arrow_len=.06)
    base.run()
