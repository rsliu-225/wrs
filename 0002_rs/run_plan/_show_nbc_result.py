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


def show_nbv_conf(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len, thickness=0.005):
    # nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + nrmls_nbv[0] / np.linalg.norm(nrmls_nbv[0]) * arrow_len,
                 rgba=(0, 0, 1, 1), thickness=0.005).attach_to(base)
    gm.gen_dashstick(cam_pos, pts_nbv[0], rgba=(.7, .7, 0, .5), thickness=thickness).attach_to(base)
    for i in range(len(confs_nbv)):
        if confs_nbv[i] > .4:
            continue
        gm.gen_sphere(pts_nbv[i], radius=.01, rgba=[confs_nbv[i], 0, 1 - confs_nbv[i], .2]).attach_to(base)


def show_nbv(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len, thickness=0.005):
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + nrmls_nbv[0] / np.linalg.norm(nrmls_nbv[0]) * arrow_len,
                 rgba=(0, 0, 1, 1), thickness=thickness).attach_to(base)
    gm.gen_dashstick(cam_pos, pts_nbv[0], rgba=(.7, .7, 0, .5), thickness=thickness).attach_to(base)


def show_nbv_all(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len, thickness=0.005):
    nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len, thickness)
    gm.gen_arrow(pts_nbv[0], pts_nbv[0] + nrmls_nbv[0] / np.linalg.norm(nrmls_nbv[0]) * arrow_len,
                 rgba=(0, 0, 1, 1), thickness=thickness).attach_to(base)
    gm.gen_dashstick(cam_pos, pts_nbv[0], rgba=(.7, .7, 0, .5), thickness=thickness).attach_to(base)


if __name__ == '__main__':
    # base = wd.World(cam_pos=[1, -3, 2], lookat_pos=[.4, 0, 1])
    base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    method = 'pcn'

    fo = 'nbc_pcn/extrude_1'

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
    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    step = 2
    if step != 1:
        _, _, _, _, seedjntagls, _ = \
            pickle.load(open(config.ROOT + f'/img/phoxi/{fo}/{str(step - 1).zfill(3)}_res.pkl', 'rb'))
    else:
        seedjntagls = rbt.get_jnt_values()

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    # gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)
    # cv2.imshow("grayimg", textureimg)
    # cv2.waitKey(0)
    pts_roi = []
    gripperframe = np.eye(4)
    for i in range(step):
        textureimg, _, pcd = rcu.load_frame(fo, f_name=f'{str(i).zfill(3)}.pkl')
        pts_roi_tmp, pcd_trans, gripperframe = \
            rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                        x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
        pts_roi.extend(list(pts_roi_tmp))
    pts_roi = np.asarray(pts_roi)

    pts_nbv, nrmls_nbv, confs_nbv, _, jnts, pts_pcn = \
        pickle.load(open(config.ROOT + f'/img/phoxi/{fo}/{str(step).zfill(3)}_res.pkl', 'rb'))
    textureimg_nxt, _, pcd_nxt = rcu.load_frame(fo, f_name=f'{str(step).zfill(3)}.pkl')
    pts_roi_nxt, pcd_trans_nxt, gripperframe_nxt = \
        rcu.extract_roi_by_armarker(textureimg_nxt, pcd_nxt, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    cam_mat4 = np.dot(np.linalg.inv(gripperframe),
                      rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((1, 0, 0), np.pi / 2)))
    cam_mat4 = np.dot(gl_transmat4, cam_mat4)
    cam_pos = cam_mat4[:3, 3]

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
    pts_nbv_origin = pcdu.trans_pcd(pts_nbv, np.linalg.inv(gl_transmat4))
    nrmls_nbv_origin = pcdu.trans_pcd(nrmls_nbv, np.linalg.inv(gl_transmat4))

    if method == 'pcn':
        pcdu.show_pcd(pts_pcn, rgba=(nu.COLOR[2][0], nu.COLOR[2][1], nu.COLOR[2][2], 1))
        pcdu.cal_nbv_pcn(pts_roi, pts_pcn, cam_pos, radius=.01, toggledebug=True)
    elif method == 'opt':
        # gm.gen_frame(rotmat=rm.rotmat_from_axangle((0, 1, 0), -np.pi / 2), length=.03, thickness=.002).attach_to(base)
        pts_pcn = pcdu.trans_pcd(pts_pcn,
                                 rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((0, 1, 0), np.pi / 2)))
        pts_pcn = np.asarray(pts_pcn) + center
        # pcdu.show_pcd(pts_pcn, rgba=(nu.COLOR[2][0], nu.COLOR[2][1], nu.COLOR[2][2], 1))
        # nu.attach_nbv_gm(pts_nbv_origin, nrmls_nbv_origin, confs_nbv, cam_pos, arrow_len=.02)
        pcdu.cal_nbv_pcn(pts_roi, pts_pcn, cam_pos, radius=.01, toggledebug=True)
    else:
        pts, nrmls, confs = pcdu.cal_conf(pts_roi, voxel_size=.005, cam_pos=cam_pos, theta=None)
        pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)
        nu.attach_nbv_gm(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.02, thickness=.001)
    pcdu.show_pcd(pts_roi, rgba=(nu.COLOR[0][0], nu.COLOR[0][1], nu.COLOR[0][2], 1))
    # show_nbv_all(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.02, thickness=0.002)
    # show_nbv_conf(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.02, thickness=0.002)
    base.run()

    '''
    show init
    '''
    pcdu.show_cam(cam_mat4)
    # m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(.6, .6, .6, .5))
    initpos, initrot = m_planner.rbt.get_gl_tcp()
    initmat4 = rm.homomat_from_posrot(initpos, initrot)

    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    # pcdu.show_pcd(pcd_gl, rgba=(.6, .6, .6, .5))
    # show_nbv(pts_nbv, nrmls_nbv, confs_nbv, cam_pos, arrow_len=.15)

    '''
    show new
    '''
    m_planner.ah.show_armjnts(armjnts=jnts)
    eepos, eerot = m_planner.rbt.get_gl_tcp()
    eemat4 = rm.homomat_from_posrot(eepos, eerot)
    transmat4 = eemat4.dot(np.linalg.inv(initmat4))

    pcd_cropped_new = pcdu.trans_pcd(pcd_gl, transmat4)
    pts_nbv_new = pcdu.trans_pcd(pts_nbv, transmat4)
    nrmls_nbv_new = [transmat4[:3, :3].dot(n) for n in nrmls_nbv]
    pcd_gl_nxt = pcdu.trans_pcd(pcd_trans_nxt, np.dot(transmat4, gl_transmat4))
    pcdu.show_pcd(pcd_gl_nxt, rgba=(0, .7, 0, .5))
    show_nbv(pts_nbv_new, nrmls_nbv_new, confs_nbv, cam_pos, arrow_len=.15)
    base.run()
