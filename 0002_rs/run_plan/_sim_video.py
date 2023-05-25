import pickle
import time

import cv2

import localenv.envloader as el
import motionplanner.motion_planner as mp
import motionplanner.rbtx_motion_planner as mpx
import utils.phoxi as phoxi
import visualization.panda.world as wd
from pcn.inference import *
import basis.robot_math as rm
import utils.recons_utils as rcu
import utils.pcd_utils as pcdu
import modeling.geometric_model as gm
import nbv.nbv_utils as nu
import bendplanner.bend_utils as bu

if __name__ == '__main__':
    base = wd.World(cam_pos=[.3, -3.2, 1.4], lookat_pos=[0, 0, 1])
    fo = 'extrude_1'
    dump_path = f'nbc_pcn/{fo}'

    rbt = el.loadXarm(showrbt=False)

    x_range = (.1, .25)
    y_range = (-.15, .02)
    # y_range = (-.02, .15)
    z_range = (-.15, -.02)
    seed = (.116, 0, .1)

    width = .008
    thickness = .002
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    i = 1
    gm.gen_frame().attach_to(base)
    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname='arm')
    if i == 0:
        seedjntagls = m_planner.get_armjnts()
    else:
        _, _, _, _, seedjntagls, _ = \
            pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', dump_path, f'{str(i).zfill(3)}_res.pkl'), 'rb'))
    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    # gl_relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    textureimg, depthimg, pcd = rcu.load_frame(dump_path, f_name=f'{str(i).zfill(3)}.pkl')

    pcd_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    pcdu.show_pcd(pcdu.trans_pcd(pcd_roi, gl_transmat4))
    cam_mat4 = np.linalg.inv(gripperframe)
    cam_mat4 = np.dot(gl_transmat4, cam_mat4)
    cam_mat4 = np.dot(cam_mat4, rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((1, 0, 0), np.pi / 2)))
    pcdu.show_cam(cam_mat4)

    pcd_cmp = np.asarray([])
    pcd_icp_list = []
    pcd_pcn_list = []
    trans_icp = np.eye(4)

    _, _, _, _, jnts, pcd_pcn = \
        pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', dump_path, f'{str(i + 1).zfill(3)}_res.pkl'), 'rb'))
    m_planner.ah.show_armjnts(rgba=(.7, .7, .7, .4), armjnts=seedjntagls)
    init_eepos, init_eerot = m_planner.rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot)
    m_planner.ah.show_armjnts(rgba=(0, 1, 0, .4), armjnts=jnts)

    # kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_pcn, rgba=(1, 1, 0, 1), n_components=16)
    # inp_pseq = nu.nurbs_inp(kpts)
    # inp_rotseq = pcdu.get_rots_wkpts(pcd_pcn, inp_pseq, k=200, show=False, rgba=(1, 0, 0, 1))
    # objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008)
    # objcm.set_homomat(init_eemat4)
    # objcm.attach_to(base)

    eepos, eerot = m_planner.rbt.get_gl_tcp()
    eemat4 = rm.homomat_from_posrot(eepos, eerot)
    transmat4 = eemat4.dot(np.linalg.inv(init_eemat4))

    pcdu.show_pcd(pcdu.trans_pcd(pcdu.trans_pcd(pcd_roi, gl_transmat4), transmat4), rgba=(0, 1, 0, 1))

    path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    m_planner.ah.show_ani(path)
    # m_planner.ah.show_animation_hold(path, objcm, objrelpos=init_eemat4[:3,3], objrelrot=init_eemat4[:3,3])
    base.run()
