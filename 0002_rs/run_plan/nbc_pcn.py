import os

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


def show_pcn_res(fo, pcn_path="D:\liu\data\output(real)-20220705T140620Z-001\output(real)"):
    # for f in os.listdir(os.path.join(config.ROOT, 'pcd_output', fo)):
    for f in os.listdir(os.path.join(pcn_path, fo)):
        print(f)
        if f[-3:] != 'pcd':
            continue
        # op = o3d.io.read_point_cloud(os.path.join(config.ROOT, 'pcd_output', fo, f))
        op = o3d.io.read_point_cloud(os.path.join(pcn_path, fo, f))
        ip = o3d.io.read_point_cloud(os.path.join(config.ROOT, 'recons_data/seq', fo, f'{f.split("_")[0]}.pcd'))

        op.paint_uniform_color([1, 1, 0])
        ip.paint_uniform_color([0, 1, 1])
        # o3d.visualization.draw_geometries([ip,op])
        base.run()


if __name__ == '__main__':
    cam_pos = [.5, .5, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    rbt = el.loadXarm(showrbt=False)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname="arm")
    seedjntagls = m_planner.get_armjnts()

    fo = 'nbc/extrude_1'

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T
    # gl_relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, gl_relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    seed = (.116, 0, .1)
    center = (.116, 0, -.0155)

    x_range = (.09, .2)
    y_range = (-.15, .02)
    z_range = (.02, .1)
    # z_range = (-.2, -.0155)

    textureimg, depthimg, pcd = rcu.load_frame(fo, f_name='000.pkl')
    # pcd_pcn = np.asarray(pcd_pcn) + np.asarray(center)

    seedjntagls = rbt.get_jnt_values('arm')
    pcd_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
    pcd_roi = pcdu.remove_outliers(pcd_roi, nb_points=100, radius=0.01)
    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    pcdu.show_pcd(pcd_gl, rgba=(1, 0, 0, 1))
    pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

    # pcd_pcn = inference.inference_sgl(pcd_roi, load_model='pcn_emd_rlen/best_cd_p_network.pth', toggledebug=True)
    # pcd_pcn = np.asarray(pcd_pcn) + np.asarray(center)
    pts_nbv, nrmls_nbv, jnts = \
        rcu.cal_nbc_pcn(pcd_roi, gripperframe, rbt, seedjntagls, center=center, gl_transmat4=gl_transmat4,
                        show_cam=True, theta=np.pi / 6, toggledebug=True)
    pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

    m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
    m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))
    # path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    # m_planner.ah.show_ani(path)
    base.run()
