import os
import h5py
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
import basis.o3dhelper as o3dh


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

    fo = 'plate_a_cubic'
    pcn_path = os.path.join(config.ROOT, 'img/phoxi/nbc_pcn/')
    # show_pcn_res(fo, pcn_path)

    tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
    # gm.gen_frame(tcppos + tcprot[:, 2] * (.03466 + .065), tcprot).attach_to(base)
    relrot = np.asarray([[0, 0, -1], [0, -1, 0], [1, 0, 0]])
    gl_transrot = np.dot(tcprot, relrot)
    gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
    gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)

    seed = (.116, 0, .1)
    center = (.116, 0, -.0155)

    x_range = (.06, .215)
    y_range = (-.15, .15)
    z_range = (.0155, .2)
    # z_range = (-.2, -.0155)

    textureimg, depthimg, pcd = rcu.load_frame(os.path.join('nbc_pcn', fo), f_name='000.pkl')
    pcd = np.asarray(pcd) / 1000
    pcd_pcn = o3d.io.read_point_cloud(os.path.join(pcn_path, fo, '000_output_lc.pcd'))
    pcd_pcn = np.asarray(pcd_pcn.points) + np.asarray(center)

    seedjntagls = rbt.get_jnt_values('arm')
    pcd_roi, pcd_trans, gripperframe = \
        rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                    x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)

    pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
    # pcdu.show_pcd(pcd_gl, rgba=(1, 0, 0, 1))
    # pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

    pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_roi, pcd_pcn, theta=np.pi / 6, toggledebug=True)
    base.run()

    pts_nbv, nrmls_nbv, jnts = \
        rcu.cal_nbc_pcn(pcd_roi, pcd_pcn, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
                        show_cam=False, theta=np.pi / 6)
    pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

    # base.run()

    m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
    m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))
    # path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
    # m_planner.ah.show_ani(path)
    base.run()
