import pickle
import time

import cv2

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as mp
import motionplanner.rbtx_motion_planner as mpx
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.recons_utils as rcu
import visualization.panda.world as wd
from pcn.inference import *

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    fo = 'extrude_1'

    rbt = el.loadXarm(showrbt=False)
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname='arm')
    rbtx = el.loadXarmx(ip='10.2.0.201')
    m_planner_x = mpx.MotionPlannerRbtX(env=None, rbt=rbt, rbtx=rbtx, armname='arm')
    # path = \
    #     m_planner.plan_start2end(start=m_planner_x.get_armjnts(),
    #                              end=pickle.load(
    #                                  open(f'{config.ROOT}/img/phoxi/nbc/plate_a_cubic/000_armjnts.pkl', 'rb')))
    # m_planner_x.movepath(path)
    rbtx.arm_jaw_to(0)

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, .0155)

    x_range = (.1, .2)
    y_range = (-.15, .02)
    z_range = (.02, .1)
    # z_range = (-.2, -.0155)
    theta = None
    max_a = np.pi / 18

    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T

    i = 0

    use_pcn = True
    if use_pcn:
        dump_path = f'phoxi/nbc_pcn/{fo}'
    else:
        dump_path = f'phoxi/nbc/{fo}'

    if not os.path.exists(os.path.join(config.ROOT, 'img', dump_path)):
        os.makedirs(os.path.join(config.ROOT, 'img', dump_path))

    while i < 2:
        seedjntagls = m_planner_x.get_armjnts()
        # seedjntagls = pickle.load(open(f'{config.ROOT}/img/phoxi/nbc/plate_a_cubic/000_jnts.pkl', 'rb'))
        pickle.dump(seedjntagls,
                    open(os.path.join(config.ROOT, 'img', dump_path, f'{str(i).zfill(3)}_jnts.pkl'), 'wb'))

        m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
        tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
        gl_transrot = np.dot(tcprot, gl_relrot)
        gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
        gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)
        gm.gen_frame(gl_transpos, tcprot).attach_to(base)

        textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img', dump_path, f'{str(i).zfill(3)}.pkl'))
        pcd = np.asarray(pcd)
        cv2.imshow('grayimg', textureimg)
        cv2.waitKey(0)
        pcd_roi, pcd_trans, gripperframe = \
            rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                        x_range=x_range, y_range=y_range, z_range=z_range, toggledebug=False)
        pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
        pcdu.show_pcd(pcd_gl, rgba=(1, 1, 1, .5))
        pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

        o3dpcd = o3dh.nparray2o3dpcd(pcd_roi - np.asarray(center))
        o3d.io.write_point_cloud(os.path.join(config.ROOT, 'recons_data', 'nbc', fo, f'{str(i).zfill(3)}.pcd'), o3dpcd)

        if use_pcn:
            # pcd_pcn = o3d.io.read_point_cloud(
            #     os.path.join(os.path.join(config.ROOT, 'img/phoxi/nbc_pcn/', fo, f'{str(i).zfill(3)}_output_lc.pcd')))
            pcd_pcn = inference_sgl(np.asarray(o3dpcd.points))
            pcd_pcn = pcd_pcn + np.asarray(center)

            # o3dpcd_o = nparray2o3dpcd(pcd_pcn)
            # o3dpcd.paint_uniform_color(COLOR[0])
            # o3dpcd_o.paint_uniform_color(COLOR[2])
            # o3d.visualization.draw_geometries([o3dpcd])
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o])
            nbv_pts, nbv_nrmls, jnts = \
                rcu.cal_nbc_pcn(pcd_roi, pcd_pcn, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
                                theta=theta, max_a=max_a, show_cam=False, toggledebug=True)
        else:
            nbv_pts, nbv_nrmls, jnts = \
                rcu.cal_nbc(pcd_roi, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
                            theta=theta, max_a=max_a, show_cam=True, toggledebug=True)
        m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))
        print(jnts)
        pickle.dump(jnts, open(os.path.join(config.ROOT, 'img', dump_path, f'{str(i + 1).zfill(3)}_jnts.pkl'), 'wb'))
        path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
        base.run()

        m_planner.ah.show_ani(path)
        m_planner_x.movepath(path)
        time.sleep(5)
        i += 1

    base.run()
