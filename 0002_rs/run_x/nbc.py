import pickle
import time

import cv2

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as mp
import motionplanner.rbtx_motion_planner as mpx
import nbv.nbv_utils as nu
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.recons_utils as rcu
import visualization.panda.world as wd
from pcn.inference import *

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    fo = 'extrude_2'

    rbt = el.loadXarm(showrbt=False)
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname='arm')
    rbtx = el.loadXarmx(ip='10.2.0.201')
    m_planner_x = mpx.MotionPlannerRbtX(env=None, rbt=rbt, rbtx=rbtx, armname='arm')

    icp = False

    seed = (.116, 0, .1)
    center = (.116, 0, -.02)

    x_range = (.1, .2)
    # y_range = (-.15, .02)
    y_range = (-.02, .15)
    z_range = (-.1, -.02)
    theta = None
    max_a = np.pi / 18

    gl_relrot = np.asarray([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).T

    i = 0

    method = 'pcn'
    if method != '':
        dump_path = f'phoxi/nbc_{method}/{fo}'
    else:
        dump_path = f'phoxi/nbc/{fo}'

    if not os.path.exists(os.path.join(config.ROOT, 'img', dump_path)):
        os.makedirs(os.path.join(config.ROOT, 'img', dump_path))

    m_planner_x.goto_init_x()
    pcd_cmp = np.asarray([])
    pcd_icp_list = []
    pcd_pcn_list = []
    trans_icp = np.eye(4)

    while i < 4:
        seedjntagls = m_planner_x.get_armjnts()
        m_planner.ah.show_armjnts(armjnts=seedjntagls, rgba=(1, 0, 0, .5))
        tcppos, tcprot = m_planner.get_tcp(armjnts=seedjntagls)
        gl_transrot = np.dot(tcprot, gl_relrot)
        gl_transpos = tcppos + tcprot[:, 2] * (.03466 + .065)
        gl_transmat4 = rm.homomat_from_posrot(gl_transpos, gl_transrot)
        gm.gen_frame(gl_transpos, tcprot).attach_to(base)

        textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img', dump_path, f'{str(i).zfill(3)}.pkl'))
        # textureimg, depthimg, pcd = phxi.loadalldata(f_name=os.path.join('img', dump_path, f'{str(i).zfill(3)}.pkl'))
        cv2.imshow('', textureimg)
        cv2.waitKey(0)

        pcd_roi, pcd_trans, gripperframe = \
            rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                        x_range=x_range, y_range=y_range, z_range=z_range)
        pcd_icp, _, _ = \
            rcu.extract_roi_by_armarker(textureimg, pcd, seed=seed,
                                        x_range=(0, x_range[1]), y_range=y_range, z_range=(z_range[0], 0))
        pcd_gl = pcdu.trans_pcd(pcd_trans, gl_transmat4)
        pcd_pcn = None
        # pcdu.show_pcd(pcd_gl, rgba=(1, 1, 1, .5))
        # pcdu.show_pcd(pcd_roi, rgba=(1, 0, 0, 1))

        o3dpcd_tmp = o3dh.nparray2o3dpcd(pcd_roi - np.asarray(center))
        o3dpcd_icp = o3dh.nparray2o3dpcd(pcd_icp - np.asarray(center))
        o3dpcd_tmp.paint_uniform_color(nu.COLOR[0])
        o3dpcd_icp.paint_uniform_color((.7, .7, .7))
        o3dpcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.01, max_nn=200))

        if i >= 1 and method != '':
            o3dpcd_pcn = o3dh.nparray2o3dpcd(pcd_pcn_list[-1] - np.asarray(center))
            o3dpcd_pcn.paint_uniform_color(nu.COLOR[2])
            o3dpcd_pcn.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=.01, max_nn=200))
            kdt_pcn = o3d.geometry.KDTreeFlann(o3dpcd_pcn)
            o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd_pcn])
            o3dpcd_tmp.remove_radius_outlier(nb_points=50, radius=.01)
            # remain_ids = []
            # for j in range(len(o3dpcd_tmp.points)):
            #     _, idx, _ = kdt_pcn.search_knn_vector_3d(o3dpcd_tmp.points[j], 10)
            #     dist = np.linalg.norm(np.asarray(o3dpcd_tmp.points[j]) - np.asarray(o3dpcd_tmp.points[idx[0]]))
            #     # a = rm.angle_between_vectors(np.asarray(o3dpcd_tmp.normals[j]), np.asarray(o3dpcd_tmp.normals[idx[0]]))
            #     # a = min([a, np.pi - a])
            #     if dist < .05:
            #         remain_ids.append(j)
            # o3dpcd_tmp = o3dpcd_tmp.select_by_index(remain_ids)

        o3d.visualization.draw_geometries([o3dpcd_tmp, o3dpcd_icp])
        pcd_icp_list.append(pcd_icp)

        if i == 0:
            pcd_cmp = pcd_roi
        else:
            if icp:
                _, _, trans_tmp = \
                    o3dh.registration_ptpt(pcd_icp_list[i], pcd_icp_list[i - 1], downsampling_voxelsize=.01,
                                           toggledebug=True)
                trans_icp = trans_tmp.dot(trans_icp)
                pcd_roi = pcdu.trans_pcd(pcd_roi, trans_icp)
                pcd_cmp = np.concatenate([pcd_cmp, pcd_roi])
            else:
                pcd_cmp = np.concatenate([pcd_cmp, pcd_roi])

        o3dpcd_cmp = nparray2o3dpcd(pcd_cmp)
        o3dpcd_cmp.paint_uniform_color((1, 0, 1))
        o3d.visualization.draw_geometries([o3dpcd_cmp])
        if method == 'pcn':
            pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
                rcu.cal_nbc_pcn(pcd_cmp, gripperframe, rbt, seedjntagls, center=center, gl_transmat4=gl_transmat4,
                                theta=theta, max_a=max_a, toggledebug_p3d=False, toggledebug=True)
        elif method == 'opt':
            pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
                rcu.cal_nbc_pcn_opt(pcd_cmp, gripperframe, rbt, seedjntagls, center=center, gl_transmat4=gl_transmat4,
                                    theta=theta, toggledebug=True)
            # pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
            #     pickle.load(open(os.path.join(f'../run_plan/tmp_res.pkl'), 'rb'))

        else:
            pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts = \
                rcu.cal_nbc(pcd_cmp, gripperframe, rbt, seedjntagls=seedjntagls, gl_transmat4=gl_transmat4,
                            theta=theta, max_a=max_a, toggledebug_p3d=False, toggledebug=True)
        pcd_pcn_list.append(pcd_pcn)
        m_planner.ah.show_armjnts(armjnts=jnts, rgba=(0, 1, 0, .5))
        print(jnts)

        path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
        # m_planner.ah.show_ani(path)
        # base.run()
        m_planner_x.movepath(path)
        time.sleep(5)
        i += 1
        pickle.dump([pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn],
                    open(os.path.join(config.ROOT, 'img', dump_path, f'{str(i).zfill(3)}_res.pkl'), 'wb'))
        phxi.dumpalldata(f_name=os.path.join('img', dump_path, f'{str(i).zfill(3)}.pkl'))

    # base.run()
