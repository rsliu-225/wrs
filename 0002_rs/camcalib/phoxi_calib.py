import cv2
import time
import yaml
import pickle

import numpy as np
from cv2 import aruco as aruco

from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utiltools.robotmath as rm
import utils.phoxi as phoxi
import utils.pcd_utils as pcdu


def getcenter(img, pcd):
    width = img.shape[1]
    # First, detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if len(corners) < 2:
        return None
    if len(ids) < 2 or len(ids) > 2:
        return None
    if ids[0] not in [0, 1] or ids[1] not in [0, 1]:
        return None
    center = np.mean(np.mean(corners, axis=0), axis=1)[0]
    center = np.array([int(center[0]), int(center[1])])
    # print(center)
    pos = pcd[width * center[1] + center[0]]

    return pos


def phoxi_calib(motion_planner_x, phxi_client, pos_num=15, pic_num=1, manual=True):
    realposlist = []
    poslist = []
    i = 0

    if manual:
        jnts_list = []

        while (i < pos_num):
            print(i)
            poslst = []
            for j in range(0, pic_num):
                grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
                tmppos = getcenter(grayimg, pcd)
                if tmppos is not None:
                    poslst.append(tmppos)
            if poslst == []:
                print("No marker!")
                cv2.imshow("random", grayimg)
                cv2.waitKey(0)
                continue

            tmppos = rm.posvec_average(poslst)
            tmprealjnts = motion_planner_x.get_armjnts()
            tmprealpos, tmprealrot = motion_planner_x.get_ee(tmprealjnts)
            tmprealpos = tmprealpos + tmprealrot[:, 2] * (206.5 + 25)
            tmprealpos = tmprealpos - tmprealrot[:, 0] * 0.75

            print("real pos", tmprealpos)
            print("temp pos", tmppos)

            cv2.imshow("random", grayimg)
            cv2.waitKey(0)
            time.sleep(2)
            if tmppos is not None:
                realposlist.append([tmprealpos[0], tmprealpos[1], tmprealpos[2]])
                poslist.append([tmppos[0], tmppos[1], tmppos[2]])
                jnts_list.append(tmprealjnts)
                i = i + 1
        pickle.dump(jnts_list, open("data/phoxi_jnts.pkl", "wb"))

    else:
        jnts_list = pickle.load(open("data/phoxi_jnts.pkl", "rb"))
        for jnts in jnts_list:
            print(jnts)
            if not motion_planner_x.is_selfcollided(jnts):
                motion_planner_x.goto_armjnts_x(jnts)
                poslst = []
                for j in range(0, pic_num):
                    grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
                    tmppos = getcenter(grayimg, pcd)
                    if tmppos is not None:
                        poslst.append(tmppos)
                if poslst == []:
                    print("No marker!")
                    continue

                tmppos = rm.posvec_average(poslst)

                tmprealjnts = motion_planner_x.get_armjnts()
                tmprealpos, tmprealrot = motion_planner_x.get_ee(tmprealjnts)
                tmprealpos = tmprealpos + tmprealrot[:, 2] * (206.5 + 25)
                tmprealpos = tmprealpos - tmprealrot[:, 0] * 0.75

                print("real pos", tmprealpos)
                print("temp pos", tmppos)

                if tmppos is not None:
                    realposlist.append([tmprealpos[0], tmprealpos[1], tmprealpos[2]])
                    poslist.append([tmppos[0], tmppos[1], tmppos[2]])

    realposarr = np.array(realposlist)
    posarr = np.array(poslist)
    amat = rm.affine_matrix_from_points(posarr.T, realposarr.T)

    pickle.dump(realposarr, open("data/phoxi_realpos.pkl", "wb"))
    pickle.dump(posarr, open("data/phoxi_campos.pkl", "wb"))
    pickle.dump(amat, open("data/phoxi_calibmat.pkl", "wb"))
    print(amat)
    return amat


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex()

    phxi_host = "10.0.1.124:18300"
    phxi_client = phoxi.Phoxi(host=phxi_host)

    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    motion_planner_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbt, rbtx, armname="lft")

    '''
    Initialize camera

    '''
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    mtx0, dist0, rvecs0, tvecs0, candfiles0 = yaml.load(open('data/phoxi_1.yaml', 'rb'))

    for armname in ["lft", "rgt"]:
        tmprealjnts = rbtx.getjnts(armname)
        rbt.movearmfk(tmprealjnts, armname)

    # amat = phoxi_calib(motion_planner_x_lft, phxi_client, manual=True)

    '''
    show calibration result
    '''
    grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
    amat = phxi_client.load_phoxicalibmat(f_name="phoxi_calibmat_0217.pkl")

    realpcd = pcdu.trans_pcd(pcd, amat)
    pcdu.show_pcd_withrbt(realpcd, realrbt=True)

    base.run()
