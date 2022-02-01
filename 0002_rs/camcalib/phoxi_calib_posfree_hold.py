import os
import pickle
import time

import cv2
import numpy as np
from cv2 import aruco as aruco
from scipy.optimize import leastsq

import config
from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utiltools.robotmath as rm


def getcenter(img, pcd, tgtids=[0, 1]):
    """
    get the center of two markers

    :param img:
    :param pcd:
    :return:

    author: yuan gao, ruishuang, revised by weiwei
    date: 20161206
    """

    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    width = img.shape[1]
    # First, detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if len(corners) < len(tgtids):
        return None
    if len(ids) != len(tgtids):
        return None
    if ids[0] not in tgtids or ids[1] not in tgtids:
        return None
    center = np.mean(np.mean(corners, axis=0), axis=1)[0]
    center = np.array([int(center[0]), int(center[1])])
    # print(center)
    pos = pcd[width * center[1] + center[0]]

    return pos


def phoxi_computeeeinphx(motion_planner_x, phxi_client, grasp, objcm, objpos, objrot, objrelpos, objrelrot,
                         criteriaradius=None):
    """

    :param criteriaradius: rough radius used to determine if the newly estimated center is correct or not
    :return:

    author: weiwei
    date: 20190110
    """

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords.T
        return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = []
    rangex = [np.array([1, 0, 0]), [0, 15, 30, 45]]
    rangey = [np.array([0, 1, 0]), [-30, -15, 15, 30]]
    rangez = [np.array([0, 0, 1]), [30, 45, 60, 75, 90, 105, 120]]
    rangeaxis = [rangex, rangey, rangez]

    for axisid in range(3):
        axis = rangeaxis[axisid][0]
        for angle in rangeaxis[axisid][1]:
            goalrot = np.dot(rm.rodrigues(axis, angle), objrot)
            objmat4_goal = rm.homobuild(objpos, goalrot)
            success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, objmat4_goal, objcm)
            if success:
                grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
                phxpos = getcenter(grayimg, pcd)
                print("phxpos:", phxpos)
                if phxpos is not None:
                    coords.append(phxpos)

    print(print(len(coords)), coords)
    if len(coords) <= 3:
        return [None, None]
    for coord in coords:
        base.pggen.plotSphere(base.render, coord, rgba=np.array([1, 1, 0, 1]), radius=5)
    coords = np.asarray(coords)

    # try:
    initialguess = np.ones(4)
    initialguess[:3] = np.mean(coords, axis=0)
    finalestimate, flag = leastsq(errfunc, initialguess, args=(coords,))
    if len(finalestimate) == 0:
        return [None, None]
    print("finalestimate", finalestimate)
    print(np.linalg.norm(coords - finalestimate[:3], axis=1))
    base.pggen.plotSphere(base.render, finalestimate[:3], rgba=np.array([0, 1, 0, 1]), radius=5)
    base.run()

    if criteriaradius is not None:
        if abs(finalestimate[3] - criteriaradius) > 5:
            return [None, None]
    return np.array(finalestimate[:3]), finalestimate[3]


def phoxi_computeeeinphx_loadpath(motion_planner_x, phxi_client, criteriaradius=None):
    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords.T
        return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = []
    path = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", "calib_rot_path.pkl"), "rb"))
    # objmat4_start = motion_planner_x.get_world_objmat4(objrelpos, objrelrot, armjnts=motion_planner_x.get_armjnts())
    # objmat4_goal = motion_planner_x.get_world_objmat4(objrelpos, objrelrot, armjnts=path[0][0])
    # success = motion_planner_x.goto_objmat4_goal_x(grasp, objmat4_start, objmat4_goal, objcm)
    # motion_planner_x.show_armjnts(armjnts=path[0][0])
    # base.run()
    success = motion_planner_x.goto_armjnts_x(path[0][0])
    if success:
        for path_temp in path:
            motion_planner_x.movepath(path_temp)
            grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
            phxpos = getcenter(grayimg, pcd)
            print("phxpos:", phxpos)
            if phxpos is not None:
                coords.append(phxpos)
            # time.sleep(10)

        print(print(len(coords)), coords)
        if len(coords) <= 3:
            return [None, None]
        for coord in coords:
            base.pggen.plotSphere(base.render, coord, rgba=np.array([1, 1, 0, 1]), radius=5)
        coords = np.asarray(coords)

        # try:
        initialguess = np.ones(4)
        initialguess[:3] = np.mean(coords, axis=0)
        finalestimate, flag = leastsq(errfunc, initialguess, args=(coords,))
        if len(finalestimate) == 0:
            return [None, None]
        print("finalestimate", finalestimate)
        print(np.linalg.norm(coords - finalestimate[:3], axis=1))
        base.pggen.plotSphere(base.render, finalestimate[:3], rgba=np.array([0, 1, 0, 1]), radius=5)
        # base.run()

        if criteriaradius is not None:
            if abs(finalestimate[3] - criteriaradius) > 5:
                return [None, None]
        return np.array(finalestimate[:3]), finalestimate[3]


def phoxi_computeboardcenterinhand(motion_planner_x, phxi_client, grasp, objcm, objpos, objrot, objrelpos, objrelrot,
                                   criteriaradius=None):
    # eeposinphx, bcradius = phoxi_computeeeinphx(motion_planner_x, phxi_client, grasp, objcm, objpos, objrot,
    #                                             objrelpos, objrelrot)
    eeposinphx, bcradius = phoxi_computeeeinphx_loadpath(motion_planner_x, phxi_client)

    # moveback
    success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos, objrot), objcm)
    if success:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hcinphx = getcenter(grayimg, pcd)
    print(hcinphx)

    movedist = 5
    objpos_hx = objpos + objrot[:, 0] * movedist
    success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos_hx, objrot), objcm)
    if success:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hxinphx = getcenter(grayimg, pcd)
    print(hxinphx)

    objpos_hy = objpos + objrot[:, 1] * movedist
    success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos_hy, objrot), objcm)
    if success:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hyinphx = getcenter(grayimg, pcd)
    print(hyinphx)

    objpos_hz = objpos + objrot[:, 2] * movedist
    success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, rm.homobuild(objpos_hz, objrot), objcm)
    if success:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hzinphx = getcenter(grayimg, pcd)
    print(hzinphx)

    frameinphx = np.array([hxinphx - hcinphx, hyinphx - hcinphx, hzinphx - hcinphx]).T
    frameinphx, r = np.linalg.qr(frameinphx)
    print(frameinphx)
    print(r)
    bcinhnd = np.dot(frameinphx.T, hcinphx - eeposinphx)
    print(bcinhnd)
    return bcinhnd


def phoxi_calib_manual(motion_planner_x, phxi_client, relpos):
    realposlist = []
    phxposlist = []

    i = 0
    while i < 15:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        phxpos = getcenter(grayimg, pcd)
        if phxpos is not None:
            tcppos, tcprot = motion_planner_x.get_ee()
            realposlist.append(tcppos + np.dot(tcprot, relpos))
            phxposlist.append(phxpos)
            print(tcppos, tcppos + np.dot(tcprot, relpos))
            print(phxpos)
            cv2.imshow("tst", grayimg)
            cv2.waitKey(0)
            i += 1
        else:
            print("marker not detected!")
            cv2.imshow("tst", grayimg)
            cv2.waitKey(0)

    realposarr = np.array(realposlist)
    phxposarr = np.array(phxposlist)
    print(realposlist)
    print(phxposlist)
    amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
    pickle.dump(realposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_realpos.pkl"), "wb"))
    pickle.dump(phxposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_ampos.pkl"), "wb"))
    pickle.dump(amat, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_calibmat_0615.pkl"), "wb"))
    print(amat)

    return amat


def phoxi_calib_auto(motion_planner_x, phxi_client, grasp, objcm, objrelpos, objrelrot, relpos, dump_id=''):
    realposlist = []
    phxposlist = []
    path = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", "calib_trans_path.pkl"), "rb"))
    # objmat4_goal = motion_planner_x.get_world_objmat4(objrelpos, objrelrot, armjnts=path[0][0])
    # success = motion_planner_x.goto_objmat4_goal_x(grasp, objrelpos, objrelrot, objmat4_goal, objcm)
    success = motion_planner_x.goto_armjnts_hold_x(grasp, objcm, objrelpos, objrelrot, armjnts=path[0][0])
    if success:
        for path_temp in path:
            grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
            phxpos = getcenter(grayimg, pcd)

            if phxpos is not None:
                objmat4 = motion_planner_x.get_world_objmat4(objrelpos, objrelrot, motion_planner_x.get_armjnts())
                objpos = objmat4[:3, 3]
                objrot = objmat4[:3, :3]
                realposlist.append(objpos + np.dot(objrot, relpos))
                phxposlist.append(phxpos)
                print(objpos)
                print(phxpos, objpos + np.dot(objrot, relpos))
            motion_planner_x.movepath(path_temp)

        realposarr = np.array(realposlist)
        phxposarr = np.array(phxposlist)

        amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
        pickle.dump(realposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_realpos.pkl"), "wb"))
        pickle.dump(phxposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_ampos.pkl"), "wb"))
        pickle.dump(amat, open(os.path.join(config.ROOT, "camcalib/data", f"phoxi_calibmat_{dump_id}.pkl"), "wb"))
        print(amat)
        return amat
    else:
        return None


def get_avaliable_objpos_rotmotion(mp, objcm, grasp, objrot_init, objrelpos, objrelrot):
    actionpos_list = []
    for x in range(800, 900, 20):
        for y in range(200, 400, 20):
            for z in range(980, 1080, 20):
                actionpos_list.append(np.array([x, y, z]))
    print(len(actionpos_list))

    rangex = [np.array([1, 0, 0]), [0, 10, 20, 30, 40, 50]]
    rangey = [np.array([0, 1, 0]), [-30, -20, -10, 0, 10, 20, 30]]
    rangez = [np.array([0, 0, 1]), [30, 40, 50, 60, 70, 80, 90, 100, 110]]
    rangeaxis = [rangex, rangey, rangez]

    available_actionpos_list = []

    for actionpos in actionpos_list:
        print("----------------")
        print("available_actionpos_list:", available_actionpos_list)
        print(actionpos)

        available_armjnts_list = []
        cnt = 0
        msc = None

        for axisid in range(3):
            axis = rangeaxis[axisid][0]
            for angle in rangeaxis[axisid][1]:
                print(axis, angle)
                cnt += 1
                goalpos = actionpos
                goalrot = np.dot(rm.rodrigues(axis, angle), objrot_init)
                objmat4_goal = rm.homobuild(goalpos, goalrot)
                armjnts = mp.get_armjnts_by_objmat4ngrasp(grasp, [objcm], objmat4_goal, msc=msc)
                if armjnts is not None:
                    print(armjnts)
                    msc = armjnts
                    available_armjnts_list.append(armjnts)
                else:
                    break
        print(len(available_armjnts_list), cnt)

        path_cnt = 0
        if len(available_armjnts_list) == cnt:
            available_armjnts_list.append(available_armjnts_list[0])
            path_show = [mp.rbt.initlftjnts]
            path_dump = []
            for i in range(len(available_armjnts_list)):
                goal = available_armjnts_list[i]
                objmat4_start = mp.get_world_objmat4(objrelpos, objrelrot, path_show[-1])
                objmat4_goal = mp.get_world_objmat4(objrelpos, objrelrot, goal)
                path = mp.plan_start2end_hold(grasp, [objmat4_start, objmat4_goal], objcm, objrelpos, objrelrot,
                                              start=path_show[-1])
                if path is not None:
                    path_show.extend(path)
                    path_dump.append(path)
                    path_cnt += 1
                else:
                    break
            print("success path:", path_cnt)
            if path_cnt == cnt + 1:
                print(actionpos, "success!")
                pickle.dump(path_dump, open(os.path.join(config.ROOT, "camcalib/data", "calib_rot_path.pkl"), "wb"))
                available_actionpos_list.append(actionpos)
                break


def get_avaliable_objpos_transmotion(mp, objcm, grasp, objrot, objrelpos, objrelrot):
    objpos_list = []
    for x in range(600, 1000, 100):
        for y in range(100, 400, 60):
            for z in range(950, 1100, 50):
                objpos_list.append(np.array([x, y, z]))

    available_armjnts_list = []
    msc = None
    # for objpos in objpos_list:
    #     objmat4_goal = rm.homobuild(objpos, objrot)
    #     armjnts = mp.get_armjnts_by_objmat4ngrasp(grasp, objcm, objmat4_goal, msc=msc)
    #     if armjnts is not None:
    #         msc = armjnts
    #         available_armjnts_list.append(armjnts)

    for objpos in objpos_list:
        objmat4_goal = rm.homobuild(objpos, np.dot(objrot, rm.rodrigues((0, 0, 1), 90)))
        armjnts = mp.get_armjnts_by_objmat4ngrasp(grasp, [objcm], objmat4_goal, msc=msc)
        if armjnts is not None:
            msc = armjnts
            available_armjnts_list.append(armjnts)

    print("success pos(has ik):", len(available_armjnts_list))

    path_cnt = 0
    path_dump = []
    path_show = [mp.rbt.initlftjnts]
    for i in range(0, len(available_armjnts_list)):
        goal = available_armjnts_list[i]
        objmat4_start = mp.get_world_objmat4(objrelpos, objrelrot, path_show[-1])
        objmat4_goal = mp.get_world_objmat4(objrelpos, objrelrot, goal)
        path = mp.plan_start2end_hold(grasp, [objmat4_start, objmat4_goal], objcm, objrelpos,
                                      objrelrot, start=path_show[-1])
        if path is not None:
            print("length of path", len(path))
            if path_cnt > 0 and len(path) > 15:
                continue
            path_dump.append(path)
            path_show.extend(path)
            mp.ah.show_objmat4(objcm, objmat4_goal)
            path_cnt += 1

    print("success pos(has path):", len(path_dump), len(available_armjnts_list), len(objpos_list))
    pickle.dump(path_dump, open(os.path.join(config.ROOT, "camcalib/data", "calib_trans_path.pkl"), "wb"))


def show_path(mp, f_name, objcm, objrelpos, objrelrot):
    realposlist = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_realpos.pkl"), "rb"))
    phxposlist = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_ampos.pkl"), "rb"))
    amat = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_calibmat_temp.pkl"), "rb"))
    phxposlist = pcdu.trans_pcd(phxposlist, amat)
    for p in realposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(1, 0, 0, 1))
    for p in phxposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(0, 0, 1, 1))
    path = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", f_name), "rb"))
    path_show = []
    for path_temp in path:
        path_show += path_temp
        objmat4_temp = mp.get_world_objmat4(objrelpos, objrelrot, path_temp[0])
        mp.ah.show_objmat4(objcm, objmat4_temp, rgba=(1, 1, 0, 1))
        base.pggen.plotSphere(base.render, objmat4_temp[:3, 3], radius=5, rgba=(0, 1, 0, 1))
    mp.rbth.plot_armjnts(path_show)
    mp.ah.show_animation_hold(path_show, objcm, objrelpos, objrelrot)


def recalculate_amat(dump_id=''):
    realposlist = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_realpos.pkl"), "rb"))
    phxposlist = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_ampos.pkl"), "rb"))
    realposarr = np.array(realposlist)
    phxposarr = np.array(phxposlist)
    amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
    pickle.dump(amat, open(os.path.join(config.ROOT, "camcalib/data", f"phoxi_calibmat_{dump_id}.pkl"), "wb"))

    phxposlist = pcdu.trans_pcd(phxposlist, amat)
    for p in realposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(1, 0, 0, 1))
    for p in phxposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(0, 0, 1, 1))


def recalculate_amat_by_removeoutliers(dump_id=''):
    import itertools
    import copy
    import heapq
    from collections import Counter
    realposlist = list(pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_realpos.pkl"), "rb")))
    phxposlist = list(pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_ampos.pkl"), "rb")))
    max_len = len(phxposlist)
    divisor = 2
    iter_times = 0
    while iter_times < 3:
        print(len(phxposlist))
        print("----------------------", iter_times, "----------------------")
        index_list_sub = list(itertools.combinations(range(int(max_len / divisor) - iter_times), divisor))
        print(len(index_list_sub))
        max_error_index_list = []
        for index_list in index_list_sub:
            index_list = list(index_list)
            index_list_new = copy.deepcopy(index_list)
            for index in index_list:
                index_list_new.append(index + iter_times)
                index_list_new.append(index + iter_times * 2)
            realposarr = np.array([p for index, p in enumerate(realposlist) if index in index_list_new])
            phxposarr = np.array([p for index, p in enumerate(phxposlist) if index in index_list_new])
            amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
            phxposlist_temp = pcdu.trans_pcd(np.array(phxposlist), amat)
            error_list = []
            for i, p in enumerate(phxposlist_temp):
                error_list.append(np.linalg.norm(phxposlist_temp[i] - realposlist[i]))
            if max(error_list) > 100:
                continue
            max_error_index_list.extend(list(map(error_list.index, heapq.nlargest(divisor, error_list))))

        remove_index_list = [t[0] for t in Counter(max_error_index_list).most_common(divisor)]
        print(Counter(max_error_index_list))
        print(remove_index_list)
        iter_times += 1
        realposlist = [realposlist[i] for i in range(len(realposlist)) if (i not in remove_index_list)]
        phxposlist = [phxposlist[i] for i in range(len(phxposlist)) if (i not in remove_index_list)]
        print("num of remain point:", len(realposlist))
        final_amat = rm.affine_matrix_from_points(np.array(phxposlist).T, np.array(realposlist).T)
        pickle.dump(final_amat, open(os.path.join(config.ROOT, "camcalib/data", f"phoxi_calibmat_{dump_id}.pkl"), "wb"))

    final_amat = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", f"phoxi_calibmat_{dump_id}.pkl"), "rb"))
    realposlist = list(pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_realpos.pkl"), "rb")))
    phxposlist = list(pickle.load(open(os.path.join(config.ROOT, "camcalib/data/phoxi_ampos.pkl"), "rb")))
    phxposlist = pcdu.trans_pcd(np.array(phxposlist), final_amat)
    for p in realposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(1, 0, 0, 1))
    for p in phxposlist:
        base.pggen.plotSphere(base.render, p, radius=5, rgba=(0, 0, 1, 1))


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)

    phxi_client = phoxi.Phoxi(host=config.PHOXI_HOST)

    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")

    '''
    set param
    '''
    objrot = np.array([(-1, 0, 0), (0, -1, 0), (0, 0, 1)]).T
    objpos = np.array([800, 240, 1000])
    objcm = el.loadObj("calibboard.stl")
    grasp = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/calibboard_pregrasps.pkl", "rb"))[0]
    for x in range(500, 700, 20):
        objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objpos=(x, 200, 1150), objrot=objrot)
        if objrelrot is not None:
            break
    print(objrelpos, objrelrot)

    '''
    calibration motion
    '''
    dump_id = '210527'
    relpos = phoxi_computeboardcenterinhand(mp_x_lft, phxi_client, grasp, objcm, objpos, objrot, objrelpos,
                                            objrelrot)
    print("relpos:", relpos)
    amat = phoxi_calib_auto(mp_x_lft, phxi_client, grasp, objcm, objrelpos, objrelrot, -relpos, dump_id=dump_id)

    '''
    planning
    '''
    # get_avaliable_objpos_rotmotion(mp_lft, objcm, grasp, objrot, objrelpos, objrelrot)
    # show_path(mp_lft, "calib_rot_path.pkl", objcm, objrelpos, objrelrot)
    # base.run()

    # get_avaliable_objpos_transmotion(mp_lft, objcm, grasp, objrot, objrelpos, objrelrot)
    # show_path(mp_lft, "calib_trans_path.pkl", objcm, objrelpos, objrelrot)
    # base.run()

    '''
    show calibration result
    '''
    # recalculate_amat_by_removeoutliers(dump_id=dump_id)
    grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
    amat = phxi_client.load_phoxicalibmat(f_name=f"phoxi_calibmat_{dump_id}.pkl")
    print(amat)
    realpcd = pcdu.trans_pcd(pcd, amat)
    pcdu.show_pcd_withrbt(realpcd, rbtx=rbtx)
    base.run()
