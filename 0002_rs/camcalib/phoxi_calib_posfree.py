import vision.misc.o3dutils as o3du
from camcalib.phoxi_calib_posfree_hold import *


def phoxi_computeeeinphx(motion_planner_x, phoxi, actionpos, actionrot, criteriaradius=None):
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
    lastarmjnts = motion_planner_x.get_armjnts()

    for axisid in range(3):
        axis = rangeaxis[axisid][0]
        for angle in rangeaxis[axisid][1]:
            goalpos = actionpos
            goalrot = np.dot(rm.rodrigues(axis, angle), actionrot)
            armjnts = motion_planner_x.get_numik(eepos=goalpos, eerot=goalrot, msc=lastarmjnts)
            if armjnts is not None and not motion_planner_x.is_selfcollided():
                lastarmjnts = armjnts
                success = motion_planner_x.goto_armjnts_x(armjnts, sleep=15)
                if success:
                    grayimg, depthnparray_float32, pcd = phoxi.getalldata()
                    phxpos = getcenter(grayimg, pcd)
                    print("phxpos:", phxpos)
                    if phxpos is not None:
                        coords.append(phxpos)
    print(print(len(coords)), coords)
    if len(coords) <= 3:
        return [None, None]
    for coord in coords:
        base.pggen.plotSphere(base.render, coord, rgba=np.array([1, 1, 0, .8]), radius=5)
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


def phoxi_computeboardcenterinhand(motion_planner_x, phxi_client, criteriaradius=None):
    """

    :param criteriaradius: rough radius used to determine if the newly estimated center is correct or not
    :return:

    """

    actionpos = np.array([650, 200, 1000])
    actionrot = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
    eeposinphx, bcradius = phoxi_computeeeinphx(motion_planner_x, phxi_client, actionpos, actionrot)
    # eeposinphx = np.array([-111.2444597, 22.72199986, 810.53215805])
    # bcradius = 40.9974884
    # moveback
    armjnts = motion_planner_x.get_numik(eepos=actionpos, eerot=actionrot,
                                           msc=motion_planner_x.get_armjnts())
    if armjnts is not None and not motion_planner_x.is_selfcollided():
        motion_planner_x.goto_armjnts_x(armjnts, sleep=20)
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hcinphx = getcenter(grayimg, pcd)
    print(hcinphx)

    movedist = 30
    actionpos_hx = actionpos + actionrot[:, 0] * movedist
    armjnts = motion_planner_x.get_numik(eepos=actionpos_hx, eerot=actionrot, msc=motion_planner_x.get_armjnts())
    if armjnts is not None and not motion_planner_x.is_selfcollided():
        motion_planner_x.goto_armjnts_x(armjnts, sleep=25)
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hxinphx = getcenter(grayimg, pcd)
    print(hxinphx)

    actionpos_hy = actionpos + actionrot[:, 1] * movedist
    armjnts = motion_planner_x.get_numik(eepos=actionpos_hy, eerot=actionrot, msc=motion_planner_x.get_armjnts())
    if armjnts is not None and not motion_planner_x.is_selfcollided():
        motion_planner_x.goto_armjnts_x(armjnts, sleep=25)
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        hyinphx = getcenter(grayimg, pcd)
    print(hyinphx)

    actionpos_hz = actionpos + actionrot[:, 2] * movedist
    armjnts = motion_planner_x.get_numik(eepos=actionpos_hz, eerot=actionrot,
                                           msc=motion_planner_x.get_armjnts())
    if armjnts is not None and not motion_planner_x.is_selfcollided():
        motion_planner_x.goto_armjnts_x(armjnts, sleep=25)
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


def phoxi_calibbyestinee(motion_planner_x, phxi_client, actionpos_list):
    realposlist = []
    phxposlist = []

    actionpos = np.array([650, 200, 1000])
    actionrot = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T

    # estimate a criteriaradius
    phxpos, criteriaradius = phoxi_computeeeinphx(motion_planner_x, phxi_client, actionpos, actionrot)
    print(phxpos, criteriaradius)
    # phxpos = np.array([-99.10865446, 55.1548938, 811.75575761])
    # criteriaradius = 46.2389240241248

    realposlist.append(actionpos)
    phxposlist.append(phxpos)
    print(phxposlist)

    for actionpos in actionpos_list:
        # motion_planner_x.goto_init()
        phxpos, _ = phoxi_computeeeinphx(motion_planner_x, phxi_client, actionpos, actionrot, criteriaradius)
        if phxpos is not None:
            realposlist.append(actionpos)
            phxposlist.append(phxpos)
            print("phxposlist", phxposlist)

    realposarr = np.asarray(realposlist)
    phxposarr = np.asarray(phxposlist)

    print(phxposarr)
    print(realposarr)
    amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
    pickle.dump(realposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_realpos.pkl"), "wb"))
    pickle.dump(phxposarr, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_ampos.pkl"), "wb"))
    pickle.dump(amat, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_calibmat_0605.pkl"), "wb"))
    print(amat)
    return amat


def phoxi_calib(motion_planner_x, phxi_client, relpos, actionpos_list):
    realposlist = []
    phxposlist = []

    # lastarmjnts = motion_planner_x.rbt.initrgtjnts
    # eerot = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
    #
    # rangex = [np.array([1, 0, 0]), [15]]
    # rangey = [np.array([0, 1, 0]), [15]]
    # rangez = [np.array([0, 0, 1]), [30]]
    # rangeaxis = [rangex, rangey, rangez]
    # lastarmjnts = motion_planner_x.get_armjnts()
    # for actionpos in actionpos_list:
    #     for axisid in range(3):
    #         axis = rangeaxis[axisid][0]
    #         for angle in rangeaxis[axisid][1]:
    #             goalrot = np.dot(rm.rodrigues(axis, angle), eerot)
    #             armjnts = motion_planner_x.get_armjnts_by_eeposrot(eepos=actionpos, eerot=goalrot, msc=lastarmjnts)
    #             if armjnts is not None and not motion_planner_x.pcdchecker.isSelfCollided(motion_planner_x.rbt):
    #                 lastarmjnts = armjnts
    #                 motion_planner_x.goto_armjnts_x(armjnts, sleep=15)
    #                 grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
    #                 phxpos = getcenter(grayimg, pcd)
    #
    #                 if phxpos is not None:
    #                     eepos, eerot = motion_planner_x.get_tcp()
    #                     realposlist.append(eepos + np.dot(eerot, relpos))
    #                     phxposlist.append(phxpos)
    #                     print(eepos)
    #                     print(phxpos, actionpos + np.dot(goalrot, relpos))
    i = 0
    while i < 15:
        grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
        phxpos = getcenter(grayimg, pcd)
        if phxpos is not None:
            eepos, eerot = motion_planner_x.get_ee()
            realposlist.append(eepos + np.dot(eerot, relpos))
            phxposlist.append(phxpos)
            print(eepos, eepos + np.dot(eerot, relpos))
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


def phoxi_calib_refinewithmodel(motion_planner_x, phoxi, rawamat, armname):
    """
    The performance of this refining method using cad model is not good.
    The reason is probably a precise mobdel is needed.

    """

    handpalmtemplate = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", "handpalmtemplatepcd.pkl"), "rb"))

    newhomomatlist = []

    lastarmjnts = motion_planner_x.get_armjnts()
    eerot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T  # horizontal, facing right
    for x in [300, 360, 420]:
        for y in range(-200, 201, 200):
            for z in [70, 90, 130, 200]:
                armjnts = motion_planner_x.get_numik(eepos=np.array([x, y, z]), eerot=eerot,
                                                       msc=lastarmjnts)
                if armjnts is not None and not motion_planner_x.is_selfcollided():
                    lastarmjnts = armjnts
                    motion_planner_x.goto_armjnts_x(armjnts)
                    tcppos, tcprot = motion_planner_x.get_ee()
                    initpos = tcppos + tcprot[:, 2] * 7
                    initrot = tcprot
                    inithomomat = rm.homobuild(initpos, initrot)
                    grayimg, depthnparray_float32, pcd = phoxi.getalldata()
                    realpcd = rm.homotransformpointarray(rawamat, pcd)
                    minx = tcppos[0] - 100
                    maxx = tcppos[0] + 100
                    miny = tcppos[1]
                    maxy = tcppos[1] + 140
                    minz = tcppos[2]
                    maxz = tcppos[2] + 70
                    realpcdcrop = o3du.cropnx3nparray(realpcd, [minx, maxx], [miny, maxy], [minz, maxz])
                    if len(realpcdcrop) < len(handpalmtemplate) / 2:
                        continue
                    hto3d = o3du.nparray2o3dpcd(rm.homotransformpointarray(inithomomat, handpalmtemplate))
                    rpo3d = o3du.nparray2o3dpcd(realpcdcrop)
                    inlinnerrmse, newhomomat = o3du.registration_icp_ptpt(hto3d, rpo3d, np.eye(4),
                                                                          maxcorrdist=2, toggledebug=False)
                    print(inlinnerrmse, ", one round is done!")
                    newhomomatlist.append(rm.homoinverse(newhomomat))
    newhomomat = rm.homomat_average(newhomomatlist, denoise=False)
    refinedamat = np.dot(newhomomat, rawamat)
    pickle.dump(refinedamat, open(os.path.join(config.ROOT, "camcalib/data", "phoxi_refinedcalibmat.pkl"), "wb"))
    print(rawamat)
    print(refinedamat)
    return refinedamat


def get_amat(amat_path="calibmat.pkl"):
    filepath = os.path.dirname(os.path.abspath(__file__)) + "\\" + amat_path
    amat = pickle.load(open(filepath, "rb"))
    return amat


def transformpcd(amat, pcd):
    """

    :param amat:
    :param pcd:
    :return:

    author: weiwei
    date: 20191228osaka
    """

    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(amat, homopcd).T

    return realpcd[:, :3]


if __name__ == '__main__':
    import os
    import pickle
    import utils.phoxi as phoxi

    '''
     set up env and param
     '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex()

    phxi_host = "10.0.1.66:18300"
    phxi_client = phoxi.Phoxi(host=phxi_host)

    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    motion_planner_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    motion_planner_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    motion_planner_x_lft.goto_init_x()
    # motion_planner_x_rgt.goto_init_x()

    '''
    set param
    '''
    objcm = el.loadObj("calibboard.stl")
    grasp = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/calibboard_pregrasps.pkl", "rb"))[0]
    objrelpos, objrelrot = motion_planner_lft.get_rel_posrot(grasp, objpos=(600, 200, 1150), objrot=objrot)
    print(objrelpos, objrelrot)

    relpos = phoxi_computeboardcenterinhand(motion_planner_x_lft, phxi_client)
    # relpos = np.array([2.37150062, -4.9622957, -42.55020686])
    motion_planner_x_lft.goto_init_hold_x(grasp, objcm, objrelpos, objrelrot)
    amat = phoxi_calib_auto(motion_planner_x_lft, phxi_client, grasp, objcm, objrelpos, objrelrot, -relpos)

    # actionpos_list = []
    # for x in range(600, 900, 50):
    #     for y in range(0, 400, 50):
    #         for z in range(950, 1200, 50):
    #             actionpos_list.append(np.array([x, y, z]))
    # print(len(actionpos_list))
    #
    # actionrot = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
    #
    # rangex = [np.array([1, 0, 0]), [0, 15, 30, 45]]
    # rangey = [np.array([0, 1, 0]), [-30, -15, 15, 30]]
    # rangez = [np.array([0, 0, 1]), [-60, -30, 30, 60, 90, 120, 150]]
    # rangeaxis = [rangex, rangey, rangez]
    #
    # available_actionpos_list = []
    #
    # for actionpos in actionpos_list:
    #     print("----------------")
    #     print("available_actionpos_list:", available_actionpos_list)
    #     print(actionpos)
    #
    #     available_armjnts_list = []
    #     cnt = 0
    #     msc = None
    #
    #     for axisid in range(3):
    #         axis = rangeaxis[axisid][0]
    #         for angle in rangeaxis[axisid][1]:
    #             print(axis, angle)
    #             cnt += 1
    #             goalpos = actionpos
    #             goalrot = np.dot(rm.rodrigues(axis, angle), actionrot)
    #             armjnts = motion_planner_lft.get_armjnts_by_eeposrot(eepos=goalpos, eerot=goalrot, msc=msc)
    #             print(armjnts)
    #             if armjnts is not None:
    #                 msc = armjnts
    #                 available_armjnts_list.append(armjnts)
    #
    #     print(len(available_armjnts_list), cnt)
    #     path_cnt = 0
    #     if len(available_armjnts_list) == cnt:
    #         print(actionpos, "success!")
    #         path_list = []
    #         start = available_armjnts_list[0]
    #         for i in range(1, len(available_armjnts_list)):
    #             goal = available_armjnts_list[i]
    #             path = motion_planner_lft.plan_start2end(start, goal)
    #             if path is not None:
    #                 # path_list.append(path)
    #                 path_list.extend(path)
    #                 start = goal
    #                 path_cnt += 1
    #         print(path_cnt)
    #         if path_cnt == cnt - 1:
    #             available_actionpos_list.append(actionpos)
    #
    #         motion_planner_lft.show_animation(path_list)
    #         base.run()

    # rawamat = pickle.load(open(os.path.join(ur3x.root, "datacalibration", "calibmat.pkl"), "rb"))
    # phoxi_calib_refine(ur3x, pxc, rawamat, armname="rgt")

    '''
    show calibration result
    '''
    grayimg, depthnparray_float32, pcd = phxi_client.getalldata()
    amat = phxi_client.load_phoxicalibmat(f_name="phoxi_calibmat_0615.pkl")
    print(amat)
    realpcd = pcdu.trans_pcd(pcd, amat)
    pcdu.show_pcd_withrbt(realpcd, realrbt=True)

    base.run()
