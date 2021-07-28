import itertools
import os
import pickle

import numpy as np

import config
import environment.bulletcdhelper as bch
from localenv import envloader as el
import utiltools.robotmath as rm
from motion import collisioncheckerball as cdck


class HandoverPlanner(object):

    def __init__(self, objname, rbt, rbtball, retractdistance=100):
        """

        :param obj: obj name (str) or objcm, objcm is for debug purpose
        :param rhx: see helper.py
        :param retractdistance: retraction distance

        author: hao, refactored by weiwei
        date: 20191206, 20200104osaka
        """
        self.objcm = el.loadObj(objname)
        self.objname = self.objcm.name
        self.rbt = rbt
        self.rbtball = rbtball
        self.retractdistance = retractdistance
        self.bcdchecker = bch.MCMchecker(toggledebug=False)
        self.cdchecker = cdck.CollisionCheckerBall(rbtball)
        self.rgthnd = rbt.rgthnd
        self.lfthnd = rbt.lfthnd

        with open(os.path.join(config.ROOT, "graspplanner/pregrasp_hndovr", f"{self.objname}_pregrasps.pkl"),
                  "rb") as f:
            self.identityglist_rgt = pickle.load(f)

        with open(os.path.join(config.ROOT, "graspplanner/pregrasp_hndovr", f"{self.objname}_pregrasps.pkl"),
                  "rb") as f:
            self.identityglist_lft = pickle.load(f)

        self.grasp = [self.identityglist_rgt, self.identityglist_lft]
        self.hndfa = [self.rgthnd, self.lfthnd]
        # paramters
        self.fpsnpmat4 = []
        self.identitygplist = []  # grasp pair list at the identity pose
        self.fpsnestedglist_rgt = {}
        # fpsnestedglist_rgt[fpid] = [g0, g1, ...],  fpsnestedglist means glist at each floating pose
        self.fpsnestedglist_lft = {}
        # fpsnestedglist_lft[fpid] = [g0, g1, ...]
        self.ikfid_fpsnestedglist_rgt = {}  # fid - feasible id
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_lft = {}

    def genhvgpsgl(self, posvec, rotmat=None, debug=False):
        """
        generate the handover grasps using the given position and orientation
        sgl means a single position
        rotmat could either be a single one or multiple (0,90,180,270, default)

        :param posvec
        :param rotmat
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identitygplist = []
        if rotmat is None:
            self.fpsnpmat4 = rm.gen_icohomomats_flat(posvec=posvec, angles=[0, 45, 90, 135, 180, 225, 270])
            # self.fpsnpmat4 = rm.gen_icohomomats_flat(posvec=posvec, angles=[90])
        else:
            self.fpsnpmat4 = [rm.homobuild(posvec, rotmat)]

        if debug:
            import copy
            for mat in self.fpsnpmat4:
                objtmp = copy.deepcopy(self.objcm)
                objtmp.sethomomat(mat)
                objtmp.reparentTo(base.render)
            base.run()

        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(config.ROOT, "graspplanner/handover")):
            os.mkdir(os.path.join(config.ROOT, "graspplanner/handover"))
        with open(os.path.join(config.ROOT, "graspplanner/handover", self.objname + "_hndovrinfo.pkl"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def genhvgplist(self, hvgplist):
        """
        generate the handover grasps using the given list of homomat

        :param hvgplist, [homomat0, homomat1, ...]
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identitygplist = []
        self.fpsnpmat4 = hvgplist
        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(config.ROOT, "graspplanner/hndovrinfo")):
            os.mkdir(os.path.join(config.ROOT, "graspplanner/hndovrinfo"))
        with open(os.path.join(config.ROOT, "graspplanner/hndovrinfo", self.objname + "_hndovrinfo.pkl"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def gethandover(self):
        """
        io interface to load the previously planned data

        :return:

        author: hao, refactored by weiwei
        date: 20191206, 20191212
        """

        with open(os.path.join(config.ROOT, "graspplanner/hndovrinfo", self.objname + "_hndovrinfo.pkl"), "rb") as file:
            self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
            self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
            self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft = pickle.load(file)

        return self.identityglist_rgt, self.identityglist_lft, self.fpsnpmat4, \
               self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
               self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
               self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft

    def __genidentitygplist(self):
        """
        fill up self.identitygplist

        :return:

        author: weiwei
        date: 20191212
        """

        rgthnd = self.rgthnd
        lfthnd = self.lfthnd
        pairidlist = list(itertools.product(range(len(self.identityglist_rgt)), range(len(self.identityglist_lft))))
        for i in range(len(pairidlist)):
            print("generating identity gplist...", i, len(pairidlist))
            # Check whether the hands collide with each or not
            ir, il = pairidlist[i]
            rgthnd.setMat(base.pg.np4ToMat4(self.identityglist_rgt[ir][2]))
            rgthnd.setjawwidth(self.identityglist_rgt[ir][0])
            lfthnd.setMat(base.pg.np4ToMat4(self.identityglist_lft[il][2]))
            lfthnd.setjawwidth(self.identityglist_lft[il][0])
            ishndcollided = self.bcdchecker.isMeshListMeshListCollided(rgthnd.cmlist, lfthnd.cmlist)
            if not ishndcollided:
                self.identitygplist.append(pairidlist[i])

    def __genfpsnestedglist(self):
        """
        generate the grasp list for the floating poses

        :return:

        author: hao chen, revised by weiwei
        date: 20191122
        """

        self.fpsnestedglist_rgt = {}
        self.fpsnestedglist_lft = {}
        for posid, icomat4 in enumerate(self.fpsnpmat4):
            print("generating nested glist at the floating poses...", posid, len(self.fpsnpmat4))
            glist = []
            for jawwidth, fc, homomat in self.identityglist_rgt:
                approach_direction = homomat[:3, 2]
                tippos = rm.homotransformpoint(icomat4, fc)
                homomat = np.dot(icomat4, homomat)
                approach_direction = np.dot(icomat4[:3, :3], approach_direction)
                glist.append([jawwidth, tippos, homomat, approach_direction])
            self.fpsnestedglist_rgt[posid] = glist
            glist = []
            for jawwidth, fc, homomat in self.identityglist_lft:
                approach_direction = homomat[:3, 2]
                tippos = rm.homotransformpoint(icomat4, fc)
                homomat = np.dot(icomat4, homomat)
                approach_direction = np.dot(icomat4[:3, :3], approach_direction)
                glist.append([jawwidth, tippos, homomat, approach_direction])
            self.fpsnestedglist_lft[posid] = glist

    def __checkik(self):
        # Check the IK of both hand in the handover pose
        ### right hand
        self.ikfid_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_lft = {}
        for posid in self.fpsnestedglist_rgt.keys():
            print("checkik rgt...", posid, len(self.fpsnestedglist_rgt.keys()))
            armname = 'rgt'
            fpglist_thispose = self.fpsnestedglist_rgt[posid]
            for i, [_, tippos, homomat, handa] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -handa
                minusworldy = np.array([0, -1, 0])
                if rm.degree_betweenvector(handa, minusworldy) < 90:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_rgt:
                                self.ikfid_fpsnestedglist_rgt[posid] = []
                            self.ikfid_fpsnestedglist_rgt[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_rgt:
                                self.ikjnts_fpsnestedglist_rgt[posid] = {}
                            self.ikjnts_fpsnestedglist_rgt[posid][i] = [msc, msc_handa]

        ### left hand
        for posid in self.fpsnestedglist_lft.keys():
            print("checkik lft...", posid, len(self.fpsnestedglist_lft.keys()))
            armname = 'lft'
            fpglist_thispose = self.fpsnestedglist_lft[posid]
            for i, [_, tippos, homomat, handa] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -handa
                plusworldy = np.array([0, 1, 0])
                if rm.degree_betweenvector(handa, plusworldy) < 90:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_lft:
                                self.ikfid_fpsnestedglist_lft[posid] = []
                            self.ikfid_fpsnestedglist_lft[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_lft:
                                self.ikjnts_fpsnestedglist_lft[posid] = {}
                            self.ikjnts_fpsnestedglist_lft[posid][i] = [msc, msc_handa]

    def checkhndenvcollision(self, homomat, obstaclecmlist, armname="rgt", debug=False):
        """

        :param homomat:
        :param obstaclecmlist:
        :return:

        author: ruishuang
        date: 20191122
        """

        if armname == "rgt":
            handtmp = self.rgthnd
        else:
            handtmp = self.lfthnd
        handtmp.sethomomat(homomat)
        handtmp.setjawwidth(handtmp.jawwidthopen)
        iscollided = self.bcdchecker.isMeshListMeshListCollided(handtmp.cmlist, obstaclecmlist)
        if debug:
            if iscollided:
                handtmp.setColor(1, 0, 0, .2)
                handtmp.reparentTo(base.render)
                # base.run()
            else:
                handtmp.setColor(0, 1, 0, .2)
                handtmp.reparentTo(base.render)

        return iscollided
