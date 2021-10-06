import math
import warnings as wns
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import motionplanner.robot_helper as rbt_helper
import basis.robot_math as rm
import copy
import localenv.envloader as el

TB = True


class FkOptimizer(object):
    def __init__(self, env, rbt, armname="lft_arm", releemat4=np.eye(4), toggledebug=TB,
                 col_ps=None, roll_limit=1e-2, pos_limit=1e-2):
        self.rbt = rbt
        self.armname = armname
        self.env = env
        self.armname = armname
        self.releemat4 = releemat4
        self.toggledebug = toggledebug
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)

        self.result = None
        self.cons = []
        b = (-360, 360)
        self.bnds = (b, b, b, b, b, b)

        self.seedjntagls = None
        self.tgtpos = None
        self.tgtrot = None

        self.jnts = []
        self.rot_err = []
        self.pos_err = []
        self.min_dist_hand = []
        self.min_dist_tool = []
        self.move_angle = []

        self.roll_limit = roll_limit
        self.pos_limit = pos_limit
        if col_ps is not None:
            self.col_ps = np.asarray(col_ps)
        else:
            self.col_ps = None

    def objctive(self, x):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        return np.linalg.norm(x - self.seedjntagls)

    def update_known(self, seedjntagls, tgtpos, tgtrot=None, movedir=None):
        self.seedjntagls = seedjntagls
        self.tgtpos = tgtpos
        self.tgtrot = tgtrot
        self.movedir = movedir

    def con_rot(self, x):
        eepos, eerot = self.rbth.get_ee(x, self.releemat4)
        err = np.degrees(rm.angle_between_vectors(eerot[:3, 0], self.tgtrot[:3, 0]))
        self.rot_err.append(err)
        return self.roll_limit - err

    def con_pos(self, x):
        eepos, eerot = self.rbth.get_ee(x, self.releemat4)
        err = np.linalg.norm(self.tgtpos - eepos)
        self.pos_err.append(err)
        return self.pos_limit - err

    def __ps2seg_min_dist(self, p1, p2, ps):
        p1_p = np.asarray([p1] * len(ps) - ps)
        p2_p = np.asarray([p2] * len(ps) - ps)
        p1_p_norm = np.linalg.norm(p1_p, axis=1)
        p2_p_norm = np.linalg.norm(p2_p, axis=1)
        p2_p1 = np.asarray([p2 - p1] * len(ps))
        dist_list = abs(np.linalg.norm(np.cross(p2_p1, p1_p), axis=1) / np.linalg.norm(p2 - p1))

        l1 = np.arccos(
            np.sum((p1_p / p1_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1)
        )
        l2 = np.arccos(
            np.sum((p2_p / p2_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1)
        )
        l1 = (l1[:] < math.pi / 2).astype(int)
        l2 = (l2[:] > math.pi / 2).astype(int)

        dist_list = np.multiply(p1_p_norm, l1) + np.multiply(p2_p_norm, l2) + np.multiply(dist_list, 1 - l1 - l2)
        min_dist = min(dist_list)
        if self.toggledebug:
            if min_dist < 10:
                i = list(dist_list).index(min_dist)
                print(l1[i], l2[i], min_dist)
                base.pggen.plotArrow(base.render, spos=p1, epos=p2, rgba=(1, 0, 0, 1))
                base.pggen.plotSphere(base.render, self.col_ps[i], rgba=(1, 0, 0, 1), radius=20)
                self.rbth.show_armjnts(rgba=(1, 1, 0, .2))
        return min_dist

    def con_hand(self, x):
        p1, _ = self.rbth.get_ee(x)
        p2, _ = self.rbth.get_tcp(x)
        min_dist = self.__ps2seg_min_dist(p1, p2, self.col_ps)
        self.min_dist_hand.append(min_dist)

        return min_dist - 50

    def con_tool(self, x):
        eepos, eerot = self.rbth.get_ee(x, releemat4=self.releemat4)
        p1 = eepos + eerot[:3, 0] * 15
        p2 = eepos + eerot[:3, 0] * 135
        min_dist_tool = self.__ps2seg_min_dist(p1, p2, self.col_ps)
        self.min_dist_tool.append(min_dist_tool)

        return min_dist_tool - 10

    def con_movement(self, x):
        tcppos, tcprot = self.rbth.get_tcp(armjnts=x)
        angle = rm.angle_between_vectors(self.movedir, tcprot[:, 2])
        self.move_angle.append(angle)
        sin = math.sin(angle)
        return sin - 0.9

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, seedjntagls, tgtpos, tgtrot=None, movedir=None, method='SLSQP'):
        """

        :param seedjntagls:
        :param tgtpos:
        :param tgtrot:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(seedjntagls, tgtpos, tgtrot, movedir)
        self.addconstraint(self.con_rot, condition="ineq")
        self.addconstraint(self.con_pos, condition="ineq")
        if self.col_ps is not None:
            self.addconstraint(self.con_hand, condition="ineq")
            if not (self.releemat4 == np.eye(4)).all():
                self.addconstraint(self.con_tool, condition="ineq")
        if movedir is not None:
            self.addconstraint(self.con_movement, condition="ineq")

        # iks = IkSolver(self.env, self.rbt, self.rbtmg, self.rbtball, self.armname)
        # q0 = iks.solve_numik3(self.tgtpos, tgtrot=None, seedjntagls=self.seedjntagls, releemat4=self.releemat4)
        sol = minimize(self.objctive, seedjntagls, method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        if self.toggledebug:
            print(sol)
            self.__debug()
            base.run()

        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

    def __debug(self):
        for p in self.col_ps:
            base.pggen.plotSphere(base.render, p, rgba=(1, 1, 0, .5), radius=20)
        fig = plt.figure(1, figsize=(6.4 * 3, 4.8 * 1.5))
        plt.subplot(231)
        self.rbth.plot_vlist(self.pos_err, title="translation error")
        plt.subplot(232)
        self.rbth.plot_vlist(self.rot_err, title="rotation error")
        plt.subplot(233)
        self.rbth.plot_vlist(self.min_dist_hand, title="min distance")
        plt.subplot(234)
        self.rbth.plot_vlist(self.min_dist_tool, title="min distance(tool)")
        plt.subplot(235)
        self.rbth.plot_vlist(np.degrees(self.move_angle), title="angle")
        plt.subplot(236)
        self.rbth.plot_armjnts(self.jnts)
        plt.show()


class NsOptimizer(object):
    def __init__(self, env, rbt, armname="lft", releemat4=np.eye(4), toggledebug=False):
        self.rbt = rbt
        self.armname = armname
        self.env = env
        self.armname = armname
        self.releemat4 = releemat4
        self.toggledebug = toggledebug
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)

        self.result = None
        self.cons = []
        b = (-math.pi, math.pi)
        self.bnds = (b, b, b, b, b, b)

        self.seedjntagls = None
        self.tgtpos = None
        self.tgtrot = None

        self.jnts = []
        self.rot_err = []
        self.pos_err = []
        self.dx_norm = []
        self.x = []

        self.ROLL_LIMIT = 5
        self.POS_LIMIT = 0.1

        self.q0 = np.zeros(6)
        self.jsharp = None
        self.armjac = None
        self.tmpq = np.zeros(6)

    def objctive(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        print("-------------")
        print(self.q0)
        print(x)
        print(q)
        self.jnts.append(np.degrees(q))
        self.x.append(np.degrees(x))
        self.dx_norm.append(np.linalg.norm(self.q0 - np.degrees(q)))
        if self.pos_err[-1] > 15:
            self.rbth.show_armjnts(armjnts=self.jnts[-1], rgba=(.7, .7, .7, .2))
        return np.linalg.norm(np.degrees(q) - self.seedjntagls)

    def update_known(self, seedjntagls, tgtpos, tgtrot=None):
        self.seedjntagls = seedjntagls
        self.tgtpos = tgtpos
        self.tgtrot = tgtrot

    def con_rot(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        eepos, eerot = self.rbth.get_ee(armjnts=np.degrees(q), releemat4=self.releemat4)
        err = np.degrees(rm.angle_between_vectors(eerot[:3, 0], self.tgtrot[:3, 0]))
        self.rot_err.append(err)
        return self.ROLL_LIMIT - err

    def con_pos(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        eepos, eerot = self.rbth.get_ee(armjnts=np.degrees(q), releemat4=self.releemat4)
        err = np.linalg.norm(self.tgtpos - eepos)
        self.pos_err.append(err)
        return self.pos_err[0] - err

    def con_jntlim0(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[0]))

    def con_jntlim1(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[1]))

    def con_jntlim2(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[2]))

    def con_jntlim3(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[3]))

    def con_jntlim4(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[4]))

    def con_jntlim5(self, x):
        q = np.radians(self.q0) + (np.identity(x.shape[0]) - self.jsharp.dot(self.armjac)).dot(x)
        return 360 - np.degrees(abs(q[5]))

    def con_collision(self, x):
        if self.rbth.is_selfcollided(armjnts=np.degrees(self.q0 + np.degrees(x))):
            return -1
        else:
            return 1

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def update_rels(self):
        iks = IkSolver(self.env, self.rbt, self.armname)

        self.q0 = iks.solve_numik3(self.tgtpos, tgtrot=None, seedjntagls=self.seedjntagls, releemat4=self.releemat4)
        eepos, eerot = self.rbth.get_ee(armjnts=self.q0, releemat4=self.releemat4)
        self.pos_err.append(np.linalg.norm(self.tgtpos - eepos))
        self.rot_err.append(np.degrees(rm.angle_between_vectors(eerot[:3, 0], self.tgtrot[:3, 0])))
        self.jnts.append(self.q0)

        self.rbth.goto_armjnts(armjnts=self.q0)
        self.armjac = self.rbth.jacobian(self.releemat4, scale=0.001)[:3, :]
        jjt = self.armjac.dot(self.armjac.T)
        dampercoeff = 1e-6  # a non-zero regulation coefficient
        damper = dampercoeff * np.identity(jjt.shape[0])
        self.jsharp = self.armjac.T.dot(np.linalg.inv(jjt + damper))

    def solve(self, seedjntagls, tgtpos, tgtrot=None, method='SLSQP'):
        """

        :param seedjntagls:
        :param tgtpos:
        :param tgtrot:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()

        self.update_known(seedjntagls, tgtpos, tgtrot)
        self.update_rels()

        self.addconstraint(self.con_rot, condition="ineq")
        self.addconstraint(self.con_pos, condition="ineq")
        # self.addconstraint(self.con_jntlim0, condition="ineq")
        # self.addconstraint(self.con_jntlim1, condition="ineq")
        # self.addconstraint(self.con_jntlim2, condition="ineq")
        # self.addconstraint(self.con_jntlim3, condition="ineq")
        # self.addconstraint(self.con_jntlim4, condition="ineq")
        # self.addconstraint(self.con_jntlim5, condition="ineq")
        # self.addconstraint(self.con_collision, condition="ineq")
        # self.rbth.show_armjnts(armjnts=self.q0, rgba=(1, 1, 0, .5))
        # base.run()
        sol = minimize(self.objctive, np.zeros(6), method=method, bounds=self.bnds, constraints=self.cons)
        q = np.degrees(np.radians(self.q0) + (np.identity(sol.x.shape[0]) - self.jsharp.dot(self.armjac)).dot(sol.x))
        print("time cost", time.time() - time_start)

        if self.toggledebug:
            print(sol)
            self.__debug()

        if sol.success:
            self.rbth.show_armjnts(armjnts=q, rgba=(0, 1, 0, .5))
            return q, sol.fun
        else:
            self.rbth.show_armjnts(armjnts=q, rgba=(1, 0, 0, .5))
            return None, None

    def __debug(self):
        fig = plt.figure(1, figsize=(6.4 * 3, 4.8 * 2))
        plt.subplot(231)
        self.rbth.plot_vlist(self.pos_err, title="translation error")
        plt.subplot(232)
        self.rbth.plot_vlist(self.rot_err, title="rotation error")
        plt.subplot(233)
        self.rbth.plot_vlist(self.dx_norm, title="dx norm")
        plt.subplot(234)
        self.rbth.plot_armjnts(self.x, show=False, title="x")
        plt.subplot(235)
        self.rbth.plot_armjnts(self.jnts, show=False)
        plt.show()


class IkSolver(object):
    def __init__(self, env, rbt, armname):
        self.rbt = rbt
        self.armname = armname
        self.env = env
        self.obscmlist = self.env.getstationaryobslist() + self.env.getchangableobslist()
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)

        self.initjnts = self.rbt.get_jnt_values(self.armname)
        # self.armlj = self.rbt.get_jnt_values

    def __initsearch(self, seedjntagls, tgtpos=None, tgtrot=None):
        if isinstance(seedjntagls, str):
            armjntsiter = self.rbt.getinitarmjnts(self.armname)
            if seedjntagls is "ccd":
                self.__ccdinitik(tgtpos, tgtrot)
            seedjntagls = self.rbt.getarmjnts(self.armname)
        elif seedjntagls is None:
            armjntsiter = self.rbt.getinitarmjnts(self.armname)
            seedjntagls = self.rbt.getarmjnts(self.armname)
        else:
            armjntsiter = copy.deepcopy(seedjntagls)
        self.rbt.movearmfk(armjntsiter, self.armname)
        return seedjntagls, armjntsiter

    def tcperror(self, tgtpos, tgtrot, releemat4=np.eye(4), scale=1.0):
        return self.rbth.tcperror(tgtpos, tgtrot, releemat4, scale)

    def jacobian(self, releemat4=np.eye(4), scale=1.0):
        return self.rbth.jacobian(releemat4, scale)

    def __ccdinitik(self, tgtpos, tgtrot):
        """
        use ccd for initialization

        :param robot:
        :param tgtpos:
        :param tgtrot:
        :param armname:
        :return:
        """
        armjntsiter = self.initjnts
        counter = -1
        for jid in self.rbt.targetjoints[::-1]:
            counter += 1
            goalvec = rm.unit_vector(tgtpos - self.armlj[jid]["linkpos"])
            initvec = rm.unit_vector(self.armlj[self.rbt.targetjoints[-1]]["linkend"] - self.armlj[jid]["linkpos"])
            rotvec = np.dot(self.armlj[jid]["rotmat"], self.armlj[jid]["rotax"])
            dq = np.zeros_like(self.rbt.targetjoints)
            if np.allclose(rotvec, initvec) or np.allclose(rotvec, goalvec):
                continue
            else:
                goalanglevec = np.cross(goalvec, rotvec)
                initanglevec = np.cross(initvec, rotvec)
                tmpvalue = rm.degree_betweenvector(initanglevec, goalanglevec)
                if tmpvalue is None:
                    continue
                else:
                    dq[counter] = tmpvalue
            armjntsiter += dq
            armjntsiter = rm.cvtRngPM360(armjntsiter)
            bdragged, jntangles = self.rbt.chkrngdrag(armjntsiter, self.armname)
            armjntsiter[:] = jntangles[:]
            self.rbt.movearmfk(jntangles, self.armname)

    def solve_numik(self, tgtpos, tgtrot=None, seedjntagls="default", localminima="accept", releemat4=np.eye(4),
                    toggledebug=False):
        """
        solve the ik numerically for the specified armname with manually specified starting configuration (msc)

        :param tgtpos: the position of the goal, 1-by-3 numpy ndarray
        :param tgtrot: the orientation of the goal, 3-by-3 numpyndarray
        :param seedjntagls: "default", "ccd", or a given jntangles nparray
        :param localminima: "randomrestart", "accept"
        :param releemat4:
        :return:
        """

        if toggledebug:
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []

        # selection option, for cropping the jacobian matrices and error vectors
        sopt = [0, 3] if tgtrot is None else [0, 6]  # [startid, endid]
        if tgtrot is None:  # use dummy tgt rot
            tgtrot = self.rbt.getarm(self.armname)[self.rbt.targetjoints[-1]]["rotmat"]

        deltapos = tgtpos - self.rbt.getarm(self.armname)[1]["linkpos"]
        if np.linalg.norm(deltapos) > 800.0:
            return None

        armjntssave = self.rbt.getarmjnts(self.armname)
        seedjntagls, armjntsiter = self.__initsearch(seedjntagls, tgtpos, tgtrot)

        wt_pos = 1 / 40000  # 1/200*1/200
        wt_ang = 1 / (math.pi * math.pi)  # 1/pi*1/pi 200->1; pi->1
        wtdiagmat = np.diag([wt_pos, wt_pos, wt_pos, wt_ang, wt_ang, wt_ang][sopt[0]:sopt[1]])
        largesterr = np.asarray([500, 500, 500, math.pi, math.pi, math.pi][sopt[0]:sopt[1]])
        largesterrnorm = largesterr.T.dot(wtdiagmat).dot(largesterr)
        errnormlast = 0.0

        for i in range(10000):
            armjac = self.jacobian(releemat4, scale=1)[sopt[0]:sopt[1], :]
            err = self.tcperror(tgtpos, tgtrot, releemat4=releemat4, scale=1)[sopt[0]:sopt[1]]
            errnorm = err.T.dot(wtdiagmat).dot(err)
            if toggledebug:
                ajpath.append(self.rbt.getarmjnts(armname=self.armname))
            if errnorm < 1e-6:
                armjntsreturn = self.rbt.getarmjnts(self.armname)
                self.rbt.movearmfk(armjntssave, self.armname)
                if toggledebug:
                    self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                return armjntsreturn
            else:
                # judge local minima
                if abs(errnorm - errnormlast) < 1e-6:
                    if localminima is "accept":
                        wns.warn("Bypassing local minima! The return value is a local minima, "
                                 "rather than the exact IK result.")
                        armjntsreturn = self.rbt.getarmjnts(self.armname)
                        self.rbt.movearmfk(armjntssave, self.armname)
                        if toggledebug:
                            self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                        return armjntsreturn
                    elif localminima is "randomrestart":
                        wns.warn("Random restart at local minima!")
                        armjnts = self.rbt.randompose(self.armname)
                        armjntsiter[:] = armjnts[:]
                        self.rbt.movearmfk(armjnts, self.armname)
                        continue
                    else:
                        break
                else:
                    strecthingcoeff = 1 / (1 + math.exp(errnorm / largesterrnorm))
                    # regularized least square to avoid singularity
                    # note1: do not use np.linalg.inv since it is not precise
                    # note2: use np.linalg.solve if the system is exactly determined, it is faster
                    # note3: use np.linalg.lstsq if there might be singularity (no regularization)
                    # see https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress
                    regcoeff = strecthingcoeff / 500 + 0.0001  # a non-zero regulation coefficient
                    # regulator = regcoeff*np.identity(armjac.shape[1])
                    # jtj_regcoeff = armjac.T.dot(wtdiagmat).dot(armjac)+regulator
                    # jstar = np.linalg.inv(jtj_regcoeff).dot(armjac.T.dot(wtdiagmat)) # lft moore-penrose inverse
                    # dq = jstar.dot(err)
                    # remove the shortest projection on the null space to get minimal motion
                    # lft moore-penrose inverse
                    jtj = armjac.T.dot(armjac)
                    regulator = regcoeff * np.identity(jtj.shape[0])
                    jstar = np.linalg.inv(jtj + regulator).dot(armjac.T)
                    # dq = jstar.dot(err)
                    # dq = strecthingcoeff*2*np.degrees(dq)
                    dq = np.radians(rm.cvtRngPM360(jstar.dot(err)))
                    dq0 = 0.3 * np.radians(seedjntagls - armjntsiter)
                    dq_minimized = strecthingcoeff * 2 * np.degrees(
                        dq + (np.identity(dq0.shape[0]) - jstar.dot(armjac)).dot(dq0))
                    if toggledebug:
                        dqbefore.append(np.degrees(dq))
                        dqcorrected.append(np.degrees(dq_minimized))
                        dqnull.append(np.degrees(dq0))
                armjntsiter += dq_minimized
                # armjntsiter = rm.cvtRngPM180(armjntsiter)
                # armjntsiter = rm.cvtRngPM360(armjntsiter)
                isdragged, jntanglesdragged = self.rbt.chkrngdrag(armjntsiter, self.armname)
                armjntsiter[:] = jntanglesdragged[:]
                self.rbt.movearmfk(jntanglesdragged, self.armname)
            errnormlast = errnorm
        self.rbt.movearmfk(armjntssave, self.armname)

        return None

    def solve_numik2(self, tgtpos, tgtrot=None, seedjntagls="default", releemat4=np.eye(4),
                     localminima="accept", allowtoggleoffnullspace=True, toggledebug=False):
        """
        solve the ik numerically for the specified armname

        :param tgtpos: the position of the goal, 1-by-3 numpy ndarray
        :param tgtrot: the orientation of the goal, 3-by-3 numpyndarray
        :param seedjntagls: "default", "ccd", or a given jntangles nparray
        :param localminima: "randomrestart", "accept"
        :param allowtoggleoffnullspace:
        :param toggledebug: show motion curve or not
        :return: armjnts: a 1-by-x numpy ndarray

        author: weiwei
        date: 20180203, 20200328, 20201113
        """

        if toggledebug:
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []

        # selection option, for cropping the jacobian matrices and error vectors
        sopt = [0, 3] if tgtrot is None else [0, 6]  # [startid, endid]
        if tgtrot is None:  # use dummy tgt rot
            tgtrot = self.rbt.getarm(self.armname)[self.rbt.targetjoints[-1]]["rotmat"]

        deltapos = tgtpos - self.rbt.getarm(self.armname)[1]["linkpos"]
        if np.linalg.norm(deltapos) > 800.0:
            return None

        armjntssave = self.rbt.getarmjnts(self.armname)
        seedjntagls, armjntsiter = self.__initsearch(seedjntagls, tgtpos, tgtrot)

        wt_pos = 1  # 1/1*1/1
        wt_ang = 1 / (4 * math.pi * math.pi)  # 1/(2*pi)*1/(2*pi) 1m->1; pi->1
        wtdiagmat = np.diag([wt_pos, wt_pos, wt_pos, wt_ang, wt_ang, wt_ang][sopt[0]:sopt[1]])
        largesterr = np.asarray([.7, .7, .7, math.pi, math.pi, math.pi][sopt[0]:sopt[1]])
        largesterrnorm = largesterr.T.dot(wtdiagmat).dot(largesterr)
        errnormlast = 0.0
        isuingnullspace = True
        jointweights = np.ones_like(armjntsiter)
        jointweights[-4:] = np.array([.5, .1, .01, .01])

        for i in range(100):
            armjac = self.jacobian(releemat4, scale=0.001)[sopt[0]:sopt[1], :]
            err = self.tcperror(tgtpos, tgtrot, releemat4=releemat4, scale=0.001)[sopt[0]:sopt[1]]
            errnorm = err.T.dot(wtdiagmat).dot(err)
            if toggledebug:
                ajpath.append(self.rbt.getarmjnts(armname=self.armname))
            if errnorm < 1e-6:
                if toggledebug:
                    self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                armjntsreturn = self.rbt.getarmjnts(self.armname)
                self.rbt.movearmfk(armjntssave, self.armname)
                return armjntsreturn
            else:
                # judge local minima
                if abs(errnorm - errnormlast) < 1e-12:
                    if toggledebug:
                        self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                    if errnorm < 1e-4:
                        wns.warn("Local minima! The return value is a local minima, "
                                 "rather than the exact IK result. Precision: 1e-4")
                        armjntsreturn = self.rbt.getarmjnts(self.armname)
                        self.rbt.movearmfk(armjntssave, self.armname)
                        return armjntsreturn
                    elif isuingnullspace and allowtoggleoffnullspace:
                        wns.warn("Try toggling off null space!")
                        isuingnullspace = False
                        continue
                    elif not isuingnullspace:
                        if localminima is "randomrestart":
                            wns.warn("Random restart!")
                            armjnts = self.rbt.randompose(self.armname)
                            armjntsiter[:] = armjnts[:]
                            self.rbt.movearmfk(armjnts, self.armname)
                            continue
                        elif localminima is "accept":
                            wns.warn("Local minima! The return value is a local minima, "
                                     "rather than the exact IK result. Precision less than 1e-4")
                            armjntsreturn = self.rbt.getarmjnts(self.armname)
                            self.rbt.movearmfk(armjntssave, self.armname)
                            return armjntsreturn
                        else:
                            break
                    else:
                        break
                else:
                    # -- notes --
                    ## note1: do not use np.linalg.inv since it is not precise
                    ## note2: use np.linalg.solve if the system is exactly determined, it is faster
                    ## note3: use np.linalg.lstsq if there might be singularity (no regularization)
                    ## see https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress
                    ## 20201113 did not figure out a way to solve the moore-penrose inverse using np.linalg.solve
                    strecthingcoeff = 1 / (1 + math.exp(errnorm / largesterrnorm))
                    dampercoeff = (strecthingcoeff + 1) * 1e-6  # a non-zero regulation coefficient
                    # -- lft moore-penrose inverse --
                    ## jtj = armjac.T.dot(armjac)
                    ## regulator = regcoeff*np.identity(jtj.shape[0])
                    ## jstar = np.linalg.inv(jtj+regulator).dot(armjac.T)
                    ## dq = jstar.dot(err)
                    # -- rgt moore-penrose inverse --
                    jjt = armjac.dot(armjac.T)
                    damper = dampercoeff * np.identity(jjt.shape[0])
                    jsharp = armjac.T.dot(np.linalg.inv(jjt + damper))
                    dq = jsharp.dot(err)
                    dq = rm.regulate_angle(-math.pi, math.pi, dq)
                    # dq = Jsharp dx+(I-Jsharp J)dq0
                    # see https://www.slideserve.com/marietta/kinematic-redundancy
                    dqref = np.multiply(np.radians(armjntssave - armjntsiter), jointweights)
                    dqref_on_ns = (np.identity(dqref.shape[0]) - jsharp.dot(armjac)).dot(dqref)
                    dqref_on_ns = rm.regulate_angle(-math.pi, math.pi, dqref_on_ns)
                    dq_minimized = strecthingcoeff * (dq + dqref_on_ns)
                    if toggledebug:
                        dqbefore.append(np.degrees(dq))
                        dqcorrected.append(np.degrees(dq_minimized))
                        dqnull.append(np.degrees(dqref_on_ns))
                if isuingnullspace:
                    armjntsiter += np.degrees(dq)
                else:
                    armjntsiter += np.degrees(dq_minimized)
                armjntsiter = rm.regulate_angle(-180, 180, armjntsiter)
                isdragged, jntanglesdragged = self.rbt.chkrngdrag(armjntsiter, self.armname)
                armjntsiter[:] = jntanglesdragged[:]
                self.rbt.movearmfk(jntanglesdragged, self.armname)
            errnormlast = errnorm
        self.rbt.movearmfk(armjntssave, self.armname)
        if toggledebug:
            self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
        return None

    def solve_numik3(self, tgtpos, tgtrot=None, seedjntagls="default", releemat4=np.eye(4),
                     localminima="accept", allowtoggleoffnullspace=True, toggledebug=False):
        """
        solve the ik numerically for the specified armname
        :param robot: see the ur3dual.Ur3SglRobot class
        :param tgtpos: the position of the goal, 1-by-3 numpy ndarray
        :param tgtrot: the orientation of the goal, 3-by-3 numpyndarray
        :param seedjntagls: "default"(None), "ccd", or a given jntangles nparray
        :param armname: a string "rgt" or "lft" indicating the arm that will be solved
        :param localminima: "randomrestart", "accept"
        :param allowtoggleoffnullspace:
        :param toggledebug: show motion curve or not
        :return: armjnts: a 1-by-x numpy ndarray
        author: weiwei
        date: 20180203, 20200328, 20201113
        """
        _WT_POS = 0.628  # 0.628m->1 == 0.01->0.00628m
        _WT_AGL = 1 / (math.pi * math.pi)  # pi->1 == 0.01->0.18degree
        _WT_MAT = np.diag([_WT_POS, _WT_POS, _WT_POS, _WT_AGL, _WT_AGL, _WT_AGL])
        _MAX_RNG = 1.5

        if toggledebug:
            dqbefore = []
            dqcorrected = []
            dqnull = []
            ajpath = []

        # selection option, for cropping the jacobian matrices and error vectors
        sopt = [0, 3] if tgtrot is None else [0, 6]  # [startid, endid]
        if tgtrot is None:  # use dummy tgt rot
            tgtrot = self.rbt.getarm(self.armname)[self.rbt.targetjoints[-1]]["rotmat"]

        deltapos = tgtpos - self.rbt.getarm(self.armname)[1]["linkpos"]
        if np.linalg.norm(deltapos) * 0.001 > _MAX_RNG:
            wns.warn("The goal is outside maximum range!")
            return None
        armjntssave = self.rbt.getarmjnts(self.armname)
        seedjntagls, armjntsiter = self.__initsearch(seedjntagls, tgtpos, tgtrot)
        armjntsref = armjntsiter.copy()
        wtdiagmat = _WT_MAT[sopt[0]:sopt[1], sopt[0]:sopt[1]]
        errnormlast = 0.0
        isuingnullspace = True

        for i in range(100):
            armjac = self.jacobian(releemat4, scale=0.001)[sopt[0]:sopt[1], :]
            err = self.tcperror(tgtpos, tgtrot, releemat4=releemat4, scale=0.001)[sopt[0]:sopt[1]]
            errnorm = err.T.dot(wtdiagmat).dot(err)
            err = .05 / errnorm * err if errnorm > .05 else err
            if toggledebug:
                print(errnorm)
                ajpath.append(self.rbt.getarmjnts(armname=self.armname))
            if errnorm < 1e-6:
                if toggledebug:
                    self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                armjntsreturn = self.rbt.getarmjnts(self.armname)
                self.rbt.movearmfk(armjntssave, self.armname)
                return armjntsreturn
            else:
                # judge local minima
                if abs(errnorm - errnormlast) < 1e-12:
                    if toggledebug:
                        self.__debug(i, dqbefore, dqnull, dqcorrected, ajpath)
                    if localminima is "nullaccept":
                        wns.warn("Local minima in null space! The return value is a local minima, "
                                 "rather than the exact IK result.")
                        armjntsreturn = self.rbt.getarmjnts(self.armname)
                        self.rbt.movearmfk(armjntssave, self.armname)
                        return armjntsreturn
                    elif errnorm < 1e-4:
                        wns.warn("Local minima! The return value is a local minima, "
                                 "rather than the exact IK result. Precision: 1e-4")
                        armjntsreturn = self.rbt.getarmjnts(self.armname)
                        self.rbt.movearmfk(armjntssave, self.armname)
                        return armjntsreturn
                    elif isuingnullspace and allowtoggleoffnullspace:
                        wns.warn("Try toggling off null space!")
                        isuingnullspace = False
                        continue
                    elif not isuingnullspace:
                        if localminima is "randomrestart":
                            wns.warn("Random restart!")
                            armjnts = self.rbt.randompose(self.armname)
                            armjntsiter[:] = armjnts[:]
                            self.rbt.movearmfk(armjnts, self.armname)
                            continue
                        elif localminima is "accept":
                            wns.warn("Local minima! The return value is a local minima, "
                                     "rather than the exact IK result. Precision less than 1e-4")
                            armjntsreturn = self.rbt.getarmjnts(self.armname)
                            self.rbt.movearmfk(armjntssave, self.armname)
                            return armjntsreturn
                        else:
                            break
                    else:
                        break
                else:
                    # -- notes --
                    ## note1: do not use np.linalg.inv since it is not precise
                    ## note2: use np.linalg.solve if the system is exactly determined, it is faster
                    ## note3: use np.linalg.lstsq if there might be singularity (no regularization)
                    ## see https://stackoverflow.com/questions/34170618/normal-equation-and-numpy-least-squares-solve-methods-difference-in-regress
                    ## 20201113 did not figure out a way to solve the moore-penrose inverse using np.linalg.solve
                    strecthingcoeff = 1 / (1 + math.exp(1 / ((errnorm / _MAX_RNG) * 1000 + 1)))
                    dampercoeff = (strecthingcoeff + 1) * 1e-6  # a non-zero regulation coefficient
                    # -- lft moore-penrose inverse --
                    ## jtj = armjac.T.dot(armjac)
                    ## regulator = regcoeff*np.identity(jtj.shape[0])
                    ## jstar = np.linalg.inv(jtj+regulator).dot(armjac.T)
                    ## dq = jstar.dot(err)
                    # -- rgt moore-penrose inverse --
                    jjt = armjac.dot(armjac.T)
                    damper = dampercoeff * np.identity(jjt.shape[0])
                    jsharp = armjac.T.dot(np.linalg.inv(jjt + damper))
                    dq = strecthingcoeff * jsharp.dot(err)
                    # dq = rm.regulate_angle(-math.pi, math.pi, dq)
                    # dq = Jsharp dx+(I-Jsharp J)dq0
                    # see https://www.slideserve.com/marietta/kinematic-redundancy
                    dqref = np.radians(np.asarray(armjntsref) - np.asarray(armjntsiter))
                    dqref_on_ns = (np.identity(dqref.shape[0]) - jsharp.dot(armjac)).dot(dqref)
                    # dqref_on_ns = rm.regulate_angle(-math.pi, math.pi, dqref_on_ns)
                    dq_minimized = (dq + dqref_on_ns)
                    # dq_minimized = dq+dq.dot(dqref_on_ns)
                    if toggledebug:
                        dqbefore.append(np.degrees(dq))
                        dqcorrected.append(np.degrees(dq_minimized))
                        dqnull.append(np.degrees(dqref_on_ns))
                if not isuingnullspace:
                    armjntsiter += np.degrees(dq)
                else:
                    armjntsiter += np.degrees(dq_minimized)
                isdragged, jntanglesdragged = self.rbt.chkrngdrag(armjntsiter, self.armname)
                armjntsiter[:] = jntanglesdragged[:]
                self.rbt.movearmfk(jntanglesdragged, self.armname)
            errnormlast = errnorm
        self.rbt.movearmfk(armjntssave, self.armname)
        if toggledebug:
            self.__debug(np.inf, dqbefore, dqnull, dqcorrected, ajpath)
        return None

    def solve_numik4(self, tgtpos, tgtrot=None, seedjntagls="default", releemat4=np.eye(4), method="SLSQP",
                     toggledebug=False, col_ps=None, roll_limit=1e-2, pos_limit=1e-2, movedir=None):
        """

        :param tgtpos:
        :param tgtrot:
        :param seedjntagls:
        :param releemat4:
        :param method: 'SLSQP' or 'COBYLA'
        :param toggledebug:
        :return:
        """
        # toggledebug = True
        seedjntagls, armjntsiter = self.__initsearch(seedjntagls, tgtpos, tgtrot)
        # opt = NsOptimizer(self.env, self.rbt, self.rbtmg, self.rbtball, self.armname, releemat4=releemat4,
        #                   toggledebug=toggledebug)
        opt = FkOptimizer(self.env, self.rbt, self.armname, releemat4=releemat4,
                          toggledebug=toggledebug, col_ps=col_ps, roll_limit=roll_limit, pos_limit=pos_limit)
        q, cost = opt.solve(seedjntagls, tgtpos, tgtrot, movedir=movedir, method=method)
        if toggledebug:
            self.rbth.draw_axis(tgtpos, tgtrot)
        return q

    def __debug(self, i, dqbefore, dqnull, dqcorrected, ajpath):
        print("iteration times:", i)
        fig = plt.figure(figsize=(6.4 * 2, 4.8 * 1.5))
        axbefore = fig.add_subplot(221)
        axbefore.set_title("Original dq")
        axnull = fig.add_subplot(222)
        axnull.set_title("dqref on Null space")
        axcorrec = fig.add_subplot(223)
        axcorrec.set_title("Minimized dq")
        axaj = fig.add_subplot(224)
        axbefore.plot(dqbefore)
        axnull.plot(dqnull)
        axcorrec.plot(dqcorrected)
        axaj.plot(ajpath)
        plt.show()

    def evaluate(self, releemat4, func_name="3", show_armjnts=False):
        import time
        eepos, eerot = self.rbth.get_ee(releemat4=releemat4)
        self.rbth.draw_axis(eepos, eerot)
        sample_list = []
        diff_list = []
        time_cost_list = []
        for x in range(800, 1000, 100):
            for y in range(0, 100, 100):
                for z in range(900, 1100, 100):
                    for r in range(-90, 90 + 1, 90):
                        sample_list.append((x, y, z, r))

        for x, y, z, r in sample_list:
            time_start = time.time()
            _, tgtrot = self.rbth.get_tcp()
            tgtpos = np.asarray((x, y, z))
            tgtrot = np.dot(rm.rotmat_from_axangle((1, 0, 0), r), tgtrot)
            self.rbth.draw_axis(tgtpos, tgtrot, length=20)
            if func_name == "1":
                armjnts = self.solve_numik(tgtpos, tgtrot, releemat4=releemat4, toggledebug=False)
            elif func_name == "2":
                armjnts = self.solve_numik2(tgtpos, tgtrot, releemat4=releemat4, toggledebug=False)
            else:
                armjnts = self.solve_numik3(tgtpos, tgtrot, releemat4=releemat4, toggledebug=False)

            time_cost_list.append(time.time() - time_start)
            if armjnts is not None:
                if show_armjnts:
                    mp_lft.rbth.show_armjnts(armjnts=armjnts, toggleendcoord=False, rgba=(1, 0, 0, .5))
                eepos, eerot = self.rbth.get_ee(armjnts=armjnts, releemat4=releemat4)
                self.rbth.draw_axis(eepos, eerot, rgba=(1, 1, 0, .2))
                diff_list.append(np.linalg.norm(armjnts - self.initjnts))
                print("diff:", diff_list[-1])
            else:
                diff_list.append(None)
                print("Failed!")
            self.rbth.goto_initarmjnts()

        print(f"Success {len(diff_list) - diff_list.count(None)} of {len(diff_list)}")
        print(f"Avg. diff {np.average([v for v in diff_list if v is not None])}")
        print(f"Avg. time cost {np.average(time_cost_list)}")

        base.run()


if __name__ == '__main__':
    '''
    set up env and param
    '''
    from localenv import envloader as el
    import motionplanner.motion_planner as m_planner

    base, env = el.loadEnv_wrs()

    rbt = el.loadUr3e(showrbt=False)
    rbth = rbt_helper.RobotHelper(env, rbt, "lft_arm")
    iks_lft = IkSolver(env, rbt, "lft_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, "lft_arm")

    # releepos = np.asarray((0, 0, 20))
    # releerot = rm.rodrigues((0, 0, 1), 30)
    # releemat4 = rm.homomat_from_posrot(releepos, releerot)
    releemat4 = np.eye(4)

    # iks_lft.evaluate(releemat4, func_name="2")

    _, tgtrot = rbth.get_tcp()
    tgtpos = np.asarray((1000, 100, 1000))
    msc = iks_lft.solve_numik3(tgtpos, tgtrot, releemat4=releemat4)
    if msc is not None:
        eepos, eerot = rbth.get_ee(armjnts=msc, releemat4=releemat4)
        rbth.draw_axis(eepos, eerot, rgba=(1, 1, 0, .5))
        mp_lft.rbth.show_armjnts(armjnts=msc, toggleendcoord=False)

    tgtpos_2 = tgtpos + np.asarray((-100, 0, -50))
    rbth.draw_axis(tgtpos_2, tgtrot, rgba=(0, 1, 0, .5))
    # q_bs = iks_lft.solve_numik(tgtpos_2, tgtrot, seedjntagls=msc, releemat4=releemat4)
    # mp_lft.show_armjnts(armjnts=q_bs, rgba=(0, 1, 0, .5), toggleendcoord=True)
    # print(np.linalg.norm(q_bs - msc))
    objitem = el.loadObjitem("bowl.stl", pos=(850, 100, 780), sample_num=1000)
    objitem.show_objcm()
    movedir = np.asarray((0, 1, 0))
    q = iks_lft.solve_numik4(tgtpos_2, tgtrot, seedjntagls=msc, releemat4=releemat4, toggledebug=True,
                             col_ps=objitem.pcd, movedir=movedir)
    if q is not None:
        eepos, eerot = rbth.get_ee(armjnts=q, releemat4=releemat4)
        rbth.draw_axis(eepos, eerot, rgba=(1, 0, 0, .5))
        mp_lft.rbth.show_armjnts(armjnts=q, rgba=(0, 1, 0, .5), toggleendcoord=False)
        print(np.linalg.norm(q - msc))

    base.run()
