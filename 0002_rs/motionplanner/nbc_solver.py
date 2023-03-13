import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import motionplanner.robot_helper as rbt_helper
import utils.pcd_utils as pcdu

TB = True


class NBCOptimizerVec(object):
    def __init__(self, rbt, max_a=np.pi / 6, max_dist=1, env=None, armname="arm", releemat4=np.eye(4), toggledebug=TB):
        self.rbt = rbt
        self.armname = armname
        self.env = env
        self.armname = armname
        self.releemat4 = releemat4
        self.toggledebug = toggledebug
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)

        self.result = None
        self.cons = []
        b = (-np.pi, np.pi)
        self.bnds = (b, b, b, b, b, b, b)

        self.seedjntagls = None
        self.tgtpos = None
        self.tgtrot = None

        self.jnts = []
        self.rot_err = []
        self.pos_err = []
        self.jd_list = []  # joints displacement
        self.mp_list = []  # manipulability
        self.sr_list = []  # angle between line of sight
        self.wo_list = []  # wrist obstruction
        self.obj_list = []
        self.ref_list = []  # reflection

        self.max_a = max_a
        self.max_dist = max_dist

        self.pt_nbv, self.nrml_nbv = [], []
        self.init_eepos, self.init_eerot, self.init_eemat4 = None, None, None
        self.laser_pos, self.cam_pos, self.cam_mat4 = (0, 0, 0), (0, 0, 0), np.eye(3)

    def objctive(self, x):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        w_e = np.linalg.norm(x - self.seedjntagls)
        w_m = self.rbt.manipulability(component_name='arm')

        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        p_new = pcdu.trans_pcd(self.pt_nbv, transmat4)[0]
        w_sr = np.degrees(rm.angle_between_vectors(config.CAM_LS, self.cam_pos - p_new))
        w_wo = np.degrees(rm.angle_between_vectors(eerot[:, 0], self.cam_pos - p_new))

        # obj = -(w_e * w_m * (-w_sr) * w_wo)
        # obj = w_e + (-w_m) + w_sr
        obj = w_e

        self.jd_list.append(w_e)
        self.mp_list.append(w_m)
        self.sr_list.append(w_sr)
        self.wo_list.append(w_wo)
        self.obj_list.append(obj)

        return obj

    def update_known(self, seedjntagls, pts_nbv, nrmls_nbv, cam_mat4):
        self.seedjntagls = seedjntagls
        self.init_eepos, self.init_eerot = self.rbt.get_gl_tcp()
        self.init_eemat4 = rm.homomat_from_posrot(self.init_eepos, self.init_eerot)
        self.pt_nbv, self.nrml_nbv, self.cam_pos, self.cam_mat4 = pts_nbv[0], nrmls_nbv[0], cam_mat4[:3, 3], cam_mat4
        self.pts_nbv, self.nrmls_nbv = pts_nbv, nrmls_nbv
        self.laser_pos = self.cam_mat4[:3, 3] - .175 * rm.unit_vector(self.cam_mat4[:3, 0]) \
                         + .049 * rm.unit_vector(self.cam_mat4[:3, 1]) \
                         + .01 * rm.unit_vector(self.cam_mat4[:3, 2])
        gm.gen_frame(self.laser_pos, self.cam_mat4[:3, :3]).attach_to(base)

    def con_rot(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        n_new = transmat4[:3, :3].dot(self.nrml_nbv)
        p_new = pcdu.trans_pcd([self.pt_nbv], transmat4)[0]
        err = rm.angle_between_vectors(n_new, self.cam_pos - p_new)
        err = min([err, np.pi - err])
        self.rot_err.append(np.degrees(err))
        return self.max_a - err

    def con_dist(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.linalg.norm(np.asarray(eepos) - self.cam_pos)
        self.pos_err.append(err)
        return self.max_dist - err

    def con_diff_x(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[0] - self.init_eepos[0])
        return .1 - err

    def con_diff_y(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[1] - self.init_eepos[1])
        return .15 - err

    def con_diff_z(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[2] - self.init_eepos[2])
        return .1 - err

    def con_reflection(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        # n_new = transmat4[:3, :3].dot(self.nrml_nbv)
        # err = rm.angle_between_vectors(n_new, self.cam_mat4[:3, 1])
        # err = min([err, np.pi - err])
        pts_new = pcdu.trans_pcd(self.pts_nbv, transmat4)
        nrmls_new = np.asarray([transmat4[:3, :3].dot(n) for n in self.nrmls_nbv])
        err = min([min(rm.angle_between_vectors(p - self.laser_pos, nrmls_new[i]),
                          np.pi - rm.angle_between_vectors(p - self.laser_pos, nrmls_new[i]))
                      for i, p in enumerate(pts_new)])
        self.ref_list.append(np.degrees(err))
        return err - np.pi / 9

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, seedjntagls, pts_nbv, nrmls_nbv, cam_mat4, method='SLSQP'):
        """

        :param seedjntagls:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(seedjntagls, pts_nbv, nrmls_nbv, cam_mat4)
        self.addconstraint(self.con_rot, condition="ineq")
        self.addconstraint(self.con_dist, condition="ineq")
        self.addconstraint(self.con_diff_x, condition="ineq")
        self.addconstraint(self.con_diff_y, condition="ineq")
        self.addconstraint(self.con_diff_z, condition="ineq")
        self.addconstraint(self.con_reflection, condition="ineq")

        sol = minimize(self.objctive, seedjntagls, method=method, bounds=self.bnds, constraints=self.cons)
        time_cost = time.time() - time_start
        print("Planning nbv time cost:", time_cost, sol.success)

        if self.toggledebug:
            # print(sol)
            self.__debug()

        if sol.success:
            self.rbth.goto_armjnts(sol.x)
            eepos, eerot = self.rbt.get_gl_tcp()
            eemat4 = rm.homomat_from_posrot(eepos, eerot)
            transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
            gm.gen_frame(eepos, eerot).attach_to(base)
            gm.gen_frame(self.init_eepos, self.init_eerot).attach_to(base)
            return sol.x, transmat4, sol.fun, time_cost
        else:
            return None, None, None, time_cost

    def __debug(self):
        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(321)
        self.rbth.plot_vlist(ax1, self.obj_list, title="Objective (Joints Displacement)", show=False)
        ax2 = plt.subplot(322)
        self.rbth.plot_vlist(ax2, self.rot_err, title="Rotation Constain", show=False)
        ax3 = plt.subplot(323)
        self.rbth.plot_vlist(ax3, self.ref_list, title="Reflection", show=False)
        ax4 = plt.subplot(324)
        self.rbth.plot_vlist(ax4, self.mp_list, title="Manipulability", show=False)
        ax5 = plt.subplot(325)
        self.rbth.plot_vlist(ax5, self.sr_list, title="Line of Sight", show=False)
        ax6 = plt.subplot(326)
        self.rbth.plot_vlist(ax6, self.wo_list, title="Wrist Obstruction", show=False)
        # ax6 = plt.subplot(326)
        # self.rbth.plot_armjnts(ax6, self.jnts, show=False)
        plt.show()
