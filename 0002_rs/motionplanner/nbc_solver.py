import math
import warnings as wns
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import config
import motionplanner.robot_helper as rbt_helper
import modeling.geometric_model as gm
import basis.robot_math as rm
import copy
import utils.pcd_utils as pcdu
import localenv.envloader as el

TB = True


class NBCOptimizer(object):
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

        self.max_a = max_a
        self.max_dist = max_dist

        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = [], [], []
        self.init_eepos, self.init_eerot, self.init_eemat4 = None, None, None

    def objctive(self, x):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        w_e = np.linalg.norm(x - self.seedjntagls)
        w_m = self.rbt.manipulability(component_name='arm')

        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        p_new = pcdu.trans_pcd([self.nbv_pts[0]], transmat4)[0]
        w_sr = rm.angle_between_vectors(config.CAM_LS, self.campos - p_new)
        w_wo = rm.angle_between_vectors(eerot[:, 0], self.campos - p_new)

        # obj = -(w_e * w_m * (-w_sr) * w_wo)
        # obj = w_e + (-w_m) + w_sr
        obj = w_e
        self.jd_list.append(w_e)
        self.mp_list.append(w_m)
        self.sr_list.append(w_sr)
        self.wo_list.append(w_wo)
        self.obj_list.append(obj)

        return obj

    def update_known(self, seedjntagls, nbv_pts, nbv_nrmls, campos, ):
        self.seedjntagls = seedjntagls
        self.init_eepos, self.init_eerot = self.rbt.get_gl_tcp()
        self.init_eemat4 = rm.homomat_from_posrot(self.init_eepos, self.init_eerot)
        self.nbv_pts, self.nbv_nrmls, self.campos = nbv_pts, nbv_nrmls, campos,

    def con_rot(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        n_new = pcdu.trans_pcd([self.nbv_nrmls[0]], transmat4)[0]
        p_new = pcdu.trans_pcd([self.nbv_pts[0]], transmat4)[0]
        err = rm.angle_between_vectors(n_new, self.campos - p_new)
        self.rot_err.append(err)
        return self.max_a - err

    def con_dist(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.linalg.norm(np.asarray(eepos) - self.campos)
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
        return .1 - err

    def con_diff_z(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[2] - self.init_eepos[2])
        return .1 - err

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, seedjntagls, nbv_pts, nbv_nrmls, campos, method='SLSQP'):
        """

        :param seedjntagls:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(seedjntagls, nbv_pts, nbv_nrmls, campos)
        self.addconstraint(self.con_rot, condition="ineq")
        # self.addconstraint(self.con_dist, condition="ineq")
        self.addconstraint(self.con_diff_x, condition="ineq")
        self.addconstraint(self.con_diff_y, condition="ineq")
        self.addconstraint(self.con_diff_z, condition="ineq")

        # iks = IkSolver(self.env, self.rbt, self.rbtmg, self.rbtball, self.armname)
        # q0 = iks.solve_numik3(self.tgtpos, tgtrot=None, seedjntagls=self.seedjntagls, releemat4=self.releemat4)
        sol = minimize(self.objctive, seedjntagls, method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        self.rbth.goto_armjnts(sol.x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        gm.gen_frame(eepos, eerot).attach_to(base)
        gm.gen_frame(self.init_eepos, self.init_eerot).attach_to(base)

        if self.toggledebug:
            # print(sol)
            self.__debug()

        if sol.success:
            return sol.x, transmat4, sol.fun
        else:
            return None, None, None

    def __debug(self):
        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(321)
        self.rbth.plot_vlist(ax1, self.obj_list, title="objective", show=False)
        ax2 = plt.subplot(322)
        self.rbth.plot_vlist(ax2, self.rot_err, title="normal to cam angle", show=False)
        ax3 = plt.subplot(323)
        self.rbth.plot_vlist(ax3, self.jd_list, title="joints displacement", show=False)
        ax4 = plt.subplot(324)
        self.rbth.plot_vlist(ax4, self.mp_list, title="manipulability", show=False)
        ax5 = plt.subplot(325)
        self.rbth.plot_vlist(ax5, self.sr_list, title="line of sight", show=False)
        ax6 = plt.subplot(326)
        self.rbth.plot_vlist(ax6, self.wo_list, title="wrist obstruction", show=False)
        # ax6 = plt.subplot(326)
        # self.rbth.plot_armjnts(ax6, self.jnts, show=False)
        plt.show()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    rbt = el.loadXarm(showrbt=True)
    nbs_opt = NBCOptimizer(rbt)

    base.run()
