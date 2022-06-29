import math
import warnings as wns
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

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

        self.max_a = max_a
        self.max_dist = max_dist

    def objctive(self, x):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        we = np.linalg.norm(x - self.seedjntagls)
        wm = self.rbt.manipulability(component_name='arm')
        return we * wm

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
        self.addconstraint(self.con_dist, condition="ineq")
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
            print(sol)
            self.__debug()
            # base.run()

        if sol.success:
            return sol.x, transmat4, sol.fun
        else:
            return None, None, None

    def __debug(self):
        plt.subplot(221)
        self.rbth.plot_vlist(self.pos_err, title="cam dist")
        plt.subplot(222)
        self.rbth.plot_vlist(self.rot_err, title="normal/cam angle")
        plt.subplot(223)
        self.rbth.plot_armjnts(self.jnts)
        plt.show()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    rbt = el.loadXarm(showrbt=True)
    nbs_opt = NBCOptimizer(rbt)

    base.run()
