import copy
import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import BendSim
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bend_utils as bu
import bender_config as bconfig
import utils.math_utils as mu


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, init_rotseq, goal_pseq, bend_times=1):
        self.bs = bs
        self.bend_times = bend_times
        self.goal_pseq = goal_pseq
        self.init_pseq = copy.deepcopy(init_pseq)
        self.init_rotseq = copy.deepcopy(init_rotseq)
        self.bs.reset(self.init_pseq, self.init_rotseq, extend=False)
        self.total_len = bu.cal_length(goal_pseq)
        self.init_l = bconfig.INIT_L
        self.init_rot = np.eye(3)
        # self.result = None
        self.cons = []
        self.ba_b = (-math.pi / 2, math.pi / 2)
        self.ra_b = (-math.pi / 2, math.pi / 2)
        self.la_b = (-math.pi / 3, math.pi / 3)
        self.l_b = (0, self.total_len + self.bs.bend_r * math.pi)
        self.bnds = (self.ba_b, self.ra_b, self.la_b, self.l_b) * self.bend_times

        self.init_bendset = None

    # def objctive(self, x):
    #     self.bs.reset(self.init_pseq, self.init_rotseq)
    #     try:
    #         self.bend_x(x)
    #         goal_pseq, res_pseq = bu.align_with_goal(bs, self.goal_pseq, self.init_rot)
    #         pseq = bu.linear_inp3d_by_step(res_pseq[:-1])
    #         err, fitness, _ = o3dh.registration_ptpt(np.asarray(pseq), np.asarray(goal_pseq), toggledebug=False)
    #         print(err, fitness)
    #     except:
    #         err = 1
    #         fitness = .1
    #     print('cost:', err / fitness)
    #     return err

    def objctive(self, x):
        self.bs.reset(self.init_pseq, self.init_rotseq, extend=False)
        try:
            self.bend_x(x)
            goal_pseq, res_pseq = bu.align_with_goal(bs, self.goal_pseq, self.init_rot)
            # err, _ = bu.avg_polylines_dist_err(np.asarray(res_pseq), np.asarray(goal_pseq), toggledebug=False)
            err, _ = bu.mindist_err(np.asarray(res_pseq), np.asarray(goal_pseq), toggledebug=False)
        except:
            err = 1
        print('cost:', err * 100)
        return err * 100

    def bend_x(self, x):
        x = np.asarray(x)
        self.bs.gen_by_bendseq(x.reshape(self.bend_times, 4), cc=False)
        return self.bs.pseq

    def update_known(self):
        return NotImplemented

    def con_end(self, x):
        last_end = 0
        for i in range(int(len(x) / 2)):
            tmp_end = x[2 * i + 1] + x[2 * i] * (self.bs.r_center + self.bs.thickness)
            if tmp_end > last_end:
                last_end = tmp_end
        # print('------constrain------')
        # print('last end', last_end)
        return self.total_len - last_end

    def con_avgdist(self, x):
        pos_list = []
        for i in range(int(len(x) / 2)):
            pos_list.append(x[2 * i + 1])
        sorted_poslist = sorted(pos_list)
        min_dist = min(abs(prev - cur) for cur, prev in zip(sorted_poslist, sorted_poslist[1:]))
        # print('------constrain------')
        # print(pos_list)
        # print(sorted_poslist)
        # print(min_dist - self.bs.bend_r)
        return min_dist - self.bs.bend_r

    def addconstraint_sort(self, i):
        self.cons.append({'type': 'ineq', 'fun': lambda x: x[4 * i + 5] - x[4 * i + 1]})

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def random_init(self):
        pos = [random.uniform(self.l_b[0], self.l_b[1]) for _ in range(self.bend_times)]
        pos.sort()
        ba = [random.uniform(self.ba_b[0], self.ba_b[1]) for _ in range(self.bend_times)]
        la = [random.uniform(self.la_b[0], self.la_b[1]) for _ in range(self.bend_times)]
        ra = [random.uniform(self.ra_b[0], self.ra_b[1]) for _ in range(self.bend_times)]
        init = np.asarray(list(zip(ba, la, ra, pos))).flatten()
        init_pseq = self.bend_x(init)
        bu.show_pseq(bu.linear_inp3d_by_step(init_pseq), rgba=(0, 0, 1, 1))

        return np.asarray(init)

    def equal_init(self):
        init = []
        for i in range(self.bend_times):
            init.append(i * self.total_len / self.bend_times)
            init.append(random.uniform(self.la_b[0], self.la_b[1]))
            init.append(random.uniform(self.ra_b[0], self.ra_b[1]))
            init.append(np.radians(360 / self.bend_times))
        init_pseq = self.bend_x(init)
        bu.show_pseq(bu.linear_inp3d_by_step(init_pseq), rgba=(0, 0, 1, 1))
        # self.bs.show(rgba=(0, 0, 1, .5))

        return np.asarray(init)

    def fit_init(self, goal_pseq, tor=.001):
        fit_pseq = bu.decimate_pseq(goal_pseq, r=bconfig.R_CENTER, tor=tor, toggledebug=False)
        self.init_bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)
        self.init_rot = bu.get_init_rot(fit_pseq)
        self.bend_times = len(self.init_bendset)
        return np.asarray(self.init_bendset).flatten()

    def update_bnds(self, bseq_flatten):
        bseq = bseq_flatten.reshape(self.bend_times, 4)
        self.bnds = []
        for b in bseq:
            self.bnds.append((b[0] - math.pi / 20, b[0] + math.pi / 20))
            self.bnds.append((b[1] - math.pi / 20, b[1] + math.pi / 20))
            self.bnds.append((b[2] - math.pi / 20, b[2] + math.pi / 20))
            self.bnds.append((b[3] - .02, b[3] + .02))

    def solve(self, method='SLSQP', init=None):
        """

        :param seedjntagls:
        :param tgtpos:
        :param tgtrot:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        # self.addconstraint(self.con_end, condition="ineq")
        # self.addconstraint(self.con_avgdist, condition="ineq")
        if init is None:
            # init = self.random_init()
            # init = self.equal_init()
            init = self.fit_init(goal_pseq, tor=.0002)
        self.update_bnds(init)
        for i in range(int(len(init) / 4) - 1):
            self.addconstraint_sort(i)
        sol = minimize(self.objctive, init, method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        if sol.success:
            self.bs.reset(self.init_pseq, self.init_rotseq, extend=False)
            print(sol.x)
            # plt.figure(figsize=(16, 5))
            plt.grid()
            plt.subplot(131)
            plt.scatter([v for i, v in enumerate(sol.x) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 0], color='red')
            plt.plot([v for i, v in enumerate(sol.x) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 0], color='red')
            plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='blue')
            plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='blue')

            plt.subplot(132)
            plt.scatter([v for i, v in enumerate(sol.x) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 1], color='red')
            plt.plot([v for i, v in enumerate(sol.x) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 1], color='red')
            plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='blue')
            plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='blue')

            plt.subplot(133)
            plt.scatter([v for i, v in enumerate(sol.x) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 2], color='red')
            plt.plot([v for i, v in enumerate(sol.x) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(sol.x) if i % 4 == 2], color='red')
            plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                        [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='blue')
            plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                     [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='blue')
            plt.show()
            return sol.x.reshape(self.bend_times, 4), sol.fun
        else:
            return None, None


if __name__ == '__main__':
    import pickle
    import bendplanner.BendSim as b_sim

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=False, granularity=np.pi / 30)

    # goal_pseq = bu.gen_polygen(5, .05)
    goal_pseq = pickle.load(open('../run_plan/goal_pseq.pkl', 'rb'))

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=1)
    res_bendseq, cost = opt.solve()

    bs.gen_by_bendseq(res_bendseq, cc=False)
    goal_pseq, res_pseq_opt = bu.align_with_goal(bs, goal_pseq, opt.init_rot)
    # _, _, _ = o3dh.registration_ptpt(np.asarray(bu.linear_inp3d_by_step(res_pseq[:-1])), np.asarray(goal_pseq),
    #                                  toggledebug=True)
    err, _ = bu.avg_polylines_dist_err(res_pseq_opt, goal_pseq, toggledebug=True)

    bs.reset(init_pseq, init_rotseq, extend=False)
    bs.gen_by_bendseq(opt.init_bendset, cc=False)
    _, res_pseq = bu.align_with_goal(bs, goal_pseq, opt.init_rot)
    err, _ = bu.avg_polylines_dist_err(res_pseq_opt, goal_pseq, toggledebug=True)

    bu.show_pseq(bu.linear_inp3d_by_step(bs.pseq), rgba=(1, 0, 0, 1))
    bu.show_pseq(bu.linear_inp3d_by_step(goal_pseq), rgba=(1, 1, 0, 1))
    base.run()
