import copy
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from direct.stdpy import threading
from scipy.optimize import minimize

import basis.o3dhelper as o3dh
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import modeling.geometric_model as gm
import visualization.panda.world as wd


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, init_rotseq, goal_pseq, goal_rotseq, bend_times=1, obj_type='max'):
        self.bs = bs
        self.bend_times = bend_times
        self.goal_pseq = goal_pseq
        self.goal_rotseq = goal_rotseq
        self.init_pseq = copy.deepcopy(init_pseq)
        self.init_rotseq = copy.deepcopy(init_rotseq)
        self.obj_type = obj_type

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

        self.ba_relb = (-math.pi / 9, math.pi / 9)
        # self.la_relb = (-math.pi / 1e8, math.pi / 1e8)
        self.la_relb = None
        self.ra_relb = (-math.pi / 9, math.pi / 9)
        self.l_relb = (-.02, .02)
        self.bnds = ([v for v in [self.ba_b, self.ra_b, self.la_b, self.l_b] if v is not None]) * self.bend_times

        self.init_bendset = None
        self.cost_list = []

        self._ploton = True
        self._plotflag = True
        self._thread_plot = threading.Thread(target=self._plot, name="plot")

    def objective_icp(self, x):
        self.bs.reset(self.init_pseq, self.init_rotseq)
        try:
            self.bend_x(x)
            goal_pseq, res_pseq = bu.align_with_init(bs, self.goal_pseq, self.init_rot)
            pseq = bu.linear_inp3d_by_step(res_pseq[:-1])
            err, fitness, _ = o3dh.registration_ptpt(np.asarray(pseq), np.asarray(goal_pseq), toggledebug=False)
            print(err, fitness)
        except:
            err = 1
            fitness = .1
        if len(self.cost_list) % 10 == 0:
            print('cost:', err / fitness)
        return err

    def objective(self, x):
        self.bs.reset(self.init_pseq, self.init_rotseq, extend=False)
        try:
            self.bend_x(x)
            goal_pseq, goal_rotseq = bu.align_with_init(self.bs, self.goal_pseq, self.init_rot, self.goal_rotseq)
            # err, _ = bu.avg_polylines_dist_err(np.asarray(res_pseq), np.asarray(goal_pseq), toggledebug=False)
            if goal_rotseq is None:
                err, _ = bu.mindist_err(self.bs.pseq[1:], goal_pseq, type=self.obj_type, toggledebug=False)
            else:
                err, _ = bu.mindist_err(self.bs.pseq[1:], goal_pseq, self.bs.rotseq[1:], goal_rotseq,
                                        type=self.obj_type, toggledebug=False)
        except:
            err = 100
        if len(self.cost_list) % 10 == 0:
            print('cost:', err)
        self.cost_list.append(err)

        return err

    def _zfill_x(self, x):
        relbnds = [self.ba_relb, self.la_relb, self.ra_relb, self.l_relb]
        x = np.asarray(x).reshape(self.bend_times, 4 - relbnds.count(None))
        for i in range(4):
            if relbnds[i] is None:
                x = np.insert(x, i, 0, axis=1)
        return x.flatten()

    def bend_x(self, x):
        x = self._zfill_x(x)
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

    def fit_init(self, goal_pseq, goal_rotseq, tor=None, cnt=None):
        if goal_rotseq is not None:
            fit_pseq, fit_rotseq = bu.decimate_rotpseq(goal_pseq, goal_rotseq, tor=tor, toggledebug=False)
            self.init_bendset = bu.rotpseq2bendset(fit_pseq, fit_rotseq, toggledebug=False)
        else:
            if tor is not None:
                fit_pseq, fit_rotseq = bu.decimate_pseq(goal_pseq, tor=tor, toggledebug=False)
            else:
                fit_pseq, fit_rotseq = bu.decimate_pseq_by_cnt(goal_pseq, cnt=cnt, toggledebug=False)
            self.init_bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)
        self.init_rot = bu.get_init_rot(fit_pseq)
        self.bend_times = len(self.init_bendset)
        return np.asarray(self.init_bendset).flatten()

    def update_bnds(self, bseq_flatten):
        bseq = bseq_flatten.reshape(self.bend_times, 4)
        relbnds = [self.ba_relb, self.la_relb, self.ra_relb, self.l_relb]
        non_inx = [i for i, v in enumerate(relbnds) if v is None]
        self.bnds = []
        for b in bseq:
            for i in range(4):
                if relbnds[i] is not None:
                    self.bnds.append((b[i] + relbnds[i][0], b[i] + relbnds[i][1]))
        if len(non_inx) > 0:
            bseq = np.delete(bseq, non_inx, 1)
        return bseq.flatten()

    def solve(self, method='SLSQP', init=None, tor=None, cnt=None, toggledebeg=False):
        """

        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """

        time_start = time.time()
        # self.addconstraint(self.con_end, condition="ineq")
        # self.addconstraint(self.con_avgdist, condition="ineq")
        if init is None:
            # init = self.random_init()
            # init = self.equal_init()
            init = self.fit_init(self.goal_pseq, self.goal_rotseq, tor=tor, cnt=cnt)
        init = self.update_bnds(init)
        # var_num = len(init) / self.bend_times
        # for i in range(int(len(init) / var_num) - 1):
        #     self.addconstraint_sort(i)

        # self._thread_plot.start()
        sol = minimize(self.objective, init, method=method, bounds=self.bnds, constraints=self.cons,
                       options={'maxiter': 100, 'ftol': 1e-04, 'iprint': 1, 'disp': True,
                                'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None})

        time_cost = time.time() - time_start
        print("time cost", time_cost, sol.success)
        # self._thread_plot.join()
        self._ploton = False

        # ax = plt.axes()
        # ax.set_title("Error")
        # ax.plot([i for i in range(len(self.cost_list))], self.cost_list, label=["Err"])
        # # ax.savefig(f"{config.ROOT}/bendplanner/tst.png")
        # plt.show()

        if sol.success:
            sol_x = self._zfill_x(sol.x)
            init = self._zfill_x(init)
            # self._plot_param(sol_x, init)
            self.bs.reset(self.init_pseq, self.init_rotseq, extend=False)
            return sol_x.reshape(self.bend_times, 4), sol.fun, time_cost
        else:
            sol_x = self._zfill_x(sol.x)
            print(sol_x)
            return None, None, time_cost

    def _plot(self):
        print("plot start")
        fig = plt.figure(1, figsize=(16, 9))
        plt.ion()
        plt.show()
        plt.title("Error")
        while self._ploton:
            if 1:
                plt.clf()
                x = [i for i in range(len(self.cost_list))]
                plt.plot(x, self.cost_list, label=["Err"])
                # plt.pause(.5)
            time.sleep(.5)
        plt.savefig(f"{config.ROOT}/bendplanner/tst.png")
        plt.close(fig)

    def _plot_param(self, sol, init):
        plt.grid()
        plt.subplot(131)
        plt.scatter([v for i, v in enumerate(sol) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 0], color='r')
        plt.plot([v for i, v in enumerate(sol) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 0], color='r')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='g')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='g')

        plt.subplot(132)
        plt.scatter([v for i, v in enumerate(sol) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 1], color='r')
        plt.plot([v for i, v in enumerate(sol) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 1], color='r')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='g')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='g')

        plt.subplot(133)
        plt.scatter([v for i, v in enumerate(sol) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 2], color='r')
        plt.plot([v for i, v in enumerate(sol) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(sol) if i % 4 == 2], color='r')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='g')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='g')
        plt.show()


if __name__ == '__main__':
    import pickle

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')
    f_name = 'random_curve'
    goal_pseq = pickle.load(open(f'goal/pseq/{f_name}.pkl', 'rb'))
    goal_rotseq = None
    # goal_pseq, goal_rotseq = pickle.load(open('../data/goal/rotpseq/skull2.pkl', 'rb'))

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    '''
    fit init param
    '''
    method = 'SLSQP'
    tor = .0002
    cnt = None
    obj_type = 'sum'

    '''
    opt
    '''
    opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, goal_rotseq=goal_rotseq, bend_times=1, obj_type=obj_type)
    res_bendseq, cost, time_cost = opt.solve(method=method, tor=tor, cnt=cnt)

    bs.gen_by_bendseq(res_bendseq, cc=False)
    goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, opt.init_rot, goal_rotseq)
    bs.show(rgba=(0, 1, 0, 1))
    # goal_cm = bu.gen_surface(goal_pseq, goal_rotseq, bconfig.THICKNESS / 2, width=bconfig.WIDTH)
    # goal_cm.attach_to(base)

    res_pseq_opt = bs.pseq[1:]
    err, _ = bu.mindist_err(res_pseq_opt, goal_pseq, toggledebug=True)
    print(err)

    bs.reset(init_pseq, init_rotseq, extend=False)
    bs.gen_by_bendseq(opt.init_bendset, cc=False)
    _, _ = bu.align_with_init(bs, goal_pseq, opt.init_rot, goal_rotseq)
    bs.show(rgba=(1, 0, 0, 1))
    res_pseq = bs.pseq[1:]
    err, _ = bu.mindist_err(res_pseq, goal_pseq, toggledebug=True)
    print(err)

    ax = plt.axes(projection='3d')
    center = np.mean(res_pseq, axis=0)
    ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
    ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
    ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
    bu.plot_pseq(ax, res_pseq, c='r')
    bu.plot_pseq(ax, res_pseq_opt, c='g')
    bu.plot_pseq(ax, goal_pseq, c='black')
    plt.show()

    bu.show_pseq(bu.linear_inp3d_by_step(res_pseq), rgba=(1, 0, 0, 1))
    bu.show_pseq(bu.linear_inp3d_by_step(goal_pseq), rgba=(1, 1, 0, 1))
    bu.show_pseq(bu.linear_inp3d_by_step(res_pseq_opt), rgba=(0, 1, 0, 1))
    base.run()
