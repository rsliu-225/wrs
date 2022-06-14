import optuna

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
from direct.stdpy import threading
import config


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

        self.ba_b = (-math.pi / 9, math.pi / 9)
        self.la_b = (-math.pi / 1e8, math.pi / 1e8)
        self.ra_b = (-math.pi / 9, math.pi / 9)
        self.l_b = (-.02, .02)

        self.init_bendset = None

        self.cost_list = []

    def bend_x(self, x):
        x = np.asarray(x)
        self.bs.gen_by_bendseq(x.reshape(self.bend_times, 4), cc=False)
        return self.bs.pseq

    def fit_init(self, goal_pseq, goal_rotseq, tor=.001):
        if goal_rotseq is not None:
            fit_pseq, fit_rotseq = bu.decimate_rotpseq(goal_pseq, goal_rotseq, tor=tor, toggledebug=False)
            self.init_bendset = bu.rotpseq2bendset(fit_pseq, fit_rotseq, toggledebug=False)
        else:
            fit_pseq, fit_rotseq = bu.decimate_pseq(goal_pseq, tor=tor, toggledebug=False)
            self.init_bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)
        self.init_rot = bu.get_init_rot(fit_pseq)
        self.bend_times = len(self.init_bendset)
        return np.asarray(self.init_bendset).flatten()

    def objective(self, trial: optuna.Trial):
        x = []
        for i, b in enumerate(self.init_bendset):
            theta = trial.suggest_uniform(f"theta{str(i)}", b[0] + self.ba_b[0], b[0] + self.ba_b[1])
            alpha = trial.suggest_uniform(f"beta{str(i)}", b[1] + self.la_b[0], b[1] + self.la_b[1])
            beta = trial.suggest_uniform(f"alpha{str(i)}", b[2] + self.ra_b[0], b[2] + self.ra_b[1])
            l = trial.suggest_uniform(f"l{str(i)}", b[3] + self.la_b[0], b[3] + self.la_b[1])
            x.extend([theta, alpha, beta, l])

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
        self.cost_list.append(err)

        return err

    def solve(self, n_trials=250):
        self.fit_init(self.goal_pseq, self.goal_rotseq, tor=.0002)
        x0 = {}
        for i, b in enumerate(self.init_bendset):
            x0[f"theta{str(i)}"] = b[0]
            x0[f"beta{str(i)}"] = b[1]
            x0[f"alpha{str(i)}"] = b[2]
            x0[f"l{str(i)}"] = b[3]

        time_start = time.time()
        sampler = optuna.samplers.CmaEsSampler(x0=x0, n_startup_trials=1, sigma0=.01)
        study = optuna.create_study(sampler=sampler)
        study.optimize(self.objective, n_trials=n_trials)
        print("time cost", time.time() - time_start)
        print(study.best_value)
        print(len(list(study.best_params.values())))
        res = np.asarray(list(study.best_params.values()))
        init = np.asarray(self.init_bendset).flatten()

        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.show()

        ax = plt.axes()
        ax.set_title("Error")
        ax.plot([i for i in range(len(self.cost_list[1:]))], self.cost_list[1:], label=["Err"])
        # ax.savefig(f"{config.ROOT}/bendplanner/tst.png")
        plt.show()

        plt.grid()
        plt.subplot(131)
        plt.scatter([v for i, v in enumerate(res) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(res) if i % 4 == 0], color='red')
        plt.plot([v for i, v in enumerate(res) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(res) if i % 4 == 0], color='red')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='blue')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 0], color='blue')

        plt.subplot(132)
        plt.scatter([v for i, v in enumerate(res) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(res) if i % 4 == 1], color='red')
        plt.plot([v for i, v in enumerate(res) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(res) if i % 4 == 1], color='red')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='blue')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 1], color='blue')

        plt.subplot(133)
        plt.scatter([v for i, v in enumerate(res) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(res) if i % 4 == 2], color='red')
        plt.plot([v for i, v in enumerate(res) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(res) if i % 4 == 2], color='red')
        plt.scatter([v for i, v in enumerate(init) if i % 4 == 3],
                    [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='blue')
        plt.plot([v for i, v in enumerate(init) if i % 4 == 3],
                 [np.degrees(v) for i, v in enumerate(init) if i % 4 == 2], color='blue')
        plt.show()

        return res.reshape(self.bend_times, 4), study.best_value


if __name__ == '__main__':
    import pickle
    import bendplanner.BendSim as b_sim

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')

    goal_pseq = pickle.load(open('../data/bend/pseq/random_curve.pkl', 'rb'))
    goal_rotseq = None
    # goal_pseq, goal_rotseq = pickle.load(open('../data/bend/rotpseq/skull2.pkl', 'rb'))

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, goal_rotseq=goal_rotseq, bend_times=1, obj_type='max')
    res_bendseq, cost = opt.solve(n_trials=100)
    for i in range(len(opt.init_bendset)):
        b1 = opt.init_bendset[i]
        b2 = res_bendseq[i]
        print(np.degrees(b1[0]), np.degrees(b2[0]))
        print(np.degrees(b1[1]), np.degrees(b2[1]))
        print(np.degrees(b1[2]), np.degrees(b2[2]))
        print(b1[3], b2[3])
        print('------------')

    bs.gen_by_bendseq(res_bendseq, cc=False)
    goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, opt.init_rot, goal_rotseq)
    bs.show(rgba=(0, 1, 0, 1))
    # goal_cm = bu.gen_surface(goal_pseq, goal_rotseq, bconfig.THICKNESS / 2, width=bconfig.WIDTH)
    # goal_cm.attach_to(base)
    # _, _, _ = o3dh.registration_ptpt(np.asarray(bu.linear_inp3d_by_step(res_pseq[:-1])), np.asarray(goal_pseq),
    #                                  toggledebug=True)
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
