import copy
import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import BendSim
from scipy import interpolate
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bend_utils as bu
import bender_config as bconfig


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, init_rotseq, goal_pseq, bend_times=1):
        self.bs = bs
        self.bend_times = bend_times
        self.goal_pseq = goal_pseq
        self.init_pseq = copy.deepcopy(init_pseq)
        self.init_rotseq = copy.deepcopy(init_rotseq)
        self.bs.reset(self.init_pseq, self.init_rotseq)
        self.total_len = bu.cal_length(goal_pseq)
        self.init_l = bconfig.INIT_L

        # self.result = None
        self.cons = []
        # self.rb = (-math.pi / 3, math.pi / 3)
        self.rb = (-math.pi / 2, math.pi / 2)
        self.lb = (0, self.total_len + self.bs.bend_r * math.pi)
        self.bnds = (self.rb, self.lb) * self.bend_times

    def objctive(self, x):
        self.bs.reset(self.init_pseq, self.init_rotseq)
        try:
            self.bend_x(x)
            _ = self.bs.move_to_org(self.init_l)
            # pseq = bu.linear_inp3d_by_step(bs.pseq)
            # err, fitness, _ = o3dh.registration_ptpt(np.asarray(pseq), np.asarray(goal_pseq), toggledebug=False)
            err, _ = avg_distance_between_polylines(np.asarray(self.bs.pseq[1:-2]), np.asarray(goal_pseq),
                                                    toggledebug=False)
        except:
            err = 1
        print('cost:', err)
        return err

    def bend_x(self, x):
        print('-----------')
        print(x)
        for i in range(int(len(x) / 2)):
            # pos, rot, angle = \
            #     bs.cal_startp(x[2 * i + 1], dir=0 if x[2 * i] < 0 else 1, toggledebug=False)
            # if pos is not None:
            bs.bend(bend_angle=x[2 * i], lift_angle=np.radians(0), bend_pos=x[2 * i + 1])
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
        self.cons.append({'type': 'ineq', 'fun': lambda x: x[2 * i + 3] - x[2 * i + 1]})

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def random_init(self):
        pos = [random.uniform(self.rb[0], self.rb[1]) for _ in range(self.bend_times)]
        pos.sort()
        rot = [random.uniform(self.lb[0], self.lb[1]) for _ in range(self.bend_times)]
        init = np.asarray(list(zip(pos, rot))).flatten()
        init_pseq = self.bend_x(init)
        bu.show_pseq(bu.linear_inp3d_by_step(init_pseq), rgba=(0, 0, 1, 1))

        return np.asarray(init)

    def equal_init(self):
        init = []
        for i in range(self.bend_times):
            init.append(np.radians(360 / self.bend_times))
            # init.append(random.uniform(self.rb[0], self.rb[1]))
            init.append(i * self.total_len / self.bend_times)
        init_pseq = self.bend_x(init)
        bu.show_pseq(bu.linear_inp3d_by_step(init_pseq), rgba=(0, 0, 1, 1))
        # self.bs.show(rgba=(0, 0, 1, .5))

        return np.asarray(init)

    def solve(self, method='SLSQP', init=None):
        """

        :param seedjntagls:
        :param tgtpos:
        :param tgtrot:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.addconstraint(self.con_end, condition="ineq")
        self.addconstraint(self.con_avgdist, condition="ineq")
        if init is None:
            # init = self.random_init()
            init = self.equal_init()
        for i in range(int(len(init) / 2) - 1):
            self.addconstraint_sort(i)
        sol = minimize(self.objctive, init, method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        if sol.success:
            print(sol.x)
            ax = plt.axes()
            ax.grid()
            ax.scatter([v for i, v in enumerate(sol.x) if i % 2 != 0],
                       [v for i, v in enumerate(sol.x) if i % 2 == 0], color='red')
            ax.plot([v for i, v in enumerate(sol.x) if i % 2 != 0],
                    [v for i, v in enumerate(sol.x) if i % 2 == 0], color='red')
            ax.scatter([v for i, v in enumerate(init) if i % 2 != 0],
                       [v for i, v in enumerate(init) if i % 2 == 0], color='blue')
            ax.plot([v for i, v in enumerate(init) if i % 2 != 0],
                    [v for i, v in enumerate(init) if i % 2 == 0], color='blue')
            plt.show()
            return sol.x, sol.fun
        else:
            return None, None


def avg_distance_between_polylines(pts1, pts2, toggledebug=False):
    def __normed_distance_along_path(polyline_x, polyline_y, polyline_z):
        polyline = np.asarray([polyline_x, polyline_y, polyline_z])
        distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))
        return np.insert(distance, 0, 0) / distance[-1]

    x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]

    s1 = __normed_distance_along_path(x1, y1, z1)
    s2 = __normed_distance_along_path(x2, y2, z2)

    interpol_xyz1 = interpolate.interp1d(s1, [x1, y1, z1])
    xyz1_on_2 = interpol_xyz1(s2)

    node_to_node_distance = np.sqrt(np.sum((xyz1_on_2 - [x2, y2, z2]) ** 2, axis=0))

    if toggledebug:
        ax = plt.axes(projection='3d')
        # z_max = max([abs(np.max(z1)), abs(np.max(z2))])
        # ax.set_zlim([-z_max, z_max])
        ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')
        ax.scatter3D(x2, y2, z2, color='green')
        ax.plot3D(x2, y2, z2, 'green')
        ax.scatter3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], color='black')
        ax.plot3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], 'black')
        plt.show()
    err = node_to_node_distance.mean()
    print('Avg. distance between polylines:', err)
    return err, xyz1_on_2


def avg_mindist_between_polylines(pts1, pts2, toggledebug=True):
    pts1 = bu.linear_inp3d_by_step(pts1)
    pts2 = bu.linear_inp3d_by_step(pts2)

    if toggledebug:
        x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
        x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]
        ax = plt.axes(projection='3d')
        ax.set_zlim([-.1, .1])
        ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')
        ax.scatter3D(x2, y2, z2, color='green')
        ax.plot3D(x2, y2, z2, 'green')
        plt.show()


def __ps2seg_max_dist(p1, p2, ps):
    p1_p = np.asarray([p1] * len(ps)) - np.asarray(ps)
    p2_p = np.asarray([p2] * len(ps)) - np.asarray(ps)
    p1_p_norm = np.linalg.norm(p1_p, axis=1)
    p2_p_norm = np.linalg.norm(p2_p, axis=1)
    p2_p1 = np.asarray([p2 - p1] * len(ps))
    dist_list = abs(np.linalg.norm(np.cross(p2_p1, p1_p), axis=1) / np.linalg.norm(p2 - p1))

    l1 = np.arccos(np.sum((p1_p / p1_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1))
    l2 = np.arccos(np.sum((p2_p / p2_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1))
    l1 = (l1[:] < math.pi / 2).astype(int)
    l2 = (l2[:] > math.pi / 2).astype(int)

    dist_list = np.multiply(p1_p_norm, l1) + np.multiply(p2_p_norm, l2) + np.multiply(dist_list, 1 - l1 - l2)
    max_dist = max(dist_list)

    return max_dist, list(dist_list).index(max_dist)


def iter_fit(pseq, tor=.001, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    ptr = 0
    while ptr < len(res_pids) - 1:
        max_err, max_inx = __ps2seg_max_dist(pseq[res_pids[ptr]], pseq[res_pids[ptr + 1]],
                                             pseq[res_pids[ptr]:res_pids[ptr + 1]])
        if max_err > tor:
            res_pids.append(max_inx + res_pids[ptr])
            res_pids = sorted(res_pids)
        else:
            ptr += 1

        if toggledebug:
            ax = plt.axes(projection='3d')
            bu.plot_pseq(ax, pseq)
            bu.plot_pseq(ax, bu.linear_inp3d_by_step(pseq[res_pids]))
            bu.plot_pseq(ax, pseq[res_pids])
            plt.show()
    return pseq[res_pids]


def pseq2bendseq(res_pseq, bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L):
    tangent_pts = []
    bendseq = []
    pos = 0
    diff_list = []
    for i in range(1, len(res_pseq) - 1):
        v1 = res_pseq[i - 1] - res_pseq[i]
        v2 = res_pseq[i] - res_pseq[i + 1]
        bend_a = rm.angle_between_vectors(v1, v2)
        rot_n = np.cross(v1, v2)
        if rot_n[2] > 0:
            bend_a = -bend_a

        pos += np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
        v3 = res_pseq[i - 1] - res_pseq[i + 1]
        n = np.cross(v1, v2)
        # lift_a = rm.angle_between_vectors(v3, [v3[0], v3[1], 0])
        # if v3[2] < 0:
        #     lift_a = -lift_a
        rot_a = rm.angle_between_vectors(np.asarray([0, 0, 1]), n)-np.pi
        lift_a = 0
        l = (bend_r / np.tan((np.pi - abs(bend_a)) / 2)) / np.cos(abs(lift_a))
        arc = abs(bend_a) * bend_r
        bendseq.append([bend_a, lift_a, rot_a, pos + init_l - l - sum(diff_list)])
        diff_list.append(2 * l - arc)

        ratio_1 = l / np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
        p1 = res_pseq[i] + (res_pseq[i - 1] - res_pseq[i]) * ratio_1
        ratio_2 = l / np.linalg.norm(res_pseq[i] - res_pseq[i + 1])
        p2 = res_pseq[i] + (res_pseq[i + 1] - res_pseq[i]) * ratio_2
        tangent_pts.append(p1)
        tangent_pts.append(p2)

    ax = plt.axes(projection='3d')
    bu.plot_pseq(ax, bu.linear_inp3d_by_step(res_pseq))
    bu.plot_pseq(ax, res_pseq)
    bu.plot_pseq(ax, tangent_pts)
    plt.show()

    return bendseq


if __name__ == '__main__':
    import pickle

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)

    # goal_pseq = bu.gen_polygen(5, .05)
    # goal_pseq = bu.gen_ramdom_curve(length=.1, step=.0005, z_max=.01, toggledebug=False)
    # goal_pseq = bu.gen_circle(.05)
    goal_pseq = np.asarray([(0, 0, 0), (0, .02, 0), (.02, .02, 0), (.02, .03, .02)])

    init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    bs = BendSim.BendSim(pseq=init_pseq, rotseq=init_rotseq, show=True)

    fit_pseq = iter_fit(goal_pseq, tor=.0005, toggledebug=False)
    init_bendseq = pseq2bendseq(fit_pseq)
    pickle.dump(init_bendseq, open('./tmp_bendseq.pkl', 'wb'))

    result_flag = bs.gen_by_bendseq(init_bendseq, toggledebug=False)
    print('Result Flag:', result_flag)

    goal_pseq, res_pseq = bu.align_with_goal(bs, goal_pseq)
    err, _ = avg_distance_between_polylines(res_pseq, goal_pseq, toggledebug=True)

    bu.show_pseq(bs.pseq, rgba=(1, 0, 0, 1))
    bu.show_pseq(bu.linear_inp3d_by_step(goal_pseq), rgba=(0, 1, 0, 1))

    # opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=len(init_bendseq))
    # res, cost = opt.solve(init=np.asarray([[v[0], v[2]] for v in init_bendseq]).flatten())
    # print(res, cost)
    # bs.bend(res[0], 0, res[1])
    # bu.show_pseq(bu.linear_inp3d_by_step(res), rgba=(1, 0, 0, 1))

    bs.show()
    base.run()
