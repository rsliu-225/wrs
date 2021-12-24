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


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, init_rotseq, goal_pseq, bend_times=1):
        self.bs = bs
        self.bend_times = bend_times
        self.goal_pseq = goal_pseq
        self.init_pseq = init_pseq
        self.init_rotseq = init_rotseq
        self.bs.reset(self.init_pseq, self.init_rotseq)
        self.total_len = bu.cal_length(goal_pseq)

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
            pseq = bu.linear_inp3d(bs.pseq)
            err, fitness, _ = o3dh.registration_ptpt(np.asarray(pseq), np.asarray(goal_pseq), toggledebug=False)
            # err, _ = average_distance_between_polylines(np.asarray(pseq), np.asarray(goal_pseq), toggledebug=True)
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
            bs.bend(rot_angle=x[2 * i], lift_angle=np.radians(0), insert_l=x[2 * i + 1])
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
        bu.show_pseq(bu.linear_inp3d(init_pseq), rgba=(0, 0, 1, 1))

        return np.asarray(init)

    def equal_init(self):
        init = []
        for i in range(self.bend_times):
            init.append(np.radians(360 / self.bend_times))
            # init.append(random.uniform(self.rb[0], self.rb[1]))
            init.append(i * self.total_len / self.bend_times)
        init_pseq = self.bend_x(init)
        bu.show_pseq(bu.linear_inp3d(init_pseq), rgba=(0, 0, 1, 1))
        # self.bs.show(rgba=(0, 0, 1, .5))

        return np.asarray(init)

    def solve(self, method='SLSQP', init_bendseq=None):
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
        if init_bendseq is None:
            # init_bendseq = self.random_init()
            init_bendseq = self.equal_init()
        print(init_bendseq)
        for i in range(int(len(init_bendseq) / 2) - 1):
            self.addconstraint_sort(i)
        sol = minimize(self.objctive, init_bendseq, method=method, bounds=self.bnds, constraints=self.cons,
                       options={'maxiter': 10000, 'disp': True})
        print("time cost", time.time() - time_start, sol.success)

        if sol.success:
            print(sol.x)
            ax = plt.axes()
            ax.grid()
            ax.scatter([v for i, v in enumerate(sol.x) if i % 2 != 0],
                       [v for i, v in enumerate(sol.x) if i % 2 == 0], color='red')
            ax.plot([v for i, v in enumerate(sol.x) if i % 2 != 0],
                    [v for i, v in enumerate(sol.x) if i % 2 == 0], color='red')
            ax.scatter([v for i, v in enumerate(init_bendseq) if i % 2 != 0],
                       [v for i, v in enumerate(init_bendseq) if i % 2 == 0], color='blue')
            ax.plot([v for i, v in enumerate(init_bendseq) if i % 2 != 0],
                    [v for i, v in enumerate(init_bendseq) if i % 2 == 0], color='blue')
            plt.show()
            return sol.x, sol.fun
        else:
            return None, None


def average_distance_between_polylines(pts1, pts2, toggledebug=False):
    def __normed_distance_along_path(polyline_x, polyline_y, polyline_z):
        polyline = np.asarray([polyline_x, polyline_y, polyline_z])
        distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))
        return np.insert(distance, 0, 0) / distance[-1]

    pts2 = np.asarray(align_pseqs(pts1, pts2))
    x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]
    s1 = __normed_distance_along_path(x1, y1, z1)
    s2 = __normed_distance_along_path(x2, y2, z2)

    interpol_xyz1 = interpolate.interp1d(s1, [x1, y1, z1])
    xyz1_on_2 = interpol_xyz1(s2)

    node_to_node_distance = np.sqrt(np.sum((xyz1_on_2 - [x2, y2, z2]) ** 2, axis=0))

    if toggledebug:
        ax = plt.axes(projection='3d')

        ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')

        ax.scatter3D(x2, y2, z2, color='green')
        ax.plot3D(x2, y2, z2, 'green')

        ax.scatter3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], color='black')
        ax.plot3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], 'black', linestyle='dotted')
        plt.show()

    return node_to_node_distance.mean(), xyz1_on_2


def align_pseqs(pseq_src, pseq_tgt):
    # v1 = np.asarray(pseq_src[1]) - np.asarray(pseq_src[0])
    # v2 = np.asarray(pseq_tgt[1]) - np.asarray(pseq_tgt[0])
    # rot = rm.rotmat_between_vectors(v1, v2)
    # pseq_tgt = [np.dot(rot, np.asarray(p)) for p in pseq_tgt]

    p1 = np.asarray(pseq_src[0])
    p2 = np.asarray(pseq_tgt[0])
    pseq_src = [np.asarray(p) - (p1 - p2) for p in pseq_src]
    return pseq_src


def align_pseqs_icp(pseq_src, pseq_tgt):
    rmse, fitness, transmat4 = o3dh.registration_ptpt(np.asarray(pseq_src), np.asarray(pseq_tgt))
    pseq_src = rm.homomat_transform_points(transmat4, pseq_src)
    print(rmse, fitness)
    return pseq_src


def iter_fit(pseq, tor=.001, toggledebug=False):
    pseq = np.asarray(pseq)
    max_err = np.inf
    res_pseq = np.asarray([pseq[0], pseq[-1]])
    checklist = []
    R = .01 + 0.0015
    while max_err > tor:
        res_pseq_inp = np.zeros((1, 3))
        diff = np.linalg.norm(np.diff(res_pseq), axis=1)
        for i, v in enumerate(diff):
            if v == 0:
                continue
            pseq_range = (list(pseq[:, 0]).index(res_pseq[i - 1][0]), list(pseq[:, 0]).index(res_pseq[i][0]))
            res_3d = bu.linear_inp2d(res_pseq[i - 1:i + 1], pseq[pseq_range[0]:pseq_range[1], 0], appendzero=True)
            res_pseq_inp = np.vstack((res_pseq_inp, res_3d))
        err_narry = np.linalg.norm(np.asarray(res_pseq_inp) - pseq, axis=1)
        max_err = np.max(err_narry)
        max_inx = list(err_narry).index(max_err)
        if max_inx in checklist:
            break
        res_pseq = np.unique(np.vstack((res_pseq, pseq[max_inx])), axis=0)
        res_pseq = res_pseq[res_pseq[:, 0].argsort(), :]
        checklist.append(max_inx)

        tangent_pts = []
        bendseq = []
        pos = 0
        for i in range(1, len(res_pseq) - 1):
            v1 = res_pseq[i - 1] - res_pseq[i]
            v2 = res_pseq[i] - res_pseq[i + 1]
            angle = rm.angle_between_vectors(v1, v2)
            n = np.cross(v1, v2)
            if n[2] > 0:
                angle = -angle
            l = R / np.tan((np.pi - abs(angle)) / 2)
            ratio_1 = l / np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
            p1 = res_pseq[i] + (res_pseq[i - 1] - res_pseq[i]) * ratio_1
            ratio_2 = l / np.linalg.norm(res_pseq[i] - res_pseq[i + 1])
            p2 = res_pseq[i] + (res_pseq[i + 1] - res_pseq[i]) * ratio_2
            tangent_pts.append(p1)
            tangent_pts.append(p2)
            pos += np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
            bendseq.append([angle, 0, pos + R * np.pi - l])
        if toggledebug:
            ax = plt.axes()
            bu.plot_pseq_2d(ax, res_pseq_inp)
            bu.plot_pseq_2d(ax, pseq)
            bu.plot_pseq_2d(ax, res_pseq)
            bu.plot_pseq_2d(ax, tangent_pts)
            plt.show()
    ax = plt.axes()
    bu.plot_pseq_2d(ax, res_pseq_inp)
    bu.plot_pseq_2d(ax, pseq)
    bu.plot_pseq_2d(ax, res_pseq)
    bu.plot_pseq_2d(ax, tangent_pts)
    plt.show()
    # for v in bendseq:
    #     print(v)
    return bendseq


if __name__ == '__main__':
    import pickle

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    # goal_pseq = bu.gen_polygen(5, .05)
    goal_pseq = bu.gen_ramdom_curve(length=.1, step=.0005, toggledebug=False)
    # goal_pseq = bu.gen_circle(.05)

    length = bu.cal_length(goal_pseq)
    print(length)
    init_pseq = [(0, 0, 0), (0, length, 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    bs = BendSim.BendSim(thickness=0.0015, width=.002, pseq=init_pseq, rotseq=init_rotseq)

    init_bendseq = iter_fit(goal_pseq, tor=.0005, toggledebug=False)
    pickle.dump(init_bendseq, open('./tmp_bendseq.pkl', 'wb'))

    bs.gen_by_bendseq(init_bendseq, toggledebug=False)
    bs.show(rgba=(0, .7, .7, .7), show_pseq=True)
    goal_pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=rm.rotmat_from_axangle((1, 0, 0), np.pi)),
                                            goal_pseq)
    goal_pseq = align_pseqs_icp(goal_pseq, bs.pseq)
    bu.show_pseq(bu.linear_inp3d(bs.pseq), rgba=(1, 0, 0, 1))
    bu.show_pseq(goal_pseq, rgba=(0, 1, 0, 1))

    # opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=len(init_bendseq))
    # res, cost = opt.solve(init_bendseq=np.asarray([[v[0], v[2]] for v in init_bendseq]).flatten())
    # print(res, cost)
    # bs.bend(res[0], 0, res[1])
    # res = align_pseqs(bs.pseq, goal_pseq)
    # # goal_pseq = align_pseqs_icp(goal_pseq, bs.pseq)
    # bu.show_pseq(bu.linear_inp3d(res), rgba=(1, 0, 0, 1))

    # bs.show()
    # print(bs.pseq)
    base.run()
