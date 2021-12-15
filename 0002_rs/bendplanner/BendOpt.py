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


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, init_rotseq, goal_pseq, bend_times=1):
        self.bs = bs
        self.bend_times = bend_times
        self.goal_pseq = goal_pseq
        self.init_pseq = init_pseq
        self.init_rotseq = init_rotseq
        self.bs.reset(self.init_pseq, self.init_rotseq)
        self.total_len = cal_length(goal_pseq)

        # self.result = None
        self.cons = []
        # self.rb = (-math.pi / 3, math.pi / 3)
        self.rb = (0, math.pi / 2)
        self.lb = (0, self.total_len - self.bs.bend_r * math.pi)
        self.bnds = (self.rb, self.lb) * self.bend_times
        print(self.bnds)

    def objctive(self, x):
        self.bs.reset(self.init_pseq, self.init_rotseq)
        # print('------objective------')
        # print(x)
        try:
            self.bend_x(x)
            pseq = linear_inp(bs.pseq)
            err, fitness, _ = o3dh.registration_ptpt(np.asarray(pseq), np.asarray(goal_pseq))
            # err, _ = average_distance_between_polylines(np.asarray(bs.pseq), np.asarray(goal_pseq), toggledebug=False)
        except:
            err = 1
        print('cost:', err)
        return err

    def bend_x(self, x):
        for i in range(int(len(x) / 2)):
            bs.bend(x[2 * i], np.radians(0), x[2 * i + 1])
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
        init_pseq = self.bend_x(np.asarray(list(zip(pos, rot))).flatten())
        show_pseq(linear_inp(init_pseq), rgba=(0, 0, 1, 1))
        # base.run()

        return np.asarray(init_pseq)

    def equal_init(self):
        init = []
        for i in range(self.bend_times):
            # init.append(np.radians(360 / self.bend_times))
            init.append(random.uniform(self.rb[0], self.rb[1]))
            init.append(i * self.total_len / self.bend_times)
        init_pseq = self.bend_x(init)
        show_pseq(linear_inp(init_pseq), rgba=(0, 0, 1, 1))
        # self.bs.show(rgba=(0, 0, 1, .5))
        return np.asarray(init)

    def solve(self, method='SLSQP'):
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
        init = self.random_init()
        # init = self.equal_init()
        for i in range(int(len(init) / 2) - 1):
            self.addconstraint_sort(i)
        sol = minimize(self.objctive, init, method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        if sol.success:
            print(sol.x)
            ax = plt.axes()
            ax.grid()
            ax.scatter([v for i, v in enumerate(sol.x) if i % 2 != 0], [v for i, v in enumerate(sol.x) if i % 2 == 0],
                       color='red')
            ax.plot([v for i, v in enumerate(sol.x) if i % 2 != 0], [v for i, v in enumerate(sol.x) if i % 2 == 0],
                    color='red')
            ax.scatter([v for i, v in enumerate(init) if i % 2 != 0], [v for i, v in enumerate(init) if i % 2 == 0],
                       color='blue')
            ax.plot([v for i, v in enumerate(init) if i % 2 != 0], [v for i, v in enumerate(init) if i % 2 == 0],
                    color='blue')
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


def gen_circle(r):
    pts = []
    for a in np.arange(0, 2 * math.pi, math.pi / 90):
        pts.append([r * math.cos(a), r * math.sin(a), 0])
    return pts


def gen_polygen(n, l):
    pseq = [np.asarray((0, 0, 0))]
    for a in np.linspace(360 / n, 360, n):
        pseq.append(np.asarray(pseq[-1]) + np.asarray([np.cos(np.radians(a)) * l, np.sin(np.radians(a)) * l, 0]))

    return pseq


def gen_ramdom_curve(kp_num=5, length=.5, step=.01, toggledebug=False):
    pseq = np.asarray([[0, 0, 0]])
    for i in range(kp_num - 1):
        a = random.uniform(-np.pi / 3, np.pi / 3)
        tmp_p = pseq[-1] + np.asarray((np.cos(a) * length / kp_num, np.sin(a) * length / kp_num, 0))
        pseq = np.vstack([pseq, tmp_p])
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='cubic')
    x = np.linspace(0, pseq[-1][0], int(length / step))
    y = inp(x)
    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
        ax.scatter3D(x, y, [0] * len(x), color='green')
        print(list(zip(x, y, [0] * len(x))))
        plt.show()
    show_pseq(pseq)
    pseq = linear_inp(np.asarray(list(zip(x, y, [0] * len(x)))))
    show_pseq(pseq, rgba=(1, 1, 0, 1))
    # base.run()

    return pseq


def linear_inp(pseq, step=.001):
    inp_pseq = []
    for i in range(len(pseq) - 1):
        p1 = np.asarray(pseq[i])
        p2 = np.asarray(pseq[i + 1])
        diff = p2 - p1
        inp_num = int(np.linalg.norm(p1 - p2) / step)
        if inp_num == 0:
            inp_pseq.append(p1)
        else:
            for j in range(inp_num):
                inp_pseq.append(p1 + j * diff / inp_num)
    return np.asarray(inp_pseq)


def cal_length(pseq):
    length = 0
    for i in range(len(pseq)):
        if i != 0:
            length += np.linalg.norm(np.asarray(pseq[i]) - np.asarray(pseq[i - 1]))
    # print(np.cumsum(np.sqrt(np.sum(np.diff(np.asarray(pseq), axis=1) ** 2, axis=0))))
    return length


def show_pseq(pseq, rgba=(1, 0, 0, 1), show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=0.0005).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=0.0005).attach_to(
                base)

def plot_pseq(pseq):
    ax = plt.axes(projection='3d')
    ax.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
    ax.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='green')
    ax.grid()
    plt.show()


def align_pseqs(pseq_src, pseq_tgt):
    # v1 = np.asarray(pseq1[1]) - np.asarray(pseq1[0])
    # v2 = np.asarray(pseq2[1]) - np.asarray(pseq2[0])
    # rot = rm.rotmat_between_vectors(v1, v2)
    # pseq2 = [np.dot(rot, np.asarray(p)) for p in pseq2]
    p1 = np.asarray(pseq_src[0])
    p2 = np.asarray(pseq_tgt[0])
    pseq_src = [np.asarray(p) - (p1 - p2) for p in pseq_src]
    return pseq_src


def align_pseqs_icp(pseq_src, pseq_tgt):
    rmse, fitness, transmat4 = o3dh.registration_ptpt(np.asarray(pseq_src), np.asarray(pseq_tgt))
    pseq_src = rm.homomat_transform_points(transmat4, pseq_src)
    print(rmse, fitness)
    return pseq_src


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    # goal_pseq = linear_inp(gen_polygen(5, .05), step=.001)
    goal_pseq = gen_ramdom_curve()

    # goal_pseq = gen_circle(.05)

    length = cal_length(goal_pseq)
    init_pseq = [(0, 0, 0), (0, length, 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    bs = BendSim.BendSim(thickness=0.0015, width=.002, pseq=init_pseq, rotseq=init_rotseq)

    opt = BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=5)
    res, cost = opt.solve()
    print(res, cost)
    bs.bend(res[0], 0, res[1])
    # res = align_pseqs(bs.pseq, goal_pseq)
    # goal_pseq = align_pseqs_icp(goal_pseq, bs.pseq)
    show_pseq(linear_inp(bs.pseq), rgba=(1, 0, 0, 1))
    show_pseq(goal_pseq, rgba=(0, 1, 0, 1))

    # bs.show()
    print(bs.pseq)
    base.run()
