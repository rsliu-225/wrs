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


class BendOptimizer(object):
    def __init__(self, bs, init_pseq, goal_pseq):
        self.bs = bs
        self.goal_pseq = goal_pseq
        self.init_pseq = init_pseq
        self.bs.reset(init_pseq)
        self.result = None
        self.cons = []
        b = (-math.pi / 2, math.pi / 2)
        l = (0, cal_length(goal_pseq))
        self.bnds = (b, l)

    def objctive(self, x):
        print(x)
        bs.bend(x[0], np.radians(0), x[1])
        rmse, fitness, _ = o3dh.registration_ptpt(np.asarray(bs.pseq), np.asarray(goal_pseq))
        return rmse

    def update_known(self):
        return NotImplemented

    def con_rot(self, x):
        return NotImplemented

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, method='SLSQP'):
        """

        :param seedjntagls:
        :param tgtpos:
        :param tgtrot:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        # self.update_known()
        self.bs.reset(self.init_pseq)
        # self.addconstraint(self.con_rot, condition="ineq")

        sol = minimize(self.objctive, np.asarray((0, .001)), method=method, bounds=self.bnds, constraints=self.cons)
        print("time cost", time.time() - time_start, sol.success)

        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None


def gen_circle(r):
    pts = []
    for a in np.arange(0, 2 * math.pi, math.pi / 90):
        pts.append([r * math.cos(a), r * math.sin(a), 0])
    return pts


def cal_length(pseq):
    length = 0
    for i in range(len(pseq)):
        if i != 0:
            length += np.linalg.norm(np.asarray(pseq[i]) - np.asarray(pseq[i - 1]))
    # print(np.cumsum(np.sqrt(np.sum(np.diff(np.asarray(pseq), axis=1) ** 2, axis=0))))
    return length


def show_pseq(pseq, rgba):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=0.0005).attach_to(base)


def align_pseqs(pseq1, pseq2):
    # v1 = np.asarray(pseq1[1]) - np.asarray(pseq1[0])
    # v2 = np.asarray(pseq2[1]) - np.asarray(pseq2[0])
    # rot = rm.rotmat_between_vectors(v1, v2)
    # pseq2 = [np.dot(rot, np.asarray(p)) for p in pseq2]
    p1 = np.asarray(pseq1[0])
    p2 = np.asarray(pseq2[0])
    pseq2 = [np.asarray(p) - (p2 - p1) for p in pseq2]
    return pseq2


def normed_distance_along_path(polyline):
    polyline = np.asarray(polyline)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))
    return np.insert(distance, 0, 0) / distance[-1]


def average_distance_between_polylines(xy1, xy2):
    s1 = normed_distance_along_path(xy1)
    s2 = normed_distance_along_path(xy2)

    interpol_xy1 = interpolate.interp1d(s1, xy1)
    xy1_on_2 = interpol_xy1(s2)

    node_to_node_distance = np.sqrt(np.sum((xy1_on_2 - xy2) ** 2, axis=0))

    return node_to_node_distance.mean()


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    goal_pseq = gen_circle(.05)
    length = cal_length(goal_pseq)
    init_pseq = [(0, 0, 0), (0, .01, 0), (0, length, 0)]
    init_rotseq = [np.eye(3), np.eye(3), np.eye(3)]
    bs = BendSim.BendSim(thickness=0.0015, width=.002, pseq=init_pseq, rotseq=init_rotseq)
    bs.bend(np.radians(50), np.radians(0), insert_l=.02)
    res_pseq = bs.pseq
    # aligned_res_pseq = align_pseqs(goal_pseq, res_pseq)

    show_pseq(goal_pseq, rgba=(1, 0, 0, 1))
    # show_pseq(aligned_res_pseq, rgba=(0, 1, 0, 1))
    # show_pseq(res_pseq, rgba=(0, 1, 1, 1))

    rmse, fitness, transmat4 = o3dh.registration_ptpt(np.asarray(res_pseq), np.asarray(goal_pseq))
    matched_res_psed = rm.homomat_transform_points(transmat4, res_pseq)
    show_pseq(matched_res_psed, rgba=(0, 1, 1, 1))
    print(rmse, fitness)

    opt = BendOptimizer(bs, init_pseq, goal_pseq)
    res, cost = opt.solve()
    print(res, cost)
    bs.bend(res[0], 0, res[1])
    show_pseq(bs.pseq, rgba=(0, 1, 0, 1))
    print(bs.pseq)
    base.run()

    # Two example polyline:
    # xy1 = [0, 1, 8, 2, 1.7], [1, 0, 6, 7, 1.9]  # it should work in 3D too
    # xy2 = [.1, .6, 4, 8.3, 2.1, 2.2, 2], [.8, .1, 2, 6.4, 6.7, 4.4, 2.3]
    xy1 = res_pseq
    res = average_distance_between_polylines(xy1, xy2)  # 0.45004578069119189
    print(res)
