import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
from scipy import interpolate
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import random
import matplotlib.pyplot as plt
import bender_config as bconfig


def gen_circle(r, step=math.pi / 90):
    pseq = []
    for a in np.arange(0, 2 * math.pi, step):
        pseq.append(np.asarray([r * math.cos(a), r * math.sin(a), 0]))
    return pseq


def gen_polygen(n, l, do_inp=True, ):
    pseq = [np.asarray((0, 0, 0))]
    for a in np.linspace(360 / n, 360, n):
        pseq.append(np.asarray(pseq[-1]) + np.asarray([np.cos(np.radians(a)) * l, np.sin(np.radians(a)) * l, 0]))
    if do_inp:
        pseq = linear_inp3d_by_step(np.asarray(pseq))
    return pseq


def gen_ramdom_curve(kp_num=5, length=.5, step=.005, toggle_z=False, do_inp=True, toggledebug=False):
    pseq = np.asarray([[0, 0, 0]])
    for i in range(kp_num - 1):
        a = random.uniform(-np.pi / 3, np.pi / 3)
        tmp_p = pseq[-1] + np.asarray((np.cos(a) * length / kp_num,
                                       np.sin(a) * length / kp_num,
                                       random.uniform(-.002, .002) if toggle_z else 0))
        pseq = np.vstack([pseq, tmp_p])
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='cubic')
    inp_z = interpolate.interp1d(pseq[:, 0], pseq[:, 2], kind='cubic')
    x = np.linspace(0, pseq[-1][0], int(length / step))
    y = inp(x)
    z = inp_z(x)
    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
        ax.scatter3D(x, y, z, color='green')
        plt.show()
    if do_inp:
        pseq = linear_inp3d_by_step(np.asarray(list(zip(x, y, z))))

    return pseq


def linear_inp3d_by_step(pseq, step=.001):
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


def linear_inp3d(pseq, x_space):
    pseq = np.asarray(pseq)
    inp_y = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='linear')
    y = inp_y(x_space)
    inp_z = interpolate.interp1d(pseq[:, 0], pseq[:, 2], kind='linear')
    z = inp_z(x_space)

    return list(zip(x_space, y, z))


def linear_inp2d(pseq, x_space, appendzero=True):
    pseq = np.asarray(pseq)
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='linear')
    y = inp(x_space)
    if appendzero:
        return list(zip(x_space, y, [0] * len(y)))
    else:
        return list(zip(x_space, y))


def cal_length(pseq):
    length = 0
    for i in range(len(pseq)):
        if i != 0:
            length += np.linalg.norm(np.asarray(pseq[i]) - np.asarray(pseq[i - 1]))
    return length


def show_pseq(pseq, rgba=(1, 0, 0, 1), show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=0.0005).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=0.0005).attach_to(
                base)


def plot_pseq(ax3d, pseq):
    pseq = np.asarray(pseq)
    # ax3d.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2])
    ax3d.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], s=2)
    ax3d.grid()


def plot_pseq_2d(ax, pseq):
    pseq = np.asarray(pseq)
    ax.scatter(pseq[:, 0], pseq[:, 1], s=2)
    ax.grid()


def align_with_goal(bs, goal_pseq):
    goal_rot = np.dot(
        rm.rotmat_from_axangle((0, 0, 1), rm.angle_between_vectors(goal_pseq[0] - goal_pseq[1], [0, 1, 0])),
        rm.rotmat_from_axangle((1, 0, 0), np.pi))
    goal_pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=goal_rot, pos=(R, 0, 0)), goal_pseq)
    bs.move_to_org(INIT_L)
    return goal_pseq, np.asarray(bs.pseq)[1:-1]


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
