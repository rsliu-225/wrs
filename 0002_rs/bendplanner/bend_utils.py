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


def gen_circle(r, step=math.pi / 90):
    pts = []
    for a in np.arange(0, 2 * math.pi, step):
        pts.append([r * math.cos(a), r * math.sin(a), 0])
    return pts


def gen_polygen(n, l, do_inp=True, ):
    pseq = [np.asarray((0, 0, 0))]
    for a in np.linspace(360 / n, 360, n):
        pseq.append(np.asarray(pseq[-1]) + np.asarray([np.cos(np.radians(a)) * l, np.sin(np.radians(a)) * l, 0]))
    if do_inp:
        pseq = linear_inp3d(np.asarray(pseq))
    return pseq


def gen_ramdom_curve(kp_num=5, length=.5, step=.005, do_inp=True, toggledebug=False):
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
    if do_inp:
        pseq = linear_inp3d(np.asarray(list(zip(x, y, [0] * len(x)))))

    return pseq


def linear_inp3d(pseq, step=.001):
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
    # print(np.cumsum(np.sqrt(np.sum(np.diff(np.asarray(pseq), axis=1) ** 2, axis=0))))
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
    ax3d.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
    ax3d.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='green')
    ax3d.grid()


def plot_pseq_2d(ax, pseq):
    pseq = np.asarray(pseq)
    ax.scatter(pseq[:, 0], pseq[:, 1], s=1)
    ax.grid()

