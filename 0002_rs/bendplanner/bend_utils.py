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
import bendplanner.bender_config as bconfig


def gen_circle(r, step=math.pi / 90):
    pseq = []
    for a in np.arange(0, 2 * math.pi, step):
        pseq.append(np.asarray([r * math.cos(a), r * math.sin(a), 0]))
    return pseq


def gen_polygen(n, l, do_inp=True):
    pseq = [np.asarray((0, 0, 0))]
    for a in np.linspace(360 / n, 360, n):
        pseq.append(np.asarray(pseq[-1]) + np.asarray([np.cos(np.radians(a)) * l, np.sin(np.radians(a)) * l, 0]))
    if do_inp:
        pseq = linear_inp3d_by_step(np.asarray(pseq))
    return pseq


def gen_ramdom_curve(kp_num=5, length=.5, step=.005, z_max=False, do_inp=True, toggledebug=False):
    pseq = np.asarray([[0, 0, 0]])
    for i in range(kp_num - 1):
        a = random.uniform(-np.pi / 3, np.pi / 3)
        tmp_p = pseq[-1] + np.asarray((np.cos(a) * length / kp_num,
                                       np.sin(a) * length / kp_num,
                                       random.uniform(-z_max, z_max) if z_max else 0))
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


def gen_screw_thread(r, lift_a, rot_num, step=math.pi / 90):
    pseq = []
    for a in np.arange(0, 2 * math.pi * rot_num, step):
        pseq.append(np.asarray([r * math.cos(a), r * math.sin(a), r * a * np.tan(lift_a)]))
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


def show_pseq(pseq, rgba=(1, 0, 0, 1), radius=0.0005, show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=radius).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=radius) \
                .attach_to(base)


def plot_frame(ax, pos, rot):
    length = .005
    x = rot[:, 0] * length
    y = rot[:, 1] * length
    z = rot[:, 2] * length
    ax.arrow3D(pos[0], pos[1], pos[2], x[0], x[1], x[2],
               mutation_scale=10, arrowstyle='->', color='r')
    ax.arrow3D(pos[0], pos[1], pos[2], y[0], y[1], y[2],
               mutation_scale=10, arrowstyle='->', color='g')
    ax.arrow3D(pos[0], pos[1], pos[2], z[0], z[1], z[2],
               mutation_scale=10, arrowstyle='->', color='b')


def plot_pseq(ax3d, pseq,c=None):
    pseq = np.asarray(pseq)
    ax3d.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2],c=c)
    ax3d.grid()


def scatter_pseq(ax3d, pseq, s=2,c=None):
    pseq = np.asarray(pseq)
    ax3d.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], s=s,c=c)
    ax3d.grid()


def plot_pseq_2d(ax, pseq):
    pseq = np.asarray(pseq)
    ax.scatter(pseq[:, 0], pseq[:, 1], s=2)
    ax.grid()


def align_with_goal(bs, goal_pseq, init_rot, init_pos=np.asarray((bconfig.R_BEND, 0, 0)), init_l=bconfig.INIT_L):
    goal_pseq = np.asarray(goal_pseq)
    # goal_rot = np.dot(
    #     # rm.rotmat_from_axangle((1, 0, 0), -np.pi / 2),
    #     np.linalg.inv(init_rot),
    #     rm.rotmat_from_axangle((1, 0, 0), np.pi),
    #     # rm.rotmat_between_vectors(np.cross(goal_pseq[1] - goal_pseq[0], goal_pseq[2] - goal_pseq[1]), [0, 0, 1]),
    # )
    # goal_rot = rm.rotmat_between_vectors(goal_pseq[1] - goal_pseq[0], [0, 1, 0])
    # goal_pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=goal_rot, pos=init_pos), goal_pseq)
    goal_pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=np.linalg.inv(init_rot),
                                                                   pos=init_pos - goal_pseq[0]), goal_pseq)
    bs.move_to_org(init_l)
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


def avg_mindist_between_polylines(pts1, pts2, toggledebug=True):
    pts1 = linear_inp3d_by_step(pts1)
    pts2 = linear_inp3d_by_step(pts2)

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
            plot_pseq(ax, pseq)
            plot_pseq(ax, linear_inp3d_by_step(pseq[res_pids]))
            plot_pseq(ax, pseq[res_pids])
            plt.show()
    print('Num. of fitting result:', len(res_pids))
    return pseq[res_pids]


def objmat4_list_inp(objmat4_list, max_inp=30):
    inp_mat4_list = []
    for i, objmat4 in enumerate(objmat4_list):
        if i > 0:
            inp_mat4_list.append(objmat4_list[i - 1])
            _, angle = rm.axangle_between_rotmat(objmat4_list[i - 1][:3, :3], objmat4[:3, :3])
            if angle < np.pi/180:
                continue
            cnt = int(np.degrees(angle)) if int(np.degrees(angle)) < max_inp else max_inp
            times = [1 / cnt * n for n in range(1, cnt)]
            # print(angle, cnt, times)

            p1, p2 = objmat4_list[i - 1][:3, 3], objmat4[:3, 3]
            r1, r2 = objmat4_list[i - 1][:3, :3], objmat4[:3, :3]

            interp_rot_list = rm.rotmat_slerp(r1, r2, 10)
            interp_p_list = [p1 + (p2 - p1) * t for t in times]

            inp_mat4_list.extend([rm.homomat_from_posrot(p, rot) for p, rot in zip(interp_p_list, interp_rot_list)])
    print("length of interpolation result:", len(inp_mat4_list))

    return inp_mat4_list
