import math
import pickle
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
import basis.trimesh as trm
import modeling.collision_model as cm


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


def plot_pseq(ax3d, pseq, c=None):
    pseq = np.asarray(pseq)
    ax3d.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], c=c)
    ax3d.grid()


def scatter_pseq(ax3d, pseq, s=2, c=None):
    pseq = np.asarray(pseq)
    ax3d.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], s=s, c=c)
    ax3d.grid()


def plot_pseq_2d(ax, pseq):
    pseq = np.asarray(pseq)
    ax.scatter(pseq[:, 0], pseq[:, 1], s=2)
    ax.grid()


def align_with_goal(bs, goal_pseq, init_rot, init_pos=np.asarray((bconfig.R_BEND, 0, 0)), init_l=bconfig.INIT_L):
    goal_pseq = np.asarray(goal_pseq)
    goal_pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=np.linalg.inv(init_rot),
                                                                   pos=init_pos - goal_pseq[0]), goal_pseq)
    bs.move_to_org(init_l)
    return goal_pseq, np.asarray(bs.pseq)[1:]


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


def avg_polylines_dist_err(pts1, pts2, toggledebug=False):
    def __normed_distance_along_path(x, y, z):
        polyline = np.asarray([x, y, z])
        distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))
        return np.insert(distance, 0, 0) / distance[-1]

    pts1 = np.round(np.asarray(pts1), decimals=5)
    pts2 = np.round(np.asarray(pts2), decimals=5)
    x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]

    s1 = __normed_distance_along_path(x1, y1, z1)
    s2 = __normed_distance_along_path(x2, y2, z2)

    interp_pts1 = interpolate.interp1d(s1, [x1, y1, z1])
    pts1_on_2 = interp_pts1(s2)

    ptp_dist = np.sqrt(np.sum((pts1_on_2 - [x2, y2, z2]) ** 2, axis=0))

    if toggledebug:
        print('Sum. distance between polylines:', ptp_dist.sum())
        print('Avg. distance between polylines:', ptp_dist.mean())
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        # z_max = max([abs(np.max(z1)), abs(np.max(z2))])
        ax.set_xlim([0, 0.08])
        ax.set_ylim([-0.04, 0.04])
        ax.set_zlim([-0.04, 0.04])
        # ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')
        ax.scatter3D(x2, y2, z2, color='black')
        ax.plot3D(x2, y2, z2, 'black')
        ax.scatter3D(pts1_on_2[0], pts1_on_2[1], pts1_on_2[2], color='g')
        # ax.plot3D(pts1_on_2[0], pts1_on_2[1], pts1_on_2[2], 'g')
        plt.show()

    err = ptp_dist.mean()

    return err, pts1_on_2


def mindist_err(kpts, pts, toggledebug=False):
    from sklearn.neighbors import KDTree
    nearest_pts = []
    err_list = []

    res_pts = linear_inp3d_by_step(pts, step=0.0001)
    kpts = linear_inp3d_by_step(kpts, step=0.001)
    kdt = KDTree(res_pts, leaf_size=100, metric='euclidean')
    for p in kpts:
        distances, indices = kdt.query([p], k=1, return_distance=True)
        err_list.append(distances[0][0])
        nearest_pts.append(res_pts[indices[0][0]])
    if toggledebug:
        print('Sum. distance between polylines:', np.asarray(err_list).sum())
        print('Avg. distance between polylines:', np.asarray(err_list).mean())
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim([0, 0.08])
        ax.set_ylim([-0.04, 0.04])
        ax.set_zlim([-0.04, 0.04])
        # ax.scatter3D(x1, y1, z1, color='red')
        plot_pseq(ax, kpts, c='r')
        scatter_pseq(ax, kpts, c='r')
        plot_pseq(ax, pts, c='g')
        # scatter_pseq(ax, kpts2, c='g', s=10)
        scatter_pseq(ax, nearest_pts, c='black', s=10)
        plt.show()
    err = np.asarray(err_list).sum()

    return err, nearest_pts


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


def get_init_rot(pseq):
    pseq = np.asarray(pseq)
    v1 = pseq[0] - pseq[1]
    v2 = pseq[1] - pseq[2]
    rot_n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
    x = np.cross(v1, rot_n)

    return np.asarray([-rm.unit_vector(x), -rm.unit_vector(v1), -rm.unit_vector(rot_n)]).T


def decimate_pseq(pseq, tor=.001, r=None, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    ptr = 0
    while ptr < len(res_pids) - 1:
        max_err, max_inx = __ps2seg_max_dist(pseq[res_pids[ptr]], pseq[res_pids[ptr + 1]],
                                             pseq[res_pids[ptr]:res_pids[ptr + 1]])
        if max_err > tor:
            curr = max_inx + res_pids[ptr]
            # if r is not None and len(res_pids) > 2:
            #     a = rm.angle_between_vectors(pseq[res_pids[ptr]] - pseq[curr],
            #                                  pseq[res_pids[ptr + 1]] - pseq[res_pids[ptr]])
            #     print(a)
            #     if abs(a * r) > np.linalg.norm(pseq[res_pids[ptr]] - pseq[curr]):
            #         ptr += 1
            #         continue
            res_pids.append(curr)
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


def get_rotseq_by_pseq(pseq):
    rotseq = []
    for i in range(1, len(pseq) - 1):
        v1 = pseq[i - 1] - pseq[i]
        v2 = pseq[i] - pseq[i + 1]
        n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
        x = np.cross(v1, n)
        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
        rotseq.append(rot)
    rotseq = [rotseq[0]] + rotseq + [rotseq[-1]]
    print(len(rotseq))
    return rotseq


def is_collinearity(p, seg):
    p = np.asarray(p)
    seg = np.asarray(seg)
    if round(np.linalg.norm(p - seg[0]) + np.linalg.norm(p - seg[1]), 8) == round(np.linalg.norm(seg[0] - seg[1]), 8):
        return True
    # if np.linalg.norm(p - seg[0]) + np.linalg.norm(seg[0] - seg[1]) == np.linalg.norm(p - seg[1]):
    #     return True
    # if np.linalg.norm(p - seg[1]) + np.linalg.norm(seg[0] - seg[1]) == np.linalg.norm(p - seg[0]):
    #     return True
    return False


def gen_stick(pseq, rotseq, r, section=5, toggledebug=False):
    vertices = []
    faces = []
    for i, p in enumerate(pseq):
        for a in np.linspace(-np.pi, np.pi, section + 1):
            vertices.append(p + rotseq[i][:, 0] * r * np.sin(a)
                            + rotseq[i][:, 2] * r * np.cos(a))
    for i in range((section + 1) * (len(pseq) - 1)):
        if i % (section + 1) == 0:
            for v in range(i, i + section):
                faces.extend([[v, v + section + 1, v + section + 2], [v, v + section + 2, v + 1]])
    if toggledebug:
        show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
        show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj', cdprimit_type='surface_balls')


def pseq2bendset(res_pseq, bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L, toggledebug=False):
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    tangent_pts = []
    bendseq = []
    n_pre = None
    rot_a = 0
    lift_a = 0
    pos = 0
    l_pos = 0
    for i in range(1, len(res_pseq) - 1):
        v1 = res_pseq[i - 1] - res_pseq[i]
        v2 = res_pseq[i] - res_pseq[i + 1]
        v3 = res_pseq[i - 1] - res_pseq[i + 1]
        pos += np.linalg.norm(v1)
        n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
        v2_xy = v2 - v2 * n
        n_xz = n - n * v2
        v2_xz = v2 - v2 * rm.unit_vector(v1)
        v3_yz = v3 - v3 * np.cross(v1, n)

        # bend_a = rm.angle_between_vectors(v1, v2_xy)
        bend_a = rm.angle_between_vectors(v1, v2)
        if round(bend_a, 8) == 0:
            print(bend_a)
            continue

        if n_pre is not None:
            # v2_yz = v2 - v2 * (np.cross(n_pre, rm.unit_vector(v1)))
            # lift_a = np.pi / 2 - rm.angle_between_vectors(n_pre, v3_yz)
            # tmp_a = rm.angle_between_vectors(np.cross(v1, n_pre), np.cross(v1, v3_yz))
            # if tmp_a < np.pi / 2:
            #     lift_a = -lift_a
            a = rm.angle_between_vectors(n_pre, n)
            # a = rm.angle_between_vectors(n_pre, n_xz)
            tmp_a = rm.angle_between_vectors(v1, np.cross(n_pre, n))
            if tmp_a is not None and tmp_a > np.pi / 2:
                rot_a += a
            else:
                rot_a -= a

        n_pre = n
        l = (bend_r * np.tan(abs(bend_a) / 2)) / np.cos(abs(lift_a))
        ratio_1 = l / np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
        ratio_2 = l / np.linalg.norm(res_pseq[i] - res_pseq[i + 1])
        p_tan1 = res_pseq[i] + (res_pseq[i - 1] - res_pseq[i]) * ratio_1
        p_tan2 = res_pseq[i] + (res_pseq[i + 1] - res_pseq[i]) * ratio_2

        if i > 1 and is_collinearity(p_tan1, [res_pseq[i - 1], tangent_pts[-1]]):
            scatter_pseq(ax, [p_tan1], s=20, c='gray')
            bendseq[-1][0] += bend_a

        else:
            if i == 1:
                l_pos += np.linalg.norm(p_tan1 - res_pseq[i - 1])
            else:
                l_pos += np.linalg.norm(p_tan1 - tangent_pts[-1])
                l_pos += abs(bendseq[-1][0]) * bend_r / np.cos(abs(bendseq[-1][1]))
            bendseq.append([bend_a, lift_a, rot_a, l_pos + init_l])
        tangent_pts.extend([p_tan1, p_tan2])

        x = np.cross(v1, n)
        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
        if toggledebug:
            gm.gen_frame(res_pseq[i - 1], rot, length=.02, thickness=.001).attach_to(base)
            plot_frame(ax, res_pseq[i - 1], rot)
    if toggledebug:
        ax.set_xlim([0, 0.1])
        ax.set_ylim([-0.05, 0.05])
        ax.set_zlim([-0.05, 0.05])
        # goal_pseq = pickle.load(open('../run_plan/goal_pseq.pkl', 'rb'))
        plot_pseq(ax, res_pseq)
        # plot_pseq(ax, goal_pseq)
        scatter_pseq(ax, [res_pseq[0]], s=10, c='y')
        scatter_pseq(ax, res_pseq[1:], s=10, c='g')
        scatter_pseq(ax, tangent_pts, s=10, c='r')
        plt.show()

    return bendseq
