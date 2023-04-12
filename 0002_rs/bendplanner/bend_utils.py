import math
import random

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import interpolate
from sklearn.neighbors import KDTree

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import basis.trimesh as trm
import bendplanner.bender_config as bconfig
import modeling.collision_model as cm
import modeling.geometric_model as gm


def plot_frame(ax, pos, rot):
    length = .005
    x = rot[:, 0] * length
    y = rot[:, 1] * length
    z = rot[:, 2] * length
    ax.arrow3D(pos[0], pos[1], pos[2], x[0], x[1], x[2], mutation_scale=10, arrowstyle='->', color='r')
    ax.arrow3D(pos[0], pos[1], pos[2], y[0], y[1], y[2], mutation_scale=10, arrowstyle='->', color='g')
    ax.arrow3D(pos[0], pos[1], pos[2], z[0], z[1], z[2], mutation_scale=10, arrowstyle='->', color='b')


def plot_frameseq(ax, pseq, rotseq, skip=5):
    for i in range(len(pseq)):
        if i % skip == 0:
            plot_frame(ax, pseq[i], rotseq[i])


def plot_pseq(ax3d, pseq, c=None, linestyle='-'):
    pseq = np.asarray(pseq)
    ax3d.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], c=c, linestyle=linestyle)
    ax3d.grid()


def scatter_pseq(ax3d, pseq, s=2, c=None):
    pseq = np.asarray(pseq)
    ax3d.scatter3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], s=s, c=c)
    ax3d.grid()


def plot_pseq_2d(ax, pseq):
    pseq = np.asarray(pseq)
    ax.scatter(pseq[:, 0], pseq[:, 1], s=2)
    ax.grid()


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


def gen_bspline(kp_num=5, length=.5, y_max=.02, toggledebug=False):
    kpts = [(0, 0, 0)]
    for j in range(kp_num - 1):
        kpts.append(((j + 1) * .02, random.uniform(-y_max, y_max), random.uniform(-y_max, y_max)))

    kpts = np.asarray(kpts).transpose()
    inp_pseq = np.asarray(
        interpolate.splev(np.linspace(0, 1, 200),
                          interpolate.splprep(kpts, k=min([kpts.shape[1] - 1, 5]))[0], der=0)
    ).transpose()
    inp_pseq = np.asarray(inp_pseq) - inp_pseq[0]
    org_len = np.linalg.norm(np.diff(inp_pseq, axis=0), axis=1).sum()
    return length * np.asarray(inp_pseq) / org_len


def gen_sgl_curve(pseq, step=.001, do_inp=True, toggledebug=False):
    length = cal_length(pseq)
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


def inp_rotp_by_step(pseq, rotseq, step=.001):
    inp_pseq = []
    inp_rotseq = []
    for i in range(len(pseq) - 1):
        p1 = np.asarray(pseq[i])
        p2 = np.asarray(pseq[i + 1])
        r1 = np.asarray(rotseq[i])
        r2 = np.asarray(rotseq[i + 1])
        diff = p2 - p1
        inp_num = int(np.linalg.norm(p1 - p2) / step)
        if inp_num == 0:
            inp_pseq.append(p1)
            inp_rotseq.append(r1)
        else:
            for j in range(inp_num):
                if (r1 == r2).all():
                    insert_rot = r1
                else:
                    rotmat_list = rm.rotmat_slerp(r1, r2, 10)
                    inx = np.floor((j / inp_num) * len(rotmat_list)) - 1
                    if inx > 9:
                        inx = 9
                    if inx == -1:
                        inx = 0
                    insert_rot = rotmat_list[int(inx)]
                inp_pseq.append(p1 + j * diff / inp_num)
                inp_rotseq.append(insert_rot)

    return np.asarray(inp_pseq), np.asarray(inp_rotseq)


def linear_inp3d(pseq, x):
    pseq = np.asarray(pseq)
    inp_y = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='linear')
    y = inp_y(x)
    inp_z = interpolate.interp1d(pseq[:, 0], pseq[:, 2], kind='linear')
    z = inp_z(x)

    return list(zip(x, y, z))


def linear_inp2d(pseq, x, appendzero=True):
    pseq = np.asarray(pseq)
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind='linear')
    y = inp(x)
    if appendzero:
        return list(zip(x, y, [0] * len(y)))
    else:
        return list(zip(x, y))


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


def align_with_init(bs, pseq, init_rot, rotseq=None, init_pos=np.asarray((bconfig.R_BEND, 0, 0)),
                    init_l=bconfig.INIT_L):
    pseq = np.asarray(pseq) - pseq[0]
    pseq = rm.homomat_transform_points(rm.homomat_from_posrot(rot=np.linalg.inv(init_rot),
                                                              pos=init_pos - pseq[0]), pseq)
    if rotseq is not None:
        rotseq = [np.dot(np.linalg.inv(init_rot), r) for r in rotseq]
    bs.move_to_org(init_l)
    return pseq, rotseq


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
        center = np.mean(pts1, axis=0)
        ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
        ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
        ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
        # ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')
        ax.scatter3D(x2, y2, z2, color='black')
        ax.plot3D(x2, y2, z2, 'black')
        ax.scatter3D(pts1_on_2[0], pts1_on_2[1], pts1_on_2[2], color='g')
        # ax.plot3D(pts1_on_2[0], pts1_on_2[1], pts1_on_2[2], 'g')
        plt.show()

    err = ptp_dist.mean()

    return err, pts1_on_2


def mindist_err(res_pts, goal_pts, res_rs=None, goal_rs=None, toggledebug=False, type='max'):
    res_pts, goal_pts = np.asarray(res_pts) * 1000, np.asarray(goal_pts) * 1000
    nearest_pts = []
    nearest_rs = []
    pos_err_list = []
    n_err_list = []
    if res_rs is None:
        res_pts = linear_inp3d_by_step(res_pts, step=.1)
    else:
        res_rs, goal_rs = np.asarray(res_rs), np.asarray(goal_rs)
        res_pts, res_rs = inp_rotp_by_step(res_pts, res_rs, step=.1)
    kdt = KDTree(res_pts, leaf_size=100, metric='euclidean')
    for i in range(len(goal_pts)):
        distances, indices = kdt.query([goal_pts[i]], k=1, return_distance=True)
        pos_err_list.append(distances[0][0])
        nearest_pts.append(res_pts[indices[0][0]])
        if res_rs is not None:
            n_err_list.append(np.degrees(rm.angle_between_vectors(goal_rs[i][:, 0], res_rs[indices[0][0]][:, 0])))
            nearest_rs.append(res_rs[indices[0][0]])
        else:
            n_err_list.append(0)
    if toggledebug:
        print('Sum. distance between polylines:', np.asarray(pos_err_list).sum())
        print('Avg. distance between polylines:', np.asarray(pos_err_list).mean())
        print('Max. distance between polylines:', max(pos_err_list))
        if res_rs is not None:
            print('Sum. angel err between polylines:', np.asarray(n_err_list).sum())
            print('Avg. angel err between polylines:', np.asarray(n_err_list).mean())
            print('Max. angel err between polylines:', max(n_err_list))
        ax = plt.axes(projection='3d')
        center = np.mean(res_pts, axis=0)
        ax.set_xlim([center[0] - 50, center[0] + 50])
        ax.set_ylim([center[1] - 50, center[1] + 50])
        ax.set_zlim([center[2] - 50, center[2] + 50])
        ax.set_xlabel('X(mm)')
        ax.set_ylabel('Y(mm)')
        ax.set_zlabel('Z(mm)')
        # ax.scatter3D(x1, y1, z1, color='red')
        plot_pseq(ax, res_pts, c='r')
        # scatter_pseq(ax, res_pts, c='r')
        plot_pseq(ax, goal_pts, c='black')
        # scatter_pseq(ax, goal_pts, c='g', s=10)
        # scatter_pseq(ax, nearest_pts, c='black', s=10)
        if res_rs is not None:
            plot_frameseq(ax, nearest_pts, nearest_rs, skip=10)
            plot_frameseq(ax, goal_pts, goal_rs, skip=10)
        plt.show()
    # err = np.asarray(pos_err_list).sum() * 10 + np.asarray(n_err_list).sum()/10
    if type == 'max':
        err = max(pos_err_list)
    elif type == 'sum':
        err = np.asarray(pos_err_list).sum()
    else:
        err = np.asarray(pos_err_list).mean()

    return err, np.asarray(nearest_pts) / 1000


def mindist_err_sfc(res_pts, goal_pts, toggledebug=False):
    from sklearn.neighbors import KDTree
    res_pts, goal_pts = np.asarray(res_pts), np.asarray(goal_pts)
    nearest_pts = []
    pos_err_list = []
    res_pts = linear_inp3d_by_step(res_pts, step=.0001)

    kdt = KDTree(res_pts, leaf_size=100, metric='euclidean')
    for i in range(len(goal_pts)):
        distances, indices = kdt.query([goal_pts[i]], k=1, return_distance=True)
        pos_err_list.append(distances[0][0])
        nearest_pts.append(res_pts[indices[0][0]])

    if toggledebug:
        print('Sum. distance between polylines:', np.asarray(pos_err_list).sum())
        print('Avg. distance between polylines:', np.asarray(pos_err_list).mean())
        ax = plt.axes(projection='3d')
        center = np.mean(res_pts, axis=0)
        ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
        ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
        ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
        # ax.scatter3D(x1, y1, z1, color='red')
        plot_pseq(ax, res_pts, c='r')
        scatter_pseq(ax, res_pts, c='r')
        plot_pseq(ax, goal_pts, c='g')
        scatter_pseq(ax, goal_pts, c='g', s=10)
        scatter_pseq(ax, nearest_pts, c='black', s=10)
        plt.show()
    err = np.asarray(pos_err_list).sum() * 10

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
    if np.array_equal(rot_n, np.asarray([0, 0, 0])):
        return -np.eye(3)
    x = np.cross(v1, rot_n)
    return np.asarray([-rm.unit_vector(x), -rm.unit_vector(v1), -rm.unit_vector(rot_n)]).T


def decimate_pseq(pseq, tor=.001, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    ptr = 0
    while ptr < len(res_pids) - 1:
        max_err, max_inx = __ps2seg_max_dist(pseq[res_pids[ptr]], pseq[res_pids[ptr + 1]],
                                             pseq[res_pids[ptr]:res_pids[ptr + 1]])
        if max_err > tor:
            curr = max_inx + res_pids[ptr]
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
    print(f'Num. of fitting result:{len(res_pids)}/{len(pseq)}')
    return np.asarray(pseq[res_pids]), get_rotseq_by_pseq(pseq[res_pids]), res_pids


def decimate_pseq_avg(pseq, tor=.001, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    avg_err = np.inf
    while avg_err / 1000 > tor:
        max_err = 0
        max_inx = -1
        for i in range(len(res_pids) - 1):
            max_err_tmp, max_inx_tmp = __ps2seg_max_dist(pseq[res_pids[i]], pseq[res_pids[i + 1]],
                                                         pseq[res_pids[i]:res_pids[i + 1]])
            if max_err_tmp > max_err:
                max_err = max_err_tmp
                max_inx = res_pids[i] + max_inx_tmp
        res_pids.append(max_inx)
        res_pids = sorted(res_pids)
        if toggledebug:
            ax = plt.axes(projection='3d')
            plot_pseq(ax, pseq)
            plot_pseq(ax, linear_inp3d_by_step(pseq[res_pids]))
            plot_pseq(ax, pseq[res_pids])
            plt.show()
        avg_err, _ = mindist_err(pseq[res_pids], pseq, toggledebug=False, type='avg')
    print(f'Num. of fitting result:{len(res_pids)}/{len(pseq)}')
    return np.asarray(pseq[res_pids]), get_rotseq_by_pseq(pseq[res_pids]), res_pids


def decimate_pseq_by_cnt(pseq, cnt=10, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    while len(res_pids) < cnt:
        max_err = 0
        max_inx = -1
        for i in range(len(res_pids) - 1):
            max_err_tmp, max_inx_tmp = __ps2seg_max_dist(pseq[res_pids[i]], pseq[res_pids[i + 1]],
                                                         pseq[res_pids[i]:res_pids[i + 1]])
            if max_err_tmp > max_err:
                max_err = max_err_tmp
                max_inx = res_pids[i] + max_inx_tmp
        res_pids.append(max_inx)
        res_pids = sorted(res_pids)
        if toggledebug:
            ax = plt.axes(projection='3d')
            plot_pseq(ax, pseq)
            plot_pseq(ax, linear_inp3d_by_step(pseq[res_pids]))
            plot_pseq(ax, pseq[res_pids])
            plt.show()
    print(f'Num. of fitting result:{len(res_pids)}/{len(pseq)}')
    return np.asarray(pseq[res_pids]), get_rotseq_by_pseq(pseq[res_pids]), res_pids


def decimate_pseq_by_cnt_curvature(pseq, thresh_r=bconfig.R_BEND, cnt=10, toggledebug=False):
    pseq = np.asarray(pseq)
    curvature_list, r_list, torsion_list = cal_curvature(pseq, show=False)
    inx_real = [i for i in range(len(r_list)) if r_list[i] > thresh_r]
    pseq_tmp = np.asarray(
        [pseq[0]] + [pseq[i + 1] for i in range(len(r_list) - 2) if r_list[i] > thresh_r] + [pseq[-1]])
    res_pids = [0, len(pseq_tmp) - 1]
    while len(res_pids) < cnt:
        max_err = 0
        max_inx = -1
        for i in range(len(res_pids) - 1):
            max_err_tmp, max_inx_tmp = __ps2seg_max_dist(pseq_tmp[res_pids[i]], pseq_tmp[res_pids[i + 1]],
                                                         pseq_tmp[res_pids[i]:res_pids[i + 1]])
            if max_err_tmp > max_err:
                max_err = max_err_tmp
                max_inx = res_pids[i] + max_inx_tmp
        res_pids.append(max_inx)
        res_pids = sorted(res_pids)

    res_pids = [inx_real[i] for i in res_pids]
    if toggledebug:
        ax = plt.axes(projection='3d')
        plot_pseq(ax, pseq)
        plot_pseq(ax, pseq[res_pids])
        plot_pseq(ax, pseq_tmp)
        plt.show()
    print(f'Num. of fitting result:{len(res_pids)}/{len(pseq)}')
    return np.asarray(pseq[res_pids]), get_rotseq_by_pseq(pseq[res_pids]), res_pids


def decimate_pseq_by_cnt_uni(pseq, cnt=10, toggledebug=False):
    pseq = linear_inp3d_by_step(pseq, .0001)
    uni_len = cal_length(pseq) / (cnt - 1)
    tmp_len = 0
    pseq_res = [pseq[0]]
    for i in range(len(pseq) - 1):
        tmp_len += np.linalg.norm(pseq[i + 1] - pseq[i])
        if tmp_len > uni_len:
            pseq_res.append(pseq[i] + (tmp_len - uni_len) * rm.unit_vector(pseq[i] - pseq[i + 1]))
            tmp_len = np.linalg.norm(pseq[i + 1] - pseq_res[-1])
    if len(pseq) < cnt:
        pseq_res.append(pseq[-1])
    else:
        pseq_res[-1] = pseq[-1]

    if toggledebug:
        ax = plt.axes(projection='3d')
        plot_pseq(ax, pseq)
        plot_pseq(ax, pseq_res)
        scatter_pseq(ax, pseq)
        scatter_pseq(ax, pseq_res)
        plt.show()
    print(f'Num. of fitting result:{len(pseq_res)}/{len(pseq)}')

    return np.asarray(pseq_res), get_rotseq_by_pseq(pseq_res)


def decimate_rotpseq(pseq, rotseq, tor=.001, toggledebug=False):
    pseq = np.asarray(pseq)
    res_pids = [0, len(pseq) - 1]
    ptr = 0
    while ptr < len(res_pids) - 1:
        max_err, max_inx = __ps2seg_max_dist(pseq[res_pids[ptr]], pseq[res_pids[ptr + 1]],
                                             pseq[res_pids[ptr]:res_pids[ptr + 1]])
        if max_err > tor:
            curr = max_inx + res_pids[ptr]
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
    print(f'Num. of fitting result:{len(res_pids)}/{len(pseq)}')

    return pseq[res_pids], [r for i, r in enumerate(rotseq) if i in res_pids]


def get_rotseq_by_pseq(pseq):
    rotseq = []
    pre_n = None
    for i in range(1, len(pseq) - 1):
        v1 = pseq[i - 1] - pseq[i]
        v2 = pseq[i] - pseq[i + 1]
        n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
        if pre_n is not None and rm.angle_between_vectors(n, pre_n) is not None:
            if rm.angle_between_vectors(n, pre_n) > np.pi / 2:
                n = -n
        x = np.cross(v1, n)
        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
        rotseq.append(rot)
        pre_n = n
    rotseq = [rotseq[0]] + rotseq + [rotseq[-1]]
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


def voxelize(pseq, rotseq, r=bconfig.THICKNESS / 2):
    objcm = gen_stick(pseq, rotseq, r)
    pcd_narry, _ = objcm.sample_surface(radius=.0005)
    pcd = o3dh.nparray2o3dpcd(np.asarray(pcd_narry))
    # pcd.scale(1, center=(0, 0, 0))
    return o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=bconfig.THICKNESS)


def onehot_voxel(voxel_grid, bnd=(200, 200, 200)):
    onehot = np.zeros(bnd)
    for v in voxel_grid.get_voxels():
        onehot[v.grid_index[0]][v.grid_index[1]][v.grid_index[2]] = 1
    return onehot


def visualize_voxel(voxel_grids, colors=[]):
    ax = plt.axes(projection='3d')
    center = np.asarray([0, 0, 0])
    for i, voxel_grid in enumerate(voxel_grids):
        pts = []
        for v in voxel_grid.get_voxels():
            print(v)
            pts.append(v.grid_index)
        color = colors[i] if i < len(colors) else None
        scatter_pseq(ax, pts, c=color)
        center = np.mean(pts, axis=0)
    ax.set_xlim([center[0] - 50, center[0] + 50])
    ax.set_ylim([center[1] - 50, center[1] + 50])
    ax.set_zlim([center[2] - 50, center[2] + 50])
    plt.show()


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
                faces.extend([[v, v + section + 1, v + section + 2],
                              [v, v + section + 2, v + 1]])
    if toggledebug:
        show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
        show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj', cdprimit_type='surface_balls')


def gen_surface(pseq, rotseq, thickness, width, toggledebug=False):
    vertices = []
    faces = []

    for i, p in enumerate(pseq):
        vertices.append(p + rotseq[i][:, 0] * thickness / 2 + rotseq[i][:, 2] * width / 2)
        vertices.append(p + rotseq[i][:, 0] * thickness / 2 - rotseq[i][:, 2] * width / 2)
    for i in range(2 * len(pseq) - 2):
        f = [i, i + 1, i + 2]
        if i % 2 == 0:
            f = f[::-1]
        faces.append(f)
    if toggledebug:
        for p in pseq:
            gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, 1], radius=0.0002).attach_to(base)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj', cdprimit_type='surface_balls')


def gen_swap(pseq, rotseq, cross_sec, toggledebug=False):
    vertices = []
    faces = []
    cross_sec.append(cross_sec[0])
    for i, p in enumerate(pseq):
        for n in cross_sec:
            vertices.append(p + rotseq[i][:, 0] * n[0] + rotseq[i][:, 2] * n[1])
    for i in range(len(cross_sec) - 3):
        faces.append([0, i + 1, i + 2])
    for i in range((len(cross_sec)) * (len(pseq) - 1)):
        if i % (len(cross_sec)) == 0:
            for v in range(i, i + len(cross_sec) - 1):
                faces.extend([[v, v + len(cross_sec), v + len(cross_sec) + 1],
                              [v, v + len(cross_sec) + 1, v + 1]])
    for i in range(len(cross_sec) - 3):
        faces.append([len(vertices) - 1, len(vertices) - 2 - i, len(vertices) - 3 - i])
    if toggledebug:
        show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
        show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
        tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
        tmp_cm.set_rgba((.7, .7, 0, .7))
        tmp_cm.attach_to(base)
    objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj', cdprimit_type='surface_balls')


def pseq2bendset(pseq, bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L, toggledebug=False):
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    tangent_pts = []
    bendseq = []
    n_seq = []
    rot_a = 0
    lift_a = 0
    pos = 0
    l_pos = 0
    for i in range(1, len(pseq) - 1):
        v1 = pseq[i - 1] - pseq[i]
        v2 = pseq[i] - pseq[i + 1]
        pos += np.linalg.norm(v1)
        n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
        bend_a = rm.angle_between_vectors(v1, v2)
        if round(bend_a, 8) == 0:
            continue

        if len(n_seq) != 0:
            a = rm.angle_between_vectors(n_seq[-1], n)
            tmp_a = rm.angle_between_vectors(v1, np.cross(n_seq[-1], n))
            if tmp_a is not None and tmp_a > np.pi / 2:
                rot_a += a
            else:
                rot_a -= a

        n_seq.append(n)
        l = (bend_r * np.tan(abs(bend_a) / 2)) / np.cos(abs(lift_a))
        ratio_1 = l / np.linalg.norm(pseq[i] - pseq[i - 1])
        ratio_2 = l / np.linalg.norm(pseq[i] - pseq[i + 1])
        p_tan1 = pseq[i] + (pseq[i - 1] - pseq[i]) * ratio_1
        p_tan2 = pseq[i] + (pseq[i + 1] - pseq[i]) * ratio_2

        if i > 1 and is_collinearity(p_tan1, [pseq[i - 1], tangent_pts[-1]]):
            scatter_pseq(ax, [p_tan1], s=20, c='gray')
            print("merge", i)
            bendseq[-1][0] += bend_a
            bendseq[-1][2] = rot_a

        else:
            if i == 1:
                l_pos += np.linalg.norm(p_tan1 - pseq[i - 1])
            else:
                l_pos += np.linalg.norm(p_tan1 - tangent_pts[-1])
                l_pos += abs(bendseq[-1][0]) * bend_r / np.cos(abs(bendseq[-1][1]))
            bendseq.append([bend_a, lift_a, rot_a, l_pos + init_l])
        tangent_pts.extend([p_tan1, p_tan2])

        x = np.cross(v1, n)
        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
        if toggledebug:
            plot_frame(ax, pseq[i - 1], rot)
    if toggledebug:
        center = np.mean(pseq, axis=0)
        ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
        ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
        ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
        # goal_pseq = pickle.load(open('../run_plan/randomc.pkl', 'rb'))
        plot_pseq(ax, pseq)
        # plot_pseq(ax, goal_pseq)
        scatter_pseq(ax, [pseq[0]], s=10, c='y')
        scatter_pseq(ax, pseq[1:], s=10, c='g')
        scatter_pseq(ax, tangent_pts, s=10, c='r')
        plt.show()

    return bendseq


# def pseq2bendset(pseq, bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L, toggledebug=False):
#     ax = plt.axes(projection='3d')
#     ax.set_box_aspect((1, 1, 1))
#     tangent_pts = []
#     bendseq = []
#     n_seq = []
#     rot_a = 0
#     lift_a = 0
#     pos = 0
#     l_pos = 0
#     for i in range(1, len(pseq) - 1):
#         v1 = pseq[i - 1] - pseq[i]
#         v2 = pseq[i] - pseq[i + 1]
#         pos += np.linalg.norm(v1)
#         n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
#         bend_a = rm.angle_between_vectors(v1, v2)
#         if round(bend_a, 8) == 0:
#             continue
#
#         if len(n_seq) != 0:
#             a = rm.angle_between_vectors(n_seq[-1], n)
#             tmp_a = rm.angle_between_vectors(v1, np.cross(n_seq[-1], n))
#             if tmp_a is not None and tmp_a > np.pi / 2:
#                 rot_a += a
#             else:
#                 rot_a -= a
#
#         n_seq.append(n)
#         l = (bend_r * np.tan(abs(bend_a) / 2)) / np.cos(abs(lift_a))
#         ratio_1 = l / np.linalg.norm(pseq[i] - pseq[i - 1])
#         ratio_2 = l / np.linalg.norm(pseq[i] - pseq[i + 1])
#         p_tan1 = pseq[i] + (pseq[i - 1] - pseq[i]) * ratio_1
#         p_tan2 = pseq[i] + (pseq[i + 1] - pseq[i]) * ratio_2
#
#         if i > 1 and is_collinearity(p_tan1, [pseq[i - 1], tangent_pts[-1]]):
#             scatter_pseq(ax, [p_tan1], s=20, c='gray')
#             bendseq[-1][0] += bend_a
#             # bendseq[-1][2] = rot_a
#
#         else:
#             if i == 1:
#                 l_pos += np.linalg.norm(p_tan1 - pseq[i - 1])
#             else:
#                 l_pos += np.linalg.norm(p_tan1 - tangent_pts[-1])
#                 l_pos += abs(bendseq[-1][0]) * bend_r / np.cos(abs(bendseq[-1][1]))
#             bendseq.append([bend_a, lift_a, rot_a, l_pos + init_l])
#         tangent_pts.extend([p_tan1, p_tan2])
#
#         x = np.cross(v1, n)
#         rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
#         if toggledebug:
#             plot_frame(ax, pseq[i - 1], rot)
#     if toggledebug:
#         center = np.mean(pseq, axis=0)
#         ax.set_xlim([center[0] - 0.05, center[0] + 0.05])
#         ax.set_ylim([center[1] - 0.05, center[1] + 0.05])
#         ax.set_zlim([center[2] - 0.05, center[2] + 0.05])
#         # goal_pseq = pickle.load(open('../run_plan/randomc.pkl', 'rb'))
#         plot_pseq(ax, pseq)
#         # plot_pseq(ax, goal_pseq)
#         scatter_pseq(ax, [pseq[0]], s=10, c='y')
#         scatter_pseq(ax, pseq[1:], s=10, c='g')
#         scatter_pseq(ax, tangent_pts, s=10, c='r')
#         plt.show()
#
#     return bendseq

def rotpseq2bendset(pseq, rotseq, bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L, toggledebug=False):
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((1, 1, 1))
    tangent_pts = []
    bendseq = []
    rot_a = 0
    pos = 0
    l_pos = 0
    n = rotseq[0][:, 2]
    for i in range(1, len(pseq) - 1):
        v1 = pseq[i - 1] - pseq[i]
        v2 = pseq[i] - pseq[i + 1]
        pos += np.linalg.norm(v1)
        x1 = rotseq[i - 1][:, 0]
        x2 = rotseq[i][:, 0]
        n1 = rotseq[i - 1][:, 2]
        n2 = rotseq[i][:, 2]
        x1_xy = x1 - x1 * n
        x2_xy = x2 - x2 * n
        v1_xy = v1 - v1 * n
        v2_xy = v2 - v2 * n
        v1_yz = v1 - v1 * x1
        v2_yz = v2 - v2 * x1
        x2_yz = x2 - x2 * x1
        # bend_a = rm.angle_between_vectors(x1_xy, x2_xy)
        bend_a = rm.angle_between_vectors(v1_xy, v2_xy)
        # bend_a = rm.angle_between_vectors(v1, v2)
        if round(bend_a, 8) == 0:
            continue
        if np.cross(v1_xy, v2_xy)[2] < 0:
            rot_a = np.pi
        else:
            rot_a = 0

        lift_a = rm.angle_between_vectors(v1_yz, v2_yz)
        # lift_a = 0

        l = (bend_r * np.tan(abs(bend_a) / 2)) / np.cos(abs(lift_a))
        ratio_1 = l / np.linalg.norm(pseq[i] - pseq[i - 1])
        ratio_2 = l / np.linalg.norm(pseq[i] - pseq[i + 1])
        p_tan1 = pseq[i] + (pseq[i - 1] - pseq[i]) * ratio_1
        p_tan2 = pseq[i] + (pseq[i + 1] - pseq[i]) * ratio_2

        if i > 1 and is_collinearity(p_tan1, [pseq[i - 1], tangent_pts[-1]]):
            scatter_pseq(ax, [p_tan1], s=20, c='gray')
            bendseq[-1][0] += bend_a

        else:
            if i == 1:
                l_pos += np.linalg.norm(p_tan1 - pseq[i - 1])
            else:
                l_pos += np.linalg.norm(p_tan1 - tangent_pts[-1])
                l_pos += abs(bendseq[-1][0]) * bend_r / np.cos(abs(bendseq[-1][1]))
            bendseq.append([bend_a, 4 * lift_a, rot_a, l_pos + init_l])
        tangent_pts.extend([p_tan1, p_tan2])

    if toggledebug:
        # for i in range(len(pseq)):
        #     plot_frame(ax, pseq[i], rotseq[i])
        center = np.mean(pseq, axis=0)
        ax.set_xlim([center[0] - 0.02, center[0] + 0.02])
        ax.set_ylim([center[1] - 0.02, center[1] + 0.02])
        ax.set_zlim([center[2] - 0.02, center[2] + 0.02])
        plot_pseq(ax, pseq)
        scatter_pseq(ax, [pseq[0]], s=10, c='y')
        scatter_pseq(ax, pseq[1:], s=10, c='g')
        scatter_pseq(ax, tangent_pts, s=10, c='r')
        plt.show()

    return bendseq


def cal_curvature(pseq, show=False):
    def _center(A, B, C):
        (x1, y1), (x2, y2), (x3, y3) = A, B, C
        a = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
        b = (x1 ** 2 + y1 ** 2) * (y3 - y2) + (x2 ** 2 + y2 ** 2) * (y1 - y3) + (x3 ** 2 + y3 ** 2) * (y2 - y1)
        c = (x1 ** 2 + y1 ** 2) * (x2 - x3) + (x2 ** 2 + y2 ** 2) * (x3 - x1) + (x3 ** 2 + y3 ** 2) * (x1 - x2)
        center = np.asarray([-b / a / 2, -c / a / 2])
        r = np.linalg.norm(np.asarray(A) - center)
        return center, r

    curture_list = []
    torsion_list = []
    r_list = []
    nrml_pre = None
    # dir_pre = np.asarray([1, 0, 0])
    for i in range(1, len(pseq) - 1):
        nrml = np.cross(pseq[i + 1] - pseq[i], pseq[i] - pseq[i - 1])
        rot = rm.rotmat_between_vectors(nrml, np.asarray([0, 0, 1]))
        A = rot.dot(pseq[i - 1])
        B = rot.dot(pseq[i])
        C = rot.dot(pseq[i + 1])
        center, r = _center(A[:2], B[:2], C[:2])
        # circle = np.asarray([np.linalg.inv(rot).dot(np.asarray([center[0] + r * np.cos(a),
        #                                                         center[1] + r * np.sin(a), A[2]]))
        #                      for a in np.linspace(-np.pi, np.pi, 60)])
        # plt.plot(circle[:, 0], circle[:, 1], circle[:, 2], color='r')
        # print(r, 1 / r)
        curture_list.append(1 / r)
        r_list.append(r)
        if nrml_pre is None:
            nrml_pre = nrml
            continue
        if rm.angle_between_vectors(np.asarray(pseq[-1] - pseq[0]), np.cross(nrml, nrml_pre)) > np.pi / 2:
            torsion_list.append(-rm.angle_between_vectors(nrml, nrml_pre))
        else:
            torsion_list.append(rm.angle_between_vectors(nrml, nrml_pre))
        # dir_pre = np.cross(nrml, nrml_pre)
        nrml_pre = nrml
    if show:
        plt.plot(range(len(pseq))[1:-1], curture_list)
        plt.show()
    return curture_list, r_list, [sum(torsion_list[:i]) for i in range(len(torsion_list))]


'''
seq planning
'''


def pnp_cnt(l):
    def _intersec(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    cnt = 0
    for i in range(len(l) - 1):
        if l[i + 1] < l[i]:
            cnt += 1
        elif len(_intersec(l[:i], range(l[i], l[i + 1]))) > 0:
            cnt += 1
    return cnt


def unstable_cnt(l):
    cnt = 0
    for i in range(1, len(l)):
        if l[i] < max(l[:i]):
            cnt += 1
    return cnt


def rank_combs(combs):
    pnp_cnt_list = []
    unstable_cnt_list = []
    for l in combs:
        pnp_cnt_list.append(pnp_cnt(l))
        unstable_cnt_list.append(unstable_cnt(l))
    _, _, combs = zip(*sorted(zip(pnp_cnt_list, unstable_cnt_list, combs)))
    # for l in combs:
    #     print(l, pnp_cnt(l), unstable_cnt(l))
    return list(combs)


def remove_combs(rmv_l, combs):
    new_combs = []
    for i, comb in enumerate(combs):
        if set(comb[:len(rmv_l) - 1]) == set(rmv_l[:len(rmv_l) - 1]) and comb[len(rmv_l) - 1] == rmv_l[-1]:
            continue
        new_combs.append(comb)
    return list(new_combs)
