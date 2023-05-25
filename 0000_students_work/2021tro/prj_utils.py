import copy
import math
import random
import time
import pickle
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from geomdl import fitting
# from geomdl.visualization import VisMPL as vis
import modeling.collision_model as cm

from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import KDTree

import math_utils as mu
import utils.pcd_utils as pcdu
import utils.run_utils as ru
import utiltools.robotmath as rm

SNAP_QI = False

def resize_drawpath(drawpath, w, h, space=5):
    """

    :param drawpath: draw path point list
    :param w:
    :param h:
    :param space: space between drawing and rectangle edge
    :return:
    """

    def __sort_w_h(a, b):
        if a > b:
            return a, b
        else:
            return b, a

    drawpath = remove_list_dup(drawpath)
    p_narray = np.array(drawpath)
    pl_w = max(p_narray[:, 0]) - min(p_narray[:, 0])
    pl_h = max(p_narray[:, 1]) - min(p_narray[:, 1])
    pl_w, pl_h = __sort_w_h(pl_w, pl_h)
    w, h = __sort_w_h(w, h)

    # if pl_w / w > 1 and pl_h / h > 1:
    scale = max([pl_w / (w - space), pl_h / (h - space)])
    p_narray = p_narray / scale

    return list(p_narray)




def rayhitmesh_p(obj, center, p, direction=np.asarray((0, 0, 1))):
    if abs(direction[0]) == 1:
        pfrom = np.asarray((center[0], p[0] + center[1], p[1] + center[2])) + 50 * direction
        pto = np.asarray((center[0], p[0] + center[1], p[1] + center[2])) - 50 * direction
    elif abs(direction[1]) == 1:
        pfrom = np.asarray((p[0] + center[0], center[1], p[1] + center[2])) + 50 * direction
        pto = np.asarray((p[0] + center[0], center[1], p[1] + center[2])) - 50 * direction
    elif abs(direction[2]) == 1:
        pfrom = np.asarray((p[0] + center[0], p[1] + center[1], center[2])) + 50 * direction
        pto = np.asarray((p[0] + center[0], p[1] + center[1], center[2])) - 50 * direction
    else:
        print('Wrong input direction!')
        return None, None
    base.pggen.plotArrow(base.render, spos=pfrom, epos=pto, length=100, rgba=(0, 1, 0, 1))
    # base.run()
    pos, nrml = rayhitmesh_closest(obj, pfrom, pto)
    return pos, -nrml


def get_vecs_angle(v1, v2):
    return math.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))


def get_knn_indices(p, kdt, k=3):
    distances, indices = kdt.query([p], k=k, return_distance=True)
    return indices[0]


def get_knn(p, kdt, k=3):
    p_nearest_inx = get_knn_indices(p, kdt, k=k)
    pcd = list(np.array(kdt.data))
    return np.asarray([pcd[p_inx] for p_inx in p_nearest_inx])


def get_nn_indices_by_distance(p, kdt, step=1.0):
    result_indices = []
    distances, indices = kdt.query([p], k=1000, return_distance=True)
    distances = distances[0]
    indices = indices[0]
    for i in range(len(distances)):
        if distances[i] < step:
            result_indices.append(indices[i])
    return result_indices


def get_kdt(p_list, dimension=3):
    time_start = time.time()
    p_narray = np.array(p_list[:, :dimension])
    kdt = KDTree(p_narray, leaf_size=100, metric='euclidean')
    print('time cost(kdt):', time.time() - time_start)
    return kdt, p_narray


def get_z_by_bilinearinp(p, kdt):
    p_nearest_inx = get_knn_indices(p, kdt, k=4)
    pcd = list(np.array(kdt.data))
    p_list = [pcd[p_inx] for p_inx in p_nearest_inx]
    return mu.bilinear_interp_2d(p[:2], [(p[0], p[1]) for p in p_list], [p[2] for p in p_list])


def __get_avg_dist(p_list):
    dist_list = []
    for i in range(1, len(p_list)):
        dist = np.linalg.norm(np.array(p_list[i]) - np.array(p_list[i - 1]))
        dist_list.append(dist)
    return np.average(dist_list)


def get_intersec(p1, p2, plane_nrml, d):
    p1_d = (np.vdot(p1, plane_nrml) + d) / np.sqrt(np.vdot(plane_nrml, plane_nrml))
    p1_d2 = (np.vdot(p2 - p1, plane_nrml)) / np.sqrt(np.vdot(plane_nrml, plane_nrml))
    n = p1_d2 / p1_d
    return p1 + n * (p2 - p1)


def get_nrml_pca(knn):
    pcv, pcaxmat = rm.computepca(knn)
    return pcaxmat[:, np.argmin(pcv)]


def __find_nxt_p_pca(drawpath_p1, drawpath_p2, kdt_d3, p0, n0, max_nn=150, direction=np.array([0, 0, 1]),
                     toggledebug=False, pcd=None, snap=True):
    v_draw = np.array(drawpath_p2) - np.array(drawpath_p1)
    if abs(direction[0]) == 1:
        v_draw = (0, v_draw[0], v_draw[1])
    elif abs(direction[1]) == 1:
        v_draw = (v_draw[0], 0, v_draw[1])
    elif abs(direction[2]) == 1:
        v_draw = (v_draw[0], v_draw[1], 0)
    else:
        print('Wrong input direction!')
        return None
    rotmat = rm.rotmat_betweenvector(direction, n0)
    v_draw = np.dot(rotmat, v_draw)
    pt = p0 + v_draw

    knn = get_knn(pt, kdt_d3, k=max_nn)
    center = pcdu.get_pcd_center(np.asarray(knn))
    nrml = get_nrml_pca(knn)
    if snap:
        p_nxt = pt - np.dot((pt - center), nrml) * nrml
    else:
        p_nxt = copy.deepcopy(pt)

    if np.dot(nrml, np.asarray([0, 0, 1])) < 0:
        nrml = -nrml
    # if np.dot(nrml, np.asarray([0, -1, 0])) < 0:
    #     nrml = -nrml
    # if np.dot(nrml, np.asarray([-1, 0, 0])) < 0:
    #     nrml = -nrml
    # if np.dot(nrml, np.asarray([0, -1, 0])) == -1:
    #     nrml = -nrml

    if toggledebug:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plot_edge = 1.5
        plot_edge_z = 1.5
        knn_p0 = get_knn(p0, kdt_d3, k=max_nn)

        x_range = (min(knn[:, 0].flatten()) - plot_edge, max(knn[:, 0].flatten()) + plot_edge)
        y_range = (min(knn[:, 1].flatten()) - plot_edge, max(knn[:, 1].flatten()) + plot_edge)
        z_range = (min(knn[:, 2].flatten()) - plot_edge_z, max(knn[:, 2].flatten()) + plot_edge_z)
        ax.scatter(knn[:, 0], knn[:, 1], knn[:, 2], c='y', s=1, alpha=.1)
        coef_l = mu.fit_plane(knn)
        mu.plot_surface_f(ax, coef_l, x_range, y_range, z_range, dense=.5, c=cm.coolwarm)

        x_range = (min(knn_p0[:, 0].flatten()) - plot_edge, max(knn_p0[:, 0].flatten()) + plot_edge)
        y_range = (min(knn_p0[:, 1].flatten()) - plot_edge, max(knn_p0[:, 1].flatten()) + plot_edge)
        z_range = (min(knn_p0[:, 2].flatten()) - plot_edge_z, max(knn_p0[:, 2].flatten()) + plot_edge_z)
        ax.scatter(knn_p0[:, 0], knn_p0[:, 1], knn_p0[:, 2], c='y', s=1, alpha=.1)
        coef_l_init = mu.fit_plane(knn_p0)
        mu.plot_surface_f(ax, coef_l_init, x_range, y_range, z_range, dense=.5, c=cm.coolwarm)

        if pcd is not None:
            pcd_edge = 5
            pcd = [p for p in pcd if
                   x_range[0] - pcd_edge < p[0] < x_range[1] + pcd_edge and
                   y_range[0] - pcd_edge < p[1] < y_range[1] + pcd_edge + 5 and
                   z_range[0] - pcd_edge < p[2] < z_range[1] + pcd_edge + 5]
            pcd = np.asarray(random.choices(pcd, k=2000))
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='k', alpha=.1, s=1)

        ax.scatter([p0[0]], [p0[1]], [p0[2]], c='r', s=10, alpha=1)
        ax.scatter([p_nxt[0]], [p_nxt[1]], [p_nxt[2]], c='g', s=10, alpha=1)
        ax.scatter([pt[0]], [pt[1]], [pt[2]], c='b', s=10, alpha=1)
        # ax.annotate3D('$q_0$', pcd_start_p, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$q_1$', p_nxt, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$q_t$', pt, xytext=(3, 3), textcoords='offset points')

        # ax.arrow3D(pcd_start_p[0], pcd_start_p[1], pcd_start_p[2],
        #            pcd_start_n[0], pcd_start_n[1], pcd_start_n[2], mutation_scale=10, arrowstyle='->')
        # ax.arrow3D(pt[0], pt[1], pt[2],
        #            nrml[0], nrml[1], nrml[2], mutation_scale=10, arrowstyle='->')
        # ax.arrow3D(pcd_start_p[0], pcd_start_p[1], pcd_start_p[2],
        #            v_draw[0], v_draw[1], v_draw[2], mutation_scale=10, arrowstyle='->')
        # ax.annotate3D('$N_0$', pcd_start_p + pcd_start_n, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$N_t$', pt + nrml, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$V_{draw}$', pcd_start_p + v_draw * 0.5, xytext=(3, 3), textcoords='offset points')

        plt.show()

    return p_nxt, nrml


def __find_nxt_p_rbf(drawpath_p1, drawpath_p2, kdt_d3, p0, n0, max_nn=150, direction=np.array([0, 0, 1]),
                     toggledebug=False, step=0.1, pcd=None, pca_trans=True):
    v_draw = np.array(drawpath_p2) - np.array(drawpath_p1)
    if abs(direction[0]) == 1:
        v_draw = (0, v_draw[0], v_draw[1])
    elif abs(direction[1]) == 1:
        v_draw = (v_draw[0], 0, v_draw[1])
    elif abs(direction[2]) == 1:
        v_draw = (v_draw[0], v_draw[1], 0)
    else:
        print('Wrong input direction!')
        return None

    rotmat = rm.rotmat_betweenvector(direction, n0)
    v_draw = np.dot(rotmat, v_draw)

    knn_p0 = get_knn(p0, kdt_d3, k=max_nn)
    if pca_trans:
        knn_p0_tr, transmat = mu.trans_data_pcv(knn_p0, random_rot=False)
        surface = RBFInterpolator(knn_p0_tr[:, :2], knn_p0_tr[:, 2])
    else:
        transmat = np.eye(3)
        surface = RBFInterpolator(knn_p0[:, :2], knn_p0[:, 2])

    tgt_len = np.linalg.norm(v_draw)
    pm = np.dot(np.linalg.inv(transmat), p0)
    pt = np.dot(np.linalg.inv(transmat), p0)
    while tgt_len > 0.005:
        p_uv = (pm + np.dot(np.linalg.inv(transmat), v_draw) * step)[:2]
        z = surface([p_uv])[0]
        pt = np.asarray([p_uv[0], p_uv[1], z])
        tgt_len -= np.linalg.norm(pt - pm)
        pm = pt

    p_nxt = np.dot(transmat, pt)
    knn = get_knn(p_nxt, kdt_d3, k=max_nn)
    nrml = get_nrml_pca(knn)

    if np.dot(nrml, np.asarray([0, 0, 1])) < 0:
        nrml = -nrml
    # if np.dot(nrml, np.asarray([0, -1, 0])) < 0:
    #     nrml = -nrml
    # if np.dot(nrml, np.asarray([-1, 0, 0])) < 0:
    #     nrml = -nrml
    if np.dot(nrml, np.asarray([0, -1, 0])) == -1:
        nrml = -nrml
    if toggledebug:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        pcd_edge = 10
        plot_edge = 5
        plot_edge_z = 1.5
        x_range = (min(knn_p0[:, 0].flatten()) - plot_edge, max(knn_p0[:, 0].flatten()) + plot_edge)
        y_range = (min(knn_p0[:, 1].flatten()) - plot_edge, max(knn_p0[:, 1].flatten()) + plot_edge)
        z_range = (min(knn_p0[:, 2].flatten()) - plot_edge_z, max(knn_p0[:, 2].flatten()) + plot_edge_z)

        xgrid = np.mgrid[x_range[0]:x_range[1], y_range[0]:y_range[1]]
        xflat = xgrid.reshape(2, -1).T
        zflat = surface(xflat)
        inp_pts = np.column_stack((xflat, zflat))
        inp_pts = np.dot(transmat, inp_pts.T).T

        Z = inp_pts[:, 2].reshape((xgrid.shape[1], xgrid.shape[2]))
        ax.plot_surface(xgrid[0], xgrid[1], Z, rstride=1, cstride=1, alpha=.5, cmap='coolwarm')
        # ax.scatter(inp_pts[:, 0], inp_pts[:, 1], inp_pts[:, 2], c='r', alpha=1, s=1)
        pcd = [p for p in pcd if
               x_range[0] - pcd_edge < p[0] < x_range[1] + pcd_edge and
               y_range[0] - pcd_edge < p[1] < y_range[1] + pcd_edge + 5 and
               z_range[0] - pcd_edge < p[2] < z_range[1] + pcd_edge + 5]
        pcd = np.asarray(random.choices(pcd, k=2000))
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='k', alpha=.1, s=1)
        ax.scatter([p0[0]], [p0[1]], [p0[2]], c='r', s=10, alpha=1)
        ax.scatter([p_nxt[0]], [p_nxt[1]], [p_nxt[2]], c='g', s=10, alpha=1)
        plt.show()

    return p_nxt, nrml


def __find_nxt_p_rbf_g(drawpath_p1, drawpath_p2, surface, transmat, kdt_d3, p0, n0, max_nn=150,
                       direction=np.array([0, 0, 1]), toggledebug=False, step=0.1, pcd=None):
    v_draw = np.array(drawpath_p2) - np.array(drawpath_p1)
    if abs(direction[0]) == 1:
        v_draw = (0, v_draw[0], v_draw[1])
    elif abs(direction[1]) == 1:
        v_draw = (v_draw[0], 0, v_draw[1])
    elif abs(direction[2]) == 1:
        v_draw = (v_draw[0], v_draw[1], 0)
    else:
        print('Wrong input direction!')
        return None

    # rotmat = rm.rotmat_betweenvector(direction, n0)
    # v_draw = np.dot(rotmat, v_draw)

    tgt_len = np.linalg.norm(v_draw)
    pm = np.dot(np.linalg.inv(transmat), p0)
    pt = np.dot(np.linalg.inv(transmat), p0)
    while tgt_len > 0.005:
        p_uv = (pm + np.dot(np.linalg.inv(transmat), v_draw) * step)[:2]
        z = surface([p_uv])[0]
        pt = np.asarray([p_uv[0], p_uv[1], z])
        tgt_len -= np.linalg.norm(pt - pm)
        pm = pt

    p_nxt = np.dot(transmat, pt)
    knn = get_knn(p_nxt, kdt_d3, k=max_nn)
    nrml = get_nrml_pca(knn)

    if np.dot(nrml, np.asarray([0, 0, 1])) < 0:
        nrml = -nrml
    # if np.dot(nrml, np.asarray([0, -1, 0])) < 0:
    #     nrml = -nrml
    # if np.dot(nrml, np.asarray([-1, 0, 0])) < 0:
    #     nrml = -nrml
    # if np.dot(nrml, np.asarray([0, -1, 0])) == -1:
    #     nrml = -nrml
    if toggledebug:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        pcd_edge = 10
        plot_edge = 5
        plot_edge_z = 1.5
        x_range = (p0[0] - plot_edge, p0[0] + plot_edge)
        y_range = (p0[1] - plot_edge, p0[1] + plot_edge)
        z_range = (p0[2] - plot_edge_z, p0[2] + plot_edge_z)

        xgrid = np.mgrid[x_range[0]:x_range[1], y_range[0]:y_range[1]]
        xflat = xgrid.reshape(2, -1).T
        zflat = surface(xflat)
        inp_pts = np.column_stack((xflat, zflat))
        inp_pts = np.dot(transmat, inp_pts.T).T

        Z = inp_pts[:, 2].reshape((xgrid.shape[1], xgrid.shape[2]))
        ax.plot_surface(xgrid[0], xgrid[1], Z, rstride=1, cstride=1, alpha=.5, cmap='coolwarm')
        # ax.scatter(inp_pts[:, 0], inp_pts[:, 1], inp_pts[:, 2], c='r', alpha=1, s=1)
        pcd = [p for p in pcd if
               x_range[0] - pcd_edge < p[0] < x_range[1] + pcd_edge and
               y_range[0] - pcd_edge < p[1] < y_range[1] + pcd_edge + 5 and
               z_range[0] - pcd_edge < p[2] < z_range[1] + pcd_edge + 5]
        pcd = np.asarray(random.choices(pcd, k=2000))
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='k', alpha=.1, s=1)
        ax.scatter([p0[0]], [p0[1]], [p0[2]], c='r', s=10, alpha=1)
        ax.scatter([p_nxt[0]], [p_nxt[1]], [p_nxt[2]], c='g', s=10, alpha=1)
        plt.show()

    return p_nxt, nrml


def __find_nxt_p(drawpath_p1, drawpath_p2, kdt_d3, pcd_start_p, pcd_start_n, direction=np.array([0, 0, 1])):
    v_draw = np.array(drawpath_p2) - np.array(drawpath_p1)
    if abs(direction[0]) == 1:
        v_draw = (0, v_draw[0], v_draw[1])
    elif abs(direction[1]) == 1:
        v_draw = (v_draw[0], 0, v_draw[1])
    elif abs(direction[2]) == 1:
        v_draw = (v_draw[0], v_draw[1], 0)
    else:
        print('Wrong input direction!')
        return None
    rotmat = rm.rotmat_betweenvector(direction, pcd_start_n)
    pt = pcd_start_p + np.dot(rotmat, v_draw)
    p_start_inx = get_knn_indices(pcd_start_p, kdt_d3, k=1)[0]
    p_nxt_inx = get_knn_indices(pt, kdt_d3, k=1)[0]
    if p_start_inx == p_nxt_inx:
        p_nxt_inx = get_knn_indices(pt, kdt_d3, k=2)[1]

    return p_nxt_inx


def __find_nxt_p_intg(drawpath_p1, drawpath_p2, kdt_d3, p0, n0, direction=np.array([0, 0, 1]), max_nn=150, snap=False,
                      toggledebug=False, pcd=None):
    v_draw = np.array(drawpath_p2) - np.array(drawpath_p1)
    v_draw_len = np.linalg.norm(v_draw)
    if abs(direction[0]) == 1:
        v_draw = (0, v_draw[0], v_draw[1])
    elif abs(direction[1]) == 1:
        v_draw = (v_draw[0], 0, v_draw[1])
    elif abs(direction[2]) == 1:
        v_draw = (v_draw[0], v_draw[1], 0)
    else:
        print('Wrong input direction!')
        return None
    rotmat = rm.rotmat_betweenvector(direction, np.asarray(n0))
    v_draw = np.dot(rotmat, v_draw)

    knn_p0 = get_knn(p0, kdt_d3, k=max_nn)
    knn_p0_tr, transmat = mu.trans_data_pcv(knn_p0)
    f_q = mu.fit_qua_surface(knn_p0_tr)

    v_draw_tr = np.dot(transmat.T, v_draw)
    f_l_base = mu.get_plane(n0, v_draw, p0)
    f_l = mu.get_plane(n0, v_draw, p0, transmat=transmat.T)

    # print('v_draw', v_draw, v_draw_tr)
    itg_axis = list(abs(v_draw_tr)).index(max(abs(v_draw_tr[:2])))

    if v_draw_tr[itg_axis] > 0:
        pt, _, F, G, x_y = mu.cal_surface_intersc(f_q, f_l, np.dot(transmat.T, p0), tgtlen=v_draw_len, mode='ub',
                                                  itg_axis=['x', 'y', 'z'][itg_axis], toggledebug=False)
    else:
        pt, _, F, G, x_y = mu.cal_surface_intersc(f_q, f_l, np.dot(transmat.T, p0), tgtlen=v_draw_len, mode='lb',
                                                  itg_axis=['x', 'y', 'z'][itg_axis], toggledebug=False)
    pt = np.dot(transmat, pt)
    # print('next p(2D):', drawpath_p2)
    # print('next p(3D):', pt)

    if snap:
        knn_pt = get_knn(pt, kdt_d3, k=max_nn)
        center = pcdu.get_pcd_center(np.asarray(knn_pt))
        nrml = get_nrml_pca(knn_pt)
        p_nxt = pt - np.dot((pt - center), nrml) * nrml
    else:
        p_nxt = pt

    if toggledebug:
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        plot_edge = 9
        plot_edge_z = 5

        x_range = (min(knn_p0[:, 0].flatten()) - plot_edge, max(knn_p0[:, 0].flatten()) + plot_edge)
        y_range = (min(knn_p0[:, 1].flatten()) - plot_edge, max(knn_p0[:, 1].flatten()) + plot_edge)
        z_range = (min(knn_p0[:, 2].flatten()) - plot_edge_z, max(knn_p0[:, 2].flatten()) + plot_edge_z)

        # if pcd is not None:
        #     pcd_edge = 6.5
        #     pcd = [p for p in pcd if
        #            x_range[0] - pcd_edge < p[0] < x_range[1] + pcd_edge and
        #            y_range[0] - pcd_edge+5 < p[1] < y_range[1] + pcd_edge and
        #            z_range[0] - pcd_edge < p[2] < z_range[1] + pcd_edge]
        #     pcd = np.asarray(random.choices(pcd, k=2000))
        #     ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='k', alpha=.1, s=1)

        ax.scatter(knn_p0[:, 0], knn_p0[:, 1], knn_p0[:, 2], c='y', s=1, alpha=.5)
        f_q_base = mu.fit_qua_surface(knn_p0)

        mu.plot_surface_f(ax, f_q_base, x_range, y_range, z_range, dense=.5, c=cm.coolwarm)
        mu.plot_surface_f(ax, f_l_base, x_range, y_range, z_range, dense=.5, axis='x', alpha=.2)
        if snap:
            ax.scatter(knn_pt[:, 0], knn_pt[:, 1], knn_pt[:, 2], c='k', s=5, alpha=.1)
            f_q_qt = mu.fit_qua_surface(knn_pt)
            mu.plot_surface_f(ax, f_q_qt, x_range, y_range, z_range, dense=.5)
            ax.scatter([pt[0]], [pt[1]], [pt[2]], c='g', s=10, alpha=1)
            ax.annotate3D('$q_t$', pt, xytext=(3, 3), textcoords='offset points')

        ax.scatter([p0[0]], [p0[1]], [p0[2]], c='r', s=10, alpha=1)
        ax.scatter([p_nxt[0]], [p_nxt[1]], [p_nxt[2]], c='b', s=10, alpha=1)
        # ax.annotate3D('$q_0$', p0, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$q_1$', p_nxt, xytext=(3, 3), textcoords='offset points')
        # ax.arrow3D(p0[0], p0[1], p0[2], n0[0], n0[1], n0[2], mutation_scale=10, arrowstyle='->', color='b')
        # ax.arrow3D(p0[0], p0[1], p0[2], v_draw[0], v_draw[1], v_draw[2], mutation_scale=10, arrowstyle='->')
        # ax.annotate3D('$N_0$', p0 + n0, xytext=(3, 3), textcoords='offset points')
        # ax.annotate3D('$V_{draw}$', p0 + v_draw * 0.5, xytext=(3, 3), textcoords='offset pixels')

        ax_tr = fig.add_subplot(1, 2, 2, projection='3d')
        knn_p0_tr = np.dot(transmat.T, knn_p0.T).T
        x_range = (min(knn_p0_tr[:, 0].flatten()) - plot_edge, max(knn_p0_tr[:, 0].flatten()) + plot_edge)
        y_range = (min(knn_p0_tr[:, 1].flatten()) - plot_edge, max(knn_p0_tr[:, 1].flatten()) + plot_edge)
        z_range = (min(knn_p0_tr[:, 2].flatten()) - plot_edge_z, max(knn_p0_tr[:, 2].flatten()) + plot_edge_z)
        ax_tr.scatter(knn_p0_tr[:, 0], knn_p0_tr[:, 1], knn_p0_tr[:, 2], c='k', s=5, alpha=.1)

        mu.plot_surface_f(ax_tr, f_q, x_range, y_range, z_range, dense=.5, c=cm.coolwarm)
        mu.plot_surface_f(ax_tr, f_l, x_range, y_range, z_range, dense=.5, axis='y', alpha=.2)

        p0_tr = np.dot(transmat.T, p0)
        p_nxt_tr = np.dot(transmat.T, p_nxt)
        n0_tr = np.dot(transmat.T, n0)

        ax_tr.scatter([p0_tr[0]], [p0_tr[1]], [p0_tr[2]], c='r', s=10, alpha=1)
        ax_tr.scatter([p_nxt_tr[0]], [p_nxt_tr[1]], [p_nxt_tr[2]], c='b', s=10, alpha=1)
        # ax_tr.annotate3D('$q_0$', p0_tr, xytext=(3, 3), textcoords='offset points')
        # ax_tr.annotate3D('$q_1$', p_nxt_tr, xytext=(3, 3), textcoords='offset points')
        # ax_tr.arrow3D(p0_tr[0], p0_tr[1], p0_tr[2], n0_tr[0], n0_tr[1], n0_tr[2],
        #               mutation_scale=10, arrowstyle='->')
        # ax_tr.arrow3D(p0_tr[0], p0_tr[1], p0_tr[2], v_draw_tr[0], v_draw_tr[1], v_draw_tr[2],
        #               mutation_scale=10, arrowstyle='->')
        # ax_tr.annotate3D('$N_0$', p0_tr + n0_tr, xytext=(3, 3), textcoords='offset points')
        # ax_tr.annotate3D('$V_{draw}$', p0_tr + v_draw_tr * 0.5, xytext=(3, 3), textcoords='offset points')
        mu.plot_intersc(ax_tr, x_y, F, p0_tr, alpha=.5, c='k', plot_edge=10)
        mu.plot_intersc(ax, x_y, F, p0_tr, transmat=transmat, alpha=.5, c='k', plot_edge=9)
        plt.show()

    knn_p_nxt = get_knn(p_nxt, kdt_d3, k=max_nn)
    p_nxt_nrml = get_nrml_pca(knn_p_nxt)
    angle = rm.angle_between_vectors(p_nxt_nrml, n0)
    if abs(angle) > np.pi / 2:
        p_nxt_nrml = -p_nxt_nrml
    # print(angle, np.asarray(pt), np.asarray(p_nxt_nrml))
    return np.asarray(p_nxt), np.asarray(p_nxt_nrml)


def __prj_stroke(stroke, drawcenter, pcd, pcd_nrmls, kdt_d3, mode='DI', objcm=None, pcd_start_p=None, error_method='ED',
                 pcd_start_n=None, surface=None, transmat=None, direction=np.asarray((0, 0, 1)), toggledebug=False):
    """

    :param stroke:
    :param drawcenter:
    :param pcd:
    :param pcd_nrmls:
    :param kdt_d3:
    :param mode: 'DI', 'EI','QI','rbf','rbf-g'
    :param objcm:
    :param pcd_start_p:
    :param pcd_start_n:
    :param direction:
    :return:
    """

    time_start = time.time()
    if pcd_start_p is None:
        if objcm is None:
            inx = get_knn_indices(drawcenter, kdt_d3)[0]
            pcd_start_p = pcd[inx]
            pcd_start_n = pcd_nrmls[inx]
            print('pcd_start_p:', pcd_start_p, 'pcd_start_n:', pcd_start_n)
        else:
            pcd_start_p, pcd_start_n = rayhitmesh_p(objcm, drawcenter, stroke[0], direction=direction)
    # base.pggen.plotSphere(base.render, pcd_start_p, radius=2, rgba=(0, 0, 1, 1))
    pos_nrml_list = [[pcd_start_p, pcd_start_n]]

    for i in range(len(stroke) - 1):
        p1, p2 = stroke[i], stroke[i + 1]
        if mode == 'EI':
            p_nxt, nrml = __find_nxt_p_pca(p1, p2, kdt_d3, pos_nrml_list[-1][0], pos_nrml_list[-1][1],
                                           direction=direction, toggledebug=toggledebug, pcd=pcd)
            pos_nrml_list.append([p_nxt, nrml])
        elif mode == 'DI':
            p_nxt_inx = __find_nxt_p(p1, p2, kdt_d3, pos_nrml_list[-1][0], pos_nrml_list[-1][1], direction=direction)
            pos_nrml_list.append([pcd[p_nxt_inx], pcd_nrmls[p_nxt_inx]])
        elif mode == 'QI':
            p_nxt, p_nxt_nrml = __find_nxt_p_intg(p1, p2, kdt_d3, pos_nrml_list[-1][0], pos_nrml_list[-1][1],
                                                  snap=SNAP_QI, direction=direction, toggledebug=toggledebug, pcd=pcd)
            pos_nrml_list.append([p_nxt, p_nxt_nrml])
            # base.pggen.plotSphere(base.render, pos=p_nxt, rgba=(1, 0, 0, 1))
            # base.pggen.plotArrow(base.render, spos=p_nxt, epos=p_nxt + p_nxt_nrml * 10, rgba=(1, 0, 0, 1))
        elif mode == 'rbf':
            p_nxt, p_nxt_nrml = __find_nxt_p_rbf(p1, p2, kdt_d3, pos_nrml_list[-1][0], pos_nrml_list[-1][1],
                                                 direction=direction, toggledebug=toggledebug, pcd=pcd, step=.01)
            pos_nrml_list.append([p_nxt, p_nxt_nrml])
        elif mode == 'rbf_g':
            p_nxt, p_nxt_nrml = __find_nxt_p_rbf_g(p1, p2, surface, transmat, kdt_d3, pos_nrml_list[-1][0],
                                                   pos_nrml_list[-1][1], direction=direction, toggledebug=toggledebug,
                                                   pcd=pcd, step=.01)
            pos_nrml_list.append([p_nxt, p_nxt_nrml])

    time_cost = time.time() - time_start
    if error_method == 'GD':
        error, error_list = get_prj_error(stroke, pos_nrml_list, method=error_method, kdt_d3=kdt_d3)
    else:
        error, error_list = get_prj_error(stroke, pos_nrml_list)
    print(f'stroke error: {error}')

    return pos_nrml_list, error, error_list, time_cost


def get_prj_error(drawpath, pos_nrml_list, method='ED', kdt_d3=None, pcd=None, max_nn=150):
    """

    :param drawpath:
    :param pos_nrml_list:
    :param method: 'ED', 'GD'
    :return:
    """
    error_list = []
    prj_len_list = []
    real_len_list = []

    if method == 'ED':
        for i in range(1, len(drawpath)):
            try:
                real_len = np.linalg.norm(np.array(drawpath[i]) - np.array(drawpath[i - 1]))
                prj_len = np.linalg.norm(np.array(pos_nrml_list[i][0]) - np.array(pos_nrml_list[i - 1][0]))
                error_list.append(round((prj_len - real_len) / real_len, 5))
                prj_len_list.append(prj_len)
                real_len_list.append(real_len)
                # print('project ED:', real_len, prj_len)
            except:
                error_list.append(None)

    elif method == 'GD':
        for i in range(1, len(drawpath)):
            try:
                real_len = np.linalg.norm(np.array(drawpath[i]) - np.array(drawpath[i - 1]))
                p1 = np.asarray(pos_nrml_list[i - 1][0])
                p2 = np.asarray(pos_nrml_list[i][0])
                v_draw = p2 - p1
                knn = get_knn(p1, kdt_d3, k=max_nn)
                knn_tr, transmat = mu.trans_data_pcv(knn)
                f_q = mu.fit_qua_surface(knn_tr)
                f_l = mu.get_plane(pos_nrml_list[i][1], v_draw, p1, transmat=transmat.T)
                v_draw_tr = np.dot(transmat.T, v_draw)
                p1_tr = np.dot(transmat.T, p1)
                p2_tr = np.dot(transmat.T, p2)
                if abs(v_draw_tr[0]) > abs(v_draw_tr[1]):
                    prj_len = mu.cal_surface_intersc_p2p(f_q, f_l, p1_tr, p2_tr, itg_axis='x')
                else:
                    prj_len = mu.cal_surface_intersc_p2p(f_q, f_l, p1_tr, p2_tr, itg_axis='y')
                error_list.append(round((prj_len - real_len) / real_len, 5))
                prj_len_list.append(prj_len)
                real_len_list.append(real_len)
                # print('project GD:', real_len, prj_len)
            except:
                error_list.append(None)

    elif method == 'rbf':
        for i in range(1, len(drawpath)):
            try:
                real_len = np.linalg.norm(np.array(drawpath[i]) - np.array(drawpath[i - 1]))
                p1 = np.asarray(pos_nrml_list[i - 1][0])
                p2 = np.asarray(pos_nrml_list[i][0])
                v_draw = p2 - p1
                knn = get_knn(p1, kdt_d3, k=max_nn)
                knn_tr, transmat = mu.trans_data_pcv(knn, random_rot=False)
                surface = RBFInterpolator(knn_tr[:, :2], knn_tr[:, 2])
                pm = np.dot(np.linalg.inv(transmat), p1)
                step = .05
                iter_times = 1 / step
                prj_len = 0

                while iter_times > 0:
                    p_uv = (pm + np.dot(np.linalg.inv(transmat), v_draw) * step)[:2]
                    z = surface([p_uv])[0]
                    pt = np.asarray([p_uv[0], p_uv[1], z])
                    prj_len += np.linalg.norm(pt - pm)
                    pm = pt
                    iter_times -= 1

                error_list.append(round((prj_len - real_len) / real_len, 5))
                prj_len_list.append(prj_len)
                real_len_list.append(real_len)
            except:
                error_list.append(None)

    elif method == 'rbf-g':
        print('pcd length', len(pcd))
        pcd = np.asarray(random.choices(pcd, k=5000))
        pcd_tr, transmat = mu.trans_data_pcv(pcd, random_rot=False)
        # base.pggen.plotAxis(base.render)
        # pcdu.show_pcd(pcd)
        # pcdu.show_pcd(pcd_tr, rgba=(1, 0, 0, 1))
        # base.run()
        surface = RBFInterpolator(pcd_tr[:, :2], pcd_tr[:, 2])
        for i in range(1, len(drawpath)):
            real_len = np.linalg.norm(np.array(drawpath[i]) - np.array(drawpath[i - 1]))
            p1 = np.asarray(pos_nrml_list[i - 1][0])
            p2 = np.asarray(pos_nrml_list[i][0])
            v_draw = p2 - p1
            pm = np.dot(np.linalg.inv(transmat), p1)
            # pm = p1
            step = .1
            iter_times = 1 / step
            prj_len = 0

            while iter_times > 0:
                p_uv = (pm + np.dot(np.linalg.inv(transmat), v_draw) * .1)[:2]
                z = surface([p_uv])[0]
                pt = np.asarray([p_uv[0], p_uv[1], z])
                prj_len += np.linalg.norm(pt - pm)
                pm = pt
                iter_times -= 1
            error_list.append(round((prj_len - real_len) / real_len, 5))
            prj_len_list.append(prj_len)
            real_len_list.append(real_len)
            # except:
            #     error_list.append(None)
    if sum(real_len_list) != 0:
        error = round(abs(sum(real_len_list) - sum(prj_len_list)) / sum(real_len_list), 5)
    else:
        error = 0
    return error, error_list


def prj_drawpath_ss_on_pcd(pcd,objcm,nrmls, drawpath, mode='DI', direction=np.asarray((0, 0, 1)), error_method='ED',
                           toggledebug=False):
    print('--------------map single stroke on pcd--------------')
    print('pcd num:', len(pcd))
    print('draw path point num:', len(drawpath))
    surface = None
    transmat = np.eye(3)
    if mode == 'rbf_g':
        time_start = time.time()
        pca_trans = True
        # pcd = np.asarray(random.choices(pcd, k=5000))
        if pca_trans:
            pcd_tr, transmat = mu.trans_data_pcv(pcd, random_rot=False)
            surface = RBFInterpolator(pcd_tr[:, :2], pcd_tr[:, 2])
        else:
            surface = RBFInterpolator(pcd[:, :2], pcd[:, 2])
        time_cost_rbf = time.time() - time_start
        print('time cost(rbf global):', time_cost_rbf)

    kdt_d3, pcd_narray_d3 = get_kdt(pcd.tolist(), dimension=3)
    pos_nrml_list, error, error_list, time_cost = \
        __prj_stroke(drawpath, [0,0,0], pcd, nrmls, kdt_d3, objcm=objcm,
                     mode=mode, pcd_start_p=None, pcd_start_n=None, direction=direction, toggledebug=toggledebug,
                     error_method=error_method,surface=surface, transmat=transmat)
    print('avg error', np.mean(error_list))
    print('projetion time cost', time_cost)
    return pos_nrml_list, error_list, time_cost


def __prj_stroke_II(stroke, drawcenter, vs, nrmls, kdt_uv, scale_list, use_binp=False):
    time_start = time.time()
    avg_scale = np.mean(scale_list)
    stroke = np.array(stroke) / avg_scale
    pos_nrml_list = []
    stroke = np.array([(-p[0], -p[1]) for p in stroke])
    for p in stroke:
        p_uv = np.array(p) + drawcenter
        knn_d3_list = []
        if use_binp:
            uv = list(np.array(kdt_uv.data))
            knn_d2_list = []
            knn_inx_list = get_knn_indices(p_uv, kdt_uv, k=4)
            for inx in knn_inx_list:
                knn_d3_list.append(vs[inx])
                knn_d2_list.append(uv[inx])

            x = mu.bilinear_interp_2d(p_uv, knn_d2_list, [p[0] for p in knn_d3_list])
            y = mu.bilinear_interp_2d(p_uv, knn_d2_list, [p[1] for p in knn_d3_list])
            z = mu.bilinear_interp_2d(p_uv, knn_d2_list, [p[2] for p in knn_d3_list])
            p = (x, y, z)
            p_avg = (np.mean([p[0] for p in knn_d3_list]), np.mean([p[1] for p in knn_d3_list]),
                     np.mean([p[2] for p in knn_d3_list]))

            if np.linalg.norm(np.asarray(p) - np.asarray(p_avg)) > 10:
                p = p_avg
        else:
            knn_inx_list = get_knn_indices(p_uv, kdt_uv, k=3)
            for inx in knn_inx_list:
                knn_d3_list.append(vs[inx])

            p = (np.mean([p[0] for p in knn_d3_list]), np.mean([p[1] for p in knn_d3_list]),
                 np.mean([p[2] for p in knn_d3_list]))

        n = nrmls[knn_inx_list[0]]
        pos_nrml_list.append([p, n])

    print('time cost(projetion)', time.time() - time_start)
    error, error_list = get_prj_error(stroke * avg_scale, pos_nrml_list)

    return pos_nrml_list, error, error_list


def __prj_stroke_SI(stroke, drawcenter, uv, vs, nrmls, faces, scale_list, error_method='ED'):
    time_start = time.time()
    avg_scale = np.mean(scale_list)
    drawpath_stroke_scaled = np.array(stroke) / avg_scale
    pos_nrml_list = []
    # drawpath_stroke_scaled = np.array([(-p[0], -p[1]) for p in drawpath_stroke_scaled])
    # plt.scatter([v[0] for v in uv], [v[1] for v in uv], color='red', marker='.')
    for p in drawpath_stroke_scaled:
        p_uv = np.array(p) + drawcenter
        # plt.scatter([p_uv[0]], [p_uv[1]], color='gold', marker='.')

        for face_id, face in enumerate(faces):
            polygon = [uv[i] for i in face]
            if cu.is_in_polygon(p_uv, polygon):
                # plt.scatter([polygon[0][0]], [polygon[0][1]], color='green', marker='.')
                v_draw = p_uv - polygon[0]
                v_draw = (v_draw[0], v_draw[1], 0)
                rotmat = rm.rotmat_betweenvector(np.array([0, 0, 1]), nrmls[face_id])
                # base.pggen.plotSphere(base.render, vs[face[0]], radius=1, rgba=(1, 1, 0, 1))
                prj_p = vs[face[0]] + np.dot(rotmat, v_draw) * scale_list[face_id]
                pos_nrml_list.append([prj_p, nrmls[face_id]])
                break
    time_cost = time.time() - time_start
    print('time cost(projetion)', time_cost)
    error, error_list = get_prj_error(stroke, pos_nrml_list, method=error_method)

    return pos_nrml_list, error, error_list, time_cost


def prj_drawpath_ss_SI(obj_item, drawpath, toggledebug=False):
    uvs, vs, nrmls, faces, avg_scale = cu.lscm_objcm(obj_item.objcm, toggledebug=toggledebug)
    uv_center = cu.get_uv_center(uvs)
    pos_nrml_list, error, error_list, time_cost = __prj_stroke_SI(drawpath, uv_center, uvs, vs, nrmls, faces, avg_scale)
    print('avg error', error)

    return pos_nrml_list, error_list, time_cost


def prj_drawpath_ms_SI(obj_item, drawpath_ms, toggledebug=False):
    time_cost_total = 0
    uvs, vs, nrmls, faces, scale_list = cu.lscm_objcm(obj_item.objcm, toggledebug=toggledebug)
    # uvs, vs, nrmls, faces, scale_list = cu.lscm_objcm(pcdu.reconstruct_surface(obj_item.pcd), toggledebug=toggledebug)
    uv_center = cu.get_uv_center(uvs)
    error_ms = []
    error_list_ms = []
    pos_nrml_list_ms = []
    for drawpath in drawpath_ms:
        pos_nrml_list, error, error_list, time_cost = \
            __prj_stroke_SI(drawpath, uv_center, uvs, vs, nrmls, faces, scale_list)
        pos_nrml_list_ms.append(pos_nrml_list)
        error_ms.append(error)
        error_list_ms.extend(error_list)
        time_cost_total += time_cost
    print('avg error', np.mean(error_ms))

    return pos_nrml_list_ms, error_list_ms, time_cost_total


def prj_drawpath_ss_II(obj_item, drawpath, toggledebug=False):
    time_start = time.time()
    uvs, vs, nrmls, faces, scale_list = cu.lscm_pcd(obj_item.pcd, obj_item.nrmls, toggledebug=toggledebug)
    uv_center = cu.get_uv_center(uvs)
    kdt_uv, _ = get_kdt(uvs, dimension=2)
    pos_nrml_list, error, error_list = __prj_stroke_II(drawpath, uv_center, vs, nrmls, kdt_uv, scale_list,
                                                       use_binp=True)
    print('avg error', error)

    return pos_nrml_list, error_list, time.time() - time_start


def prj_drawpath_ms_II(obj_item, drawpath_ms, toggledebug=False):
    time_strat = time.time()
    uvs, vs, nrmls, faces, scale_list = cu.lscm_pcd(obj_item.pcd, obj_item.nrmls, toggledebug=toggledebug)
    uv_center = cu.get_uv_center(uvs)

    error_ms = []
    error_list_ms = []

    kdt_uv, _ = get_kdt(uvs, dimension=2)
    pos_nrml_list_ms = []
    for drawpath in drawpath_ms:
        pos_nrml_list, error, error_list = \
            __prj_stroke_II(drawpath, uv_center, vs, nrmls, kdt_uv, scale_list, use_binp=True)
        pos_nrml_list_ms.append(pos_nrml_list)
        error_ms.append(error)
        error_list_ms.extend(error_list)
    print('avg error', np.mean(error_ms))

    return pos_nrml_list_ms, error_list_ms, time.time() - time_strat


def prj_drawpat_ms_II_temp(obj_item, drawpath_ms, sample_num=None, toggledebug=False):
    time_start = time.time()
    uvs, vs, nrmls, faces, scale_list = \
        cu.lscm_parametrization_objcm_temp(obj_item.objcm, toggledebug=toggledebug, sample_num=sample_num)
    uv_center = cu.get_uv_center(uvs)

    error_ms = []
    error_list_ms = []

    kdt_uv, _ = get_kdt(uvs, dimension=2)
    pos_nrml_list_ms = []
    for drawpath in drawpath_ms:
        pos_nrml_list, error, error_list = \
            __prj_stroke_II(drawpath, uv_center, vs, nrmls, kdt_uv, scale_list, use_binp=True)
        pos_nrml_list_ms.append(pos_nrml_list)
        error_ms.append(error)
        error_list_ms.extend(error_list)
    print('avg error', np.mean(error_ms))

    return pos_nrml_list_ms, error_list_ms, time.time() - time_start


def show_drawpath(pos_nrml_list, color=(1, 0, 0), show_nrmls=False, transparency=1.0):
    for p, n in pos_nrml_list:
        if p is not None:
            base.pggen.plotSphere(base.render, np.array(p), radius=1, rgba=(color[0], color[1], color[2], 1))
            if show_nrmls:
                base.pggen.plotArrow(base.render, spos=p, epos=p + 10 * n,
                                     rgba=(color[0], color[1], color[2], transparency))


def get_penmat4(pos_nrml_list, pen_orgrot=(1, 0, 0)):
    objmat4_list = []
    for p, n in pos_nrml_list:
        # TODO: remove if
        # if rm.angle_between_vectors(n, np.asarray((0, 1, 0))) > 3:
        #     rot = rm.rotmat_betweenvector(pen_orgrot, -n)
        # else:
        #     rot = rm.rotmat_betweenvector(pen_orgrot, n)
        rot = rm.rotmat_betweenvector(pen_orgrot, n)
        objmat4_list.append(rm.homobuild(p, rot))
    return objmat4_list


def remove_list_dup(l):
    result = []
    temp = []
    for i in l:
        if str(i) not in temp:
            temp.append(str(i))
            result.append(i)
    return result


def flatten_nested_list(nested_list):
    return [p for s in nested_list for p in s]


def get_connection_error(pos_nrml_list, size=(80, 80), step=1):
    def __extract_pos(pos_nrml_list):
        stroke_list = []
        for s in pos_nrml_list:
            stroke = []
            for t in s:
                stroke.append(t[0])
            stroke_list.append(stroke)
        return stroke_list

    def __cal_dist(stroke_pos_list, step=step):
        grid_points_dist = []
        x_strip = size[0] // step

        for i in range(0, x_strip + 1):
            for j in range(x_strip + 1, len(stroke_pos_list)):
                if i % 2 != 0:
                    h_id = (j - x_strip - 1) * step
                else:
                    h_id = size[0] - (j - x_strip - 1) * step

                if j % 2 != 0:
                    v_id = size[1] - i * step
                else:
                    v_id = i * step

                p1 = np.asarray(stroke_pos_list[i][h_id])
                p2 = np.asarray(stroke_pos_list[j][v_id])
                base.pggen.plotSphere(base.render, p1, radius=2, rgba=(1, 0, 0, 1))
                base.pggen.plotSphere(base.render, p2, radius=2, rgba=(0, 1, 0, 1))

                grid_points_dist.append(np.linalg.norm(p1 - p2))
                # print(i, j, h_id, v_id, dist)

        return grid_points_dist

    stroke_pos_list = __extract_pos(pos_nrml_list)
    grid_points_dist = __cal_dist(stroke_pos_list)
    return len(grid_points_dist), np.average(grid_points_dist)


def dump_mapping_res(f_name, tgt_item, drawpath, pos_nrml_list, time_cost):
    res_dict = {'drawpath': drawpath, 'pos_nrml_list': pos_nrml_list, 'time_cost': time_cost,
                'objpcd': tgt_item.pcd}
    pickle.dump(res_dict, open(os.path.join(config.ROOT, 'log/mapping', f_name), 'wb'))


if __name__ == '__main__':
    """
    set up env and param
    """

    import visualization.panda.world as wd

    base = wd.World(cam_pos=np.array([-.2, -.7, .42]), lookat_pos=np.array([0, 0, 0]))

    DRAWREC_SIZE = [80, 80]
    """
    load mesh model
    """
    stl_f_name = 'cylinder_surface.stl'

    objpos = (0, 0, 0)
    objrot = (0, 0, 0)
    direction = np.asarray((0, 0, 1))
    # direction = np.asarray((0, -1, 0))
    bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
    bowl_model.set_rgba([.3, .3, .3, .3])
    bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi, 0, 0))
    bowl_model.attach_to(base)

    pn_direction = np.array([0, 0, -1])

    pcd, nrmls = bowl_model.sample_surface(toggle_option='normals', radius=.002)
    selection = nrmls.dot(-pn_direction) > .1
    pcd = pcd[selection]
    nrmls = nrmls[selection]
    tree = cKDTree(pcd)

    pt_direction = rm.orthogonal_vector(pn_direction, toggle_unit=True)
    tmp_direction = np.cross(pn_direction, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, pn_direction))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array([-.07, -.03, .1])
    twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[1, 1, 1, .3])
    twod_plane.attach_to(base)

    circle_radius = .05
    line_segs = [[homomat[:3, 3], homomat[:3, 3] + pt_direction * .05],
                 [homomat[:3, 3] + pt_direction * .05, homomat[:3, 3] + pt_direction * .05 + tmp_direction * .05]]
    gm.gen_linesegs(line_segs).attach_to(base)
    gm.gen_arrow(spos=line_segs[0][0], epos=line_segs[0][1], thickness=0.004).attach_to(base)
    spt = homomat[:3, 3]



    """
    single stroke
    """
    # metrology based method
    pos_nrml_list, error_list_msd, time_cost = \
        prj_drawpath_ss_on_pcd(pcd,objcm,nrmls, drawpath, mode='EI', direction=direction, toggledebug=True)
    show_drawpath(pos_nrml_list, color=(0, 0, 1))
    base.run()

    # rbf
    # pos_nrml_list, error_list_msd, time_cost = \
    #     prj_drawpath_ss_on_pcd(tgt_item, drawpath, mode='rbf', direction=direction, toggledebug=False)
    # show_drawpath(pos_nrml_list, color=(0, 0, 1))
    # base.run()
