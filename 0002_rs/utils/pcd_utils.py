import copy
import itertools
import math
import os
import random
import time

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree

import basis.o3dhelper as o3d_helper
import basis.o3dhelper as o3dh
import basis.robot_math as rm
import basis.trimesh.sample as ts
import config
import modeling.collision_model as cm
import modeling.geometric_model as gm
import motionplanner.nbv_pcn_opt_solver as nbv_solver
import utils.math_utils as mu
from basis import trimesh

COLOR = np.asarray([[31, 119, 180, 255], [44, 160, 44, 255], [214, 39, 40, 255], [255, 127, 14, 255]]) / 255


def make_3dax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


def get_knn_indices(p, kdt, k=3):
    distances, indices = kdt.query([p], k=k, return_distance=True)
    return indices[0]


def get_knn(p, kdt, k=3):
    p_nearest_inx = get_knn_indices(p, kdt, k=k)
    pcd = list(np.array(kdt.data))
    return np.asarray([pcd[p_inx] for p_inx in p_nearest_inx])


def get_min_dist(p, kdt):
    p_nearest_inx = get_knn_indices(p, kdt, k=1)
    pcd = list(np.array(kdt.data))
    # gm.gen_sphere(np.asarray(pcd[p_nearest_inx[0]]), radius=.002).attach_to(base)
    return np.linalg.norm(np.asarray(p) - np.asarray(pcd[p_nearest_inx[0]]))


def get_nn_indices_by_dist(p, kdt, step=1.0):
    result_indices = []
    distances, indices = kdt.query([p], k=200, return_distance=True)
    distances = distances[0]
    indices = indices[0]
    for i in range(len(distances)):
        if distances[i] < step:
            result_indices.append(indices[i])
    return result_indices


def get_knn_by_dist(p, kdt, radius=1.0):
    p_nearest_inx = get_nn_indices_by_dist(p, kdt, step=radius)
    pcd = list(np.array(kdt.data))
    return np.asarray([pcd[p_inx] for p_inx in p_nearest_inx])


def get_kdt(p_list, dimension=3):
    time_start = time.time()
    p_list = np.asarray(p_list)
    p_narray = np.array(p_list[:, :dimension])
    kdt = KDTree(p_narray, leaf_size=100, metric='euclidean')
    # print('time cost(kdt):', time.time() - time_start)
    return kdt, p_narray


def get_nrml_pca(knn):
    pcv, pcaxmat = rm.compute_pca(knn)
    return np.asarray(pcaxmat[:, np.argmin(pcv)])


def get_frame_pca(knn):
    pcv, pcaxmat = rm.compute_pca(knn)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x = pcaxmat[:, inx[2]]
    y = pcaxmat[:, inx[1]]
    z = pcaxmat[:, inx[0]]
    if z[2] > 0:
        z = -z
    if y[2] < 0:
        y = -y
    return np.asarray([y, x, z]).T


def get_objpcd(objcm, objmat4=np.eye(4), sample_num=100000, toggledebug=False):
    objpcd = objcm.sample_surface(nsample=sample_num, toggle_option=None)
    objpcd = trans_pcd(objpcd, objmat4)

    if toggledebug:
        objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        objpcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([objpcd])
    print(f"---------------success sample {sample_num} points---------------")

    return objpcd


def remove_pcd_zeros(pcd):
    return pcd[np.all(pcd != 0, axis=1)]


def get_pcd_center(pcd):
    return np.array((np.mean(pcd[:, 0]), np.mean(pcd[:, 1]), np.mean(pcd[:, 2])))


def get_pcd_tip(pcd, axis=0):
    """
    get the smallest point along an axis

    :param pcd:
    :param axis: 0-x,1-y,2-z
    :return: 3D point
    """

    return pcd[list(pcd[:, axis]).index(min(list(pcd[:, axis])))]


def crop_pcd(pcd, x_range, y_range, z_range, zeros=False):
    pcd_res = []
    for p in pcd:
        if x_range[0] < p[0] < x_range[1] and y_range[0] < p[1] < y_range[1] and z_range[0] < p[2] < z_range[1]:
            pcd_res.append(p)
        else:
            if zeros:
                pcd_res.append(np.asarray([0, 0, 0]))
    return np.array(pcd_res)


def trans_pcd(pcd, transmat):
    pcd = copy.deepcopy(np.asarray(pcd))
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def trans_p(p, transmat=None):
    if transmat is None:
        return p
    return trans_pcd(np.asarray([p]), transmat)[0]


def get_pcd_w_h(objpcd_std):
    def __sort_w_h(a, b):
        if a > b:
            return a, b
        else:
            return b, a

    return __sort_w_h(max(objpcd_std[:, 0]) - min(objpcd_std[:, 0]), max(objpcd_std[:, 1]) - min(objpcd_std[:, 1]))


def get_org_convexhull(pcd, color=(1, 1, 1), transparency=1, toggledebug=False):
    """
    create CollisionModel by pcd

    :param pcd:
    :param color:
    :param transparency:
    :return: CollisionModel
    """

    convexhull = trimesh.Trimesh(vertices=pcd)
    convexhull = convexhull.convex_hull
    obj = cm.CollisionModel(initor=convexhull, cdprimit_type="ball")
    if toggledebug:
        obj.set_rgba((color[0], color[1], color[2], transparency))
        obj.attach_to(base)
        obj.show_localframe()

    return obj


def get_std_convexhull(pcd, origin="center", color=(1, 1, 1), transparency=1, toggledebug=False, toggleransac=True):
    """
    create CollisionModel by pcd, standardized rotation

    :param pcd:
    :param origin: "center" or "tip"
    :param color:
    :param transparency:
    :param toggledebug:
    :param toggleransac:
    :return: CollisionModel, position of origin
    """

    rot_angle = get_rot_frompcd(pcd, toggledebug=toggledebug, toggleransac=toggleransac)
    center = get_pcd_center(pcd)
    origin_pos = np.array(center)

    pcd = pcd - np.array([center]).repeat(len(pcd), axis=0)
    pcd = trans_pcd(pcd, rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((0, 0, 1), -rot_angle)))

    convexhull = trimesh.Trimesh(vertices=pcd)
    convexhull = convexhull.convex_hull
    obj = cm.CollisionModel(initor=convexhull)
    obj_w, obj_h = get_pcd_w_h(pcd)

    if origin == "tip":
        tip = get_pcd_tip(pcd, axis=0)
        origin_pos = center + trans_pcd(np.array([tip]),
                                        rm.homomat_from_posrot((0, 0, 0),
                                                               rm.rotmat_from_axangle((0, 0, 1), rot_angle)))[0]
        pcd = pcd - np.array([tip]).repeat(len(pcd), axis=0)

        convexhull = trimesh.Trimesh(vertices=pcd)
        convexhull = convexhull.convex_hull
        obj = cm.CollisionModel(initor=convexhull)

    if toggledebug:
        obj.set_rgba((color[0], color[1], color[2], transparency))
        obj.attach_to(base)
        obj.show_localframe()
        print("Rotation angle:", rot_angle)
        print("Origin:", center)
        base.pggen.plotSphere(base.render, center, radius=2, rgba=(1, 0, 0, 1))

    return obj, obj_w, obj_h, origin_pos, rot_angle


def get_rot_frompcd(pcd, toggledebug=False, toggleransac=True):
    """

    :param pcd:
    :param toggledebug:
    :param toggleransac: use ransac linear regression or not
    :return: grasp center and rotation angle
    """

    if max(pcd[:, 0]) - min(pcd[:, 0]) > max(pcd[:, 1]) - min(pcd[:, 1]):
        X = pcd[:, 0].reshape((len(pcd), 1))
        y = pcd[:, 1]
    else:
        X = pcd[:, 1].reshape((len(pcd), 1))
        y = pcd[:, 0]
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    ransac_coef = ransac.estimator_.coef_

    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    lr_coef = lr.coef_

    if toggledebug:
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)

        # Predict data of estimated models
        line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        line_y = lr.predict(line_X)
        line_y_ransac = ransac.predict(line_X)

        # Compare estimated coefficients
        print("Estimated coefficients (linear regression, RANSAC):", lr.coef_, ransac.estimator_.coef_)

        plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
                    label='Inliers')
        plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
                    label='Outliers')
        plt.plot(line_X, line_y, color='navy', linewidth=2, label='Linear regressor')
        plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
                 label='RANSAC regressor')
        plt.legend(loc='lower right')
        plt.xlabel("Input")
        plt.ylabel("Response")
        plt.show()

    if toggleransac:
        coef = ransac_coef[0]
    else:
        coef = lr_coef[0]

    if max(pcd[:, 0]) - min(pcd[:, 0]) > max(pcd[:, 1]) - min(pcd[:, 1]):
        return math.degrees(math.atan(coef))
    else:
        return math.degrees(math.atan(1 / coef))


def get_max_cluster(pts, eps=.003, min_samples=2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []
    max_len = 0

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            if len(cluster) > max_len:
                max_len = len(cluster)
                res = cluster
                mask = [class_member_mask & core_samples_mask]

    return np.asarray(res), mask


def get_nearest_cluster(pts, seed=(0, 0, 0), eps=.003, min_samples=2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []
    max_len = 0
    min_dist = 100
    # gm.gen_sphere(seed, radius=.001).attach_to(base)

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            center = np.mean(cluster, axis=0)
            dist = np.linalg.norm(np.asarray(seed) - center)
            # if len(cluster) > max_len:
            #     max_len = len(cluster)
            #     res = cluster
            #     mask = [class_member_mask & core_samples_mask]
            if dist < min_dist:
                min_dist = dist
                res = cluster
                mask = [class_member_mask & core_samples_mask]

    return np.asarray(res), mask


def reconstruct_surface(pcd, radii=[5], toggledebug=False):
    print("---------------reconstruct surface bp---------------")
    pcd = np.asarray(pcd)
    tmmesh = o3d_helper.reconstructsurfaces_bp(pcd, radii=radii, doseparation=False)
    obj = cm.CollisionModel(initor=tmmesh)
    if toggledebug:
        obj.set_rgba((1, 1, 1, 1))
        obj.attach_to(base)
    return obj


def reconstruct_surface_list(pcd, radii=[5], color=(1, 1, 1), transparency=1, toggledebug=False):
    pcd = np.asarray(pcd)
    tmmeshlist = o3d_helper.reconstructsurfaces_bp(pcd, radii=radii, doseparation=True)
    obj_list = []
    for tmmesh in tmmeshlist:
        obj = cm.CollisionModel(initor=tmmesh)
        obj_list.append(obj)
        if toggledebug:
            obj.set_rgba((color[0], color[1], color[2], transparency))
            obj.attach_to(base)
    return obj_list


def get_pcdidx_by_pos(pcd, realpos, diff=10, dim=3):
    idx = 0
    distance = 100
    result_point = None
    for i in range(len(pcd)):
        point = pcd[i]
        if realpos[0] - diff < point[0] < realpos[0] + diff and realpos[1] - diff < point[1] < realpos[1] + diff:
            temp_distance = np.linalg.norm(realpos[:dim] - point[:dim])
            # print(i, point, temp_distance, distance)
            if temp_distance < distance:
                distance = temp_distance
                idx = i
                result_point = point
    return idx, result_point


def get_objpcd_withnrmls(objcm, objmat4=np.eye(4), sample_num=100000, toggledebug=False, sample_edge=False):
    objpcd_nrmls = []
    faces = objcm.objtrm.faces
    vertices = objcm.objtrm.vertices
    face_nrmls = objcm.objtrm.face_normals
    nrmls = objcm.objtrm.vertex_normals

    if sample_num is not None:
        objpcd, faceid = ts.sample_surface(objcm.objtrm, count=sample_num)
        objpcd = list(objpcd)
        for i in faceid:
            objpcd_nrmls.append(np.array(face_nrmls[i]))
    else:
        objpcd = vertices
        objpcd_nrmls.extend(nrmls)

    # v_temp = []
    # for i, face in enumerate(faces):
    #     for j, v in enumerate(face):
    #         if v not in v_temp:
    #             v_temp.append(v)
    #             objpcd.append(vertices[v])
    #             objpcd_nrmls.append(nrmls[i])

    if sample_edge:
        for i, face in enumerate(faces):
            for v_pair in itertools.combinations(face, 2):
                edge_plist = mu.linear_interp_3d(vertices[v_pair[0]], vertices[v_pair[1]], step=.5)
                objpcd.extend(edge_plist)
                objpcd_nrmls.extend(np.repeat([nrmls[i]], len(edge_plist), axis=0))

    objpcd = np.asarray(objpcd)
    objpcd = trans_pcd(objpcd, objmat4)
    # objpcd_nrmls = np.asarray([-n if np.dot(n, np.asarray([0, 0, 1])) < 0 else n for n in objpcd_nrmls])

    if toggledebug:
        # objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        # objpcd.paint_uniform_color([1, 0.706, 0])
        # o3d.visualization.draw_geometries([objpcd])

        # objcm.sethomomat(objmat4)
        # objcm.setColor(1, 1, 1, 0.7)
        # objcm.reparentTo(base.render)
        show_pcd(objpcd, rgba=(1, 0, 0, 1))
        for i, n in enumerate(objpcd_nrmls):
            import random
            v = random.choice(range(0, 10000))
            if v == 1:
                gm.gen_arrow(spos=objpcd[i], epos=objpcd[i] + 10 * n).attach_to(base)
                gm.gen_sphere(pos=objpcd[i], rgba=(1, 0, 0, 1)).attach_to(base)
        base.run()

    return objpcd, np.asarray(objpcd_nrmls)


def get_objpcd_partial(objcm, objmat4=np.eye(4), sample_num=100000, toggledebug=False):
    objpcd = np.asarray(ts.sample_surface(objcm.objtrm, count=sample_num)[0])
    objpcd = trans_pcd(objpcd, objmat4)

    grid = {}
    for p in objpcd:
        x = round(p[0], 0)
        y = round(p[1], 0)
        if str((x, y)) in grid.keys():
            grid[str((x, y))].append(p)
        else:
            grid[str((x, y))] = [p]
    objpcd_partial = []
    for k, v in grid.items():
        z_max = max(np.array(v)[:, 2])
        for p in v:
            objpcd_partial.append([p[0], p[1], z_max])
    objpcd_partial = np.array(objpcd_partial)

    print("Length of org pcd", len(objpcd))
    print("Length of partial pcd", len(objpcd_partial))

    if toggledebug:
        objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        objpcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([objpcd])

        objpcd_partial = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd_partial))
        objpcd_partial.paint_uniform_color([0, 0.706, 1])
        o3d.visualization.draw_geometries([objpcd_partial])

        # pcddnp = base.pg.genpointcloudnp(objpcd)
        # pcddnp.reparentTo(base.render)

    return objpcd_partial


def get_objpcd_partial_bycam_pos(objcm, objmat4=np.eye(4), smp_num=100000, cam_pos=np.array([.86, .08, 1.78]),
                                 toggledebug=False):
    def __sigmoid(angle):
        angle = np.degrees(angle)
        return 1 / (1 + np.exp((angle - 90) / 90)) - 0.5

    objpcd, _ = objcm.sample_surface(radius=.0005, nsample=smp_num)
    objpcd_partial = []
    area_list = objcm.objtrm.area_faces
    area_sum = sum(area_list)

    for i, n in enumerate(objcm.objtrm.face_normals):
        n = np.dot(n, objmat4[:3, :3])
        angle = rm.angle_between_vectors(n, np.array(cam_pos - np.mean(objpcd, axis=0)))

        if angle > np.pi / 2:
            continue
        else:
            objcm_tmp = copy.deepcopy(objcm)
            # print(i, angle, __sigmoid(angle))
            mask_tmp = [False] * len(objcm.objtrm.face_normals)
            mask_tmp[i] = True
            objcm_tmp.objtrm.update_faces(mask_tmp)
            pcd_tmp, _ = \
                objcm_tmp.sample_surface(radius=.0005,
                                         nsample=int(smp_num / area_sum * area_list[i] * __sigmoid(angle) * 100))
            objpcd_partial.extend(np.asarray(pcd_tmp))
    if len(objpcd_partial) > smp_num:
        objpcd_partial = random.sample(objpcd_partial, smp_num)
    objpcd_partial = np.array(objpcd_partial)
    objpcd_partial = trans_pcd(objpcd_partial, objmat4)

    print("Length of org pcd", len(objpcd))
    print("Length of source pcd", len(objpcd_partial))

    if toggledebug:
        objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        objpcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([objpcd])

        objpcd_partial = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd_partial))
        objpcd_partial.paint_uniform_color([0, 0.706, 1])
        o3d.visualization.draw_geometries([objpcd_partial])

    return objpcd_partial


def get_nrmls(pcd, camera_location=(800, -200, 1800), toggledebug=False):
    pcd_o3d = o3d_helper.nparray2o3dpcd(pcd)
    pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=1000))
    # for n in np.asarray(pcd.normals)[:10]:
    #     print(n)
    # print("----------------")
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd_o3d, camera_location=camera_location)
    # for n in np.asarray(pcd.normals)[:10]:
    #     print(n)
    pcd_nrmls = np.asarray(pcd_o3d.normals)
    pcd_nrmls = np.asarray([-n if np.dot(n, np.asarray([0, 0, 1])) < 0 else n for n in pcd_nrmls])

    if toggledebug:
        for i, n in enumerate(pcd_nrmls):
            import random
            v = random.choice(range(0, 100))
            if v == 1:
                base.pggen.plotArrow(base.render, spos=pcd[i], epos=pcd[i] + 10 * n)
        base.run()
    return pcd_nrmls


def get_plane(pcd, dist_threshold=0.0002, toggledebug=False):
    pcd_o3d = o3d_helper.nparray2o3dpcd(pcd)
    plane, inliers = pcd_o3d.segment_plane(distance_threshold=dist_threshold, ransac_n=30, num_iterations=100)
    plane_pcd = pcd[inliers]
    center = get_pcd_center(plane_pcd)
    if toggledebug:
        show_pcd(pcd[inliers], rgba=(1, 1, 0, 1))
        gm.gen_arrow(spos=center, epos=plane[:3] * plane[3], rgba=(1, 0, 0, 1)).attach_to(base)

        pt_direction = rm.orthogonal_vector(plane[:3], toggle_unit=True)
        tmp_direction = np.cross(plane[:3], pt_direction)
        plane_rotmat = np.column_stack((pt_direction, tmp_direction, plane[:3]))
        homomat = np.eye(4)
        homomat[:3, :3] = plane_rotmat
        homomat[:3, 3] = center
        gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[1, 1, 0, .3]).attach_to(base)

    return plane[:3], plane[3]


def surface_inp(p, v, kdt_d3, inp=0.0005, max_nn=100):
    pseq = []
    rotseq = []
    times = int(np.linalg.norm(v) / inp)
    knn = get_knn(p, kdt_d3, k=max_nn)
    n = get_nrml_pca(knn)

    for _ in range(times):
        v_cur = rm.unit_vector(v - v * n)
        # v_cur = rm.unit_vector(np.dot(rotmat, v))
        pt = np.asarray(p) + v_cur * inp
        knn = get_knn(pt, kdt_d3, k=max_nn)
        center = get_pcd_center(np.asarray(knn))
        n_cur = get_nrml_pca(knn)
        p_cur = pt - np.dot((pt - center), n_cur) * n_cur
        if np.dot(n_cur, np.asarray([0, 0, 1])) < 0:
            n_cur = -n_cur
        pseq.append(p_cur)
        rot = np.asarray([rm.unit_vector(n_cur), -rm.unit_vector(np.cross(n_cur, np.cross(n_cur, v_cur))),
                          rm.unit_vector(np.cross(n_cur, v_cur))]).T
        rotseq.append(rot)
        p = p_cur
        n = n_cur
    return pseq, rotseq


def skeleton(pcd):
    import sknw
    from skimage.morphology import skeletonize

    pcd = o3dh.nparray2o3dpcd(np.asarray(pcd))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=.001)
    voxel_bin = np.zeros([100, 100, 100])
    for v in voxel_grid.get_voxels():
        voxel_bin[v.grid_index[0]][v.grid_index[1]][v.grid_index[2]] = 1
    ax = make_3dax(True)
    ax.voxels(voxel_bin, shade=False)
    skeleton = skeletonize(voxel_bin)
    graph = sknw.build_sknw(skeleton, multi=True)

    exit_node = list(set([s for s, e in graph.edges()] + [e for s, e in graph.edges()]))
    nodes = graph.nodes()
    stroke_list = [[nodes[i]['o'][::-1]] for i in nodes if i not in exit_node]

    for (s, e) in graph.edges():
        for cnt in range(10):
            stroke = []
            try:
                ps = graph[s][e][cnt]['pts']
                for i in range(len(ps)):
                    if i % 3 == 0:
                        stroke.append([ps[i][0], ps[i][1], ps[i][2]])
                # stroke.append([ps[-1, 1], ps[-1, 0]])
                stroke_list.append(stroke)
                print(stroke)
            except:
                break

    for stroke in np.asarray(stroke_list):
        stroke = np.asarray(stroke)
        ax.scatter(stroke[:, 0], stroke[:, 1], stroke[:, 2])
    plt.show()


def extract_lines_from_pcd(img, pcd, z_range, line_thresh=0.002, line_size_thresh=300, toggledebug=False):
    import pyransac3d as pyrsc
    lines = []

    if toggledebug:
        cv2.imshow('', img)
        cv2.waitKey(0)
        pcd_pix = pcd.reshape(img.shape[0], img.shape[1], 3)
        if z_range is not None:
            mask_1 = np.where(pcd_pix[:, :, 2] < z_range[1], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
                np.uint8)
            mask_2 = np.where(pcd_pix[:, :, 2] > z_range[0], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
                np.uint8)
            mask = cv2.bitwise_and(mask_1, mask_2)
            img = cv2.bitwise_and(img, mask)
            cv2.imshow('', mask)
            cv2.waitKey(0)
            cv2.imshow('', img)
            cv2.waitKey(0)

    if z_range is not None:
        pcd_crop = crop_pcd(pcd, x_range=(0, 1), y_range=(-1, 1), z_range=z_range)
    else:
        pcd_crop = copy.deepcopy(pcd)
    show_pcd(pcd_crop, rgba=(1, 1, 1, .5))

    while 1:
        print(f'------------{len(pcd_crop)}------------')
        line = pyrsc.Line()
        line.fit(pcd_crop, thresh=line_thresh, maxIteration=1000)
        print('Candidate line segment size:', len(line.inliers))
        # show_pcd(pcd_crop[line.inliers], rgba=(1, 0, 0, 1))

        if len(line.inliers) > line_size_thresh:
            lines.append([line.A, pcd_crop[line.inliers]])
            # gm.gen_sphere(line.B, rgba=(0, 0, 1, 1), radius=.002).attach_to(base)
            print(line.A, line.B)
            pcd_crop = np.delete(pcd_crop, line.inliers, axis=0)
        else:
            break
    return lines


def remove_outliers(pts, nb_points=50, radius=0.005, toggledebug=False):
    o3dpcd = o3dh.nparray2o3dpcd(np.asarray(pts))
    o3dpcd, ind = o3dpcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    if toggledebug:
        inlier_cloud = o3dpcd.select_by_index(ind)
        outlier_cloud = o3dpcd.select_by_index(ind, invert=True)
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return np.asarray(o3dpcd.points)


def cal_conf(pts, voxel_size=.01, cam_pos=(0, 0, 0), theta=None, toggledebug=False):
    o3dpcd = o3d_helper.nparray2o3dpcd(pts)
    downpcd = o3dpcd.voxel_down_sample(voxel_size=voxel_size)
    # downpcd = o3dpcd.uniform_down_sample(10)
    # o3d.visualization.draw_geometries([downpcd])
    kdt_d3, _ = get_kdt(pts)
    zeta1_list = []
    d_list = []
    p_list = []
    nrmls = []
    conf_list = []
    for p in np.asarray(downpcd.points):
        knn = get_knn_by_dist(p, kdt_d3, radius=voxel_size)
        # knn = get_knn(p, kdt_d3, k=50)
        if len(knn) < 5:
            continue
        pcv_unsort, pcaxmat = rm.compute_pca(knn)
        pcv = sorted(pcv_unsort, reverse=True)
        zeta1 = pcv[0] - pcv[1]
        zeta2 = pcv[1] - pcv[2]
        zeta3 = pcv[2]
        d = [zeta1, zeta2, zeta3].index(max([zeta1, zeta2, zeta3]))
        zeta1_list.append(zeta1)
        d_list.append(d)
        p_list.append(p)
        inx = sorted(range(len(pcv_unsort)), key=lambda k: pcv_unsort[k])
        n = pcaxmat[:, inx[0]]
        if rm.angle_between_vectors(n, cam_pos - p) > np.pi / 2:
            n = -n
        nrmls.append(n)

    for i in range(len(zeta1_list)):
        c = 1 - (zeta1_list[i] - min(zeta1_list)) / (max(zeta1_list) - min(zeta1_list))
        if d_list[i] == 0:
            rgba = (c, 0, 1 - c, 1)
        else:
            c = 1
            rgba = (1, 1, 1, 1)
        conf_list.append(c)
        if toggledebug:
            gm.gen_sphere(p_list[i], radius=.001, rgba=rgba).attach_to(base)
            if c < .5:
                gm.gen_arrow(spos=p_list[i], epos=p_list[i] + nrmls[i] * .05, thickness=.002,
                             rgba=rgba).attach_to(base)

    if theta is not None:
        res_inx_list = []
        for i in range(len(p_list)):
            a = rm.angle_between_vectors(nrmls[i], cam_pos - p_list[i])
            if a > np.pi / 2:
                a = np.pi - rm.angle_between_vectors(nrmls[i], cam_pos - p_list[i])
            if a > theta:
                res_inx_list.append(i)
        p_list = np.asarray(p_list)[res_inx_list]
        nrmls = np.asarray(nrmls)[res_inx_list]
        conf_list = np.asarray(conf_list)[res_inx_list]
    return p_list, nrmls, conf_list


def extract_main_vec(pts, nrmls, confs, threshold=np.radians(30), toggledebug=False):
    inx = sorted(range(len(confs)), key=lambda k: confs[k])
    confs = np.asarray(confs)[inx]
    pts = np.asarray(pts)[inx]
    # nrmls = np.asarray([rm.unit_vector(n) for n in nrmls])[inx]
    nrmls = np.asarray(nrmls)[inx]
    res_list = list(range(len(confs)))
    nbv_inx_dict = {}

    while len(res_list) > 0:
        i = res_list[0]
        n = np.asarray(nrmls[i])
        a_narry = np.arccos(np.dot(np.asarray(nrmls[res_list]), n))
        res_list_tmp = list(np.argwhere(a_narry > threshold).flatten())
        rm_list_tmp = list(np.argwhere(a_narry <= threshold).flatten())
        nbv_inx_dict[i] = np.asarray(res_list)[rm_list_tmp]
        res_list = np.asarray(res_list)[res_list_tmp]

    if toggledebug:
        for k, v in nbv_inx_dict.items():
            print(k, v, confs[k])
            p = np.asarray(pts[k])
            n = np.asarray(nrmls[k])
            gm.gen_arrow(p, p + n * .03, thickness=.002, rgba=(confs[k], 0, 1 - confs[k], 1)).attach_to(base)
            for i in v:
                p = np.asarray(pts[i])
                n = np.asarray(nrmls[i])
                gm.gen_arrow(p, p + n * .03, thickness=.001, rgba=(confs[k], 0, 1 - confs[k], .1)).attach_to(base)

    nbv_inx_list = list(nbv_inx_dict.keys())
    return np.asarray(pts)[nbv_inx_list], np.asarray(nrmls)[nbv_inx_list], np.asarray(confs)[nbv_inx_list]


def sort_kpts(kpts, seed):
    sort_ids = []
    while len(sort_ids) < len(kpts):
        dist_list = np.linalg.norm(kpts - seed, axis=1)
        sort_ids_tmp = np.argsort(dist_list)
        for i in sort_ids_tmp:
            if i not in sort_ids:
                sort_ids.append(i)
                break
        seed = kpts[sort_ids[-1]]
    return kpts[sort_ids]


def get_kpts_gmm(objpcd, n_components=20, means_init=None, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0, means_init=means_init).fit(X)
    kpts = sort_kpts(gmix.means_, seed=np.asarray([0, 0, 0]))

    if show:
        for i, p in enumerate(kpts[1:]):
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)

    kdt, _ = get_kdt(objpcd)
    kpts_rotseq = []
    for i, p in enumerate(kpts[:-1]):
        knn = get_knn(kpts[i], kdt, k=int(len(objpcd) / n_components))
        pcv, pcaxmat = rm.compute_pca(knn)
        y_v = kpts[i + 1] - kpts[i]
        x_v = pcaxmat[:, np.argmin(pcv)]
        if len(kpts_rotseq) != 0:
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 0], x_v) > np.pi / 2:
                x_v = -x_v
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 1], y_v) > np.pi / 2:
                y_v = -y_v
        z_v = np.cross(x_v, y_v)

        rot = np.asarray([rm.unit_vector(x_v), rm.unit_vector(y_v), rm.unit_vector(z_v)]).T
        kpts_rotseq.append(rot)
    kpts_rotseq.append(kpts_rotseq[-1])

    return np.asarray(kpts), np.asarray(kpts_rotseq)


def get_rots_wkpts(objpcd, kpts, k=None, show=True, rgba=(1, 0, 0, 1)):
    kdt, _ = get_kdt(objpcd)
    kpts_rotseq = []
    for i, p in enumerate(kpts[:-1]):
        knn = get_knn(kpts[i], kdt, k=int(len(objpcd) / len(kpts)) if k is None else k)
        pcv, pcaxmat = rm.compute_pca(knn)
        y_v = kpts[i + 1] - kpts[i]
        x_v = pcaxmat[:, np.argmin(pcv)]
        if len(kpts_rotseq) != 0:
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 0], x_v) > np.pi / 2:
                x_v = -x_v
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 1], y_v) > np.pi / 2:
                y_v = -y_v
        z_v = np.cross(x_v, y_v)

        rot = np.asarray([rm.unit_vector(x_v), rm.unit_vector(y_v), rm.unit_vector(z_v)]).T
        kpts_rotseq.append(rot)
    kpts_rotseq.append(kpts_rotseq[-1])
    if show:
        for i, p in enumerate(kpts[1:]):
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)
            gm.gen_frame(p, kpts_rotseq[i], thickness=.001, length=.01).attach_to(base)
    return np.asarray(kpts_rotseq)


def cal_distribution(pts, kpts, voxel_size=0.001, radius=.005):
    pts = np.asarray(pts)
    o3dpcd = o3dh.nparray2o3dpcd(pts)
    o3dpcd_down = o3dpcd.voxel_down_sample(voxel_size=voxel_size)
    confs = []
    kdt_i = o3d.geometry.KDTreeFlann(o3dpcd_down)
    for i, p in enumerate(np.asarray(kpts)):
        k, _, _ = kdt_i.search_radius_vector_3d(p, radius)
        confs.append(k)
    # print(confs, np.std(np.asarray(confs)),
    #       min(confs), len(np.asarray(o3dpcd_down.points)) / len(kpts))
    # o3dpcd.paint_uniform_color((1, 0, 0))
    # o3dpcd_down.paint_uniform_color((0, 1, 0))
    # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_down])
    return confs


def cal_nbv(pts, nrmls, confs, toggledebug=False):
    inx = sorted(range(len(confs)), key=lambda k: confs[k])
    confs = np.asarray(confs)[inx]
    pts = np.asarray(pts)[inx]
    nrmls = np.asarray(nrmls)[inx]
    show_pcd(pts, rgba=COLOR[0])
    pts, nrmls, confs = extract_main_vec(pts, nrmls, confs, threshold=np.radians(30), toggledebug=toggledebug)
    return np.asarray(pts)[np.argsort(confs)], \
           np.asarray(nrmls)[np.argsort(confs)], \
           np.asarray(confs)[np.argsort(confs)]


def cal_nbv_pcn(pts, pts_pcn, cam_pos=(0, 0, 0), theta=None, radius=.01, toggledebug=False):
    def _normalize(l):
        return [(v - min(l)) / (max(l) - min(l)) for v in l]

    _, _, trans = o3dh.registration_icp_ptpt(pts_pcn, pts, maxcorrdist=.02, toggledebug=False)
    pts_pcn = trans_pcd(pts_pcn, trans)
    if toggledebug:
        show_pcd(pts_pcn, rgba=COLOR[2])
        show_pcd(pts, rgba=COLOR[0])
    o3d_pcn = o3dh.nparray2o3dpcd(pts_pcn)
    o3d_pts = o3dh.nparray2o3dpcd(pts)
    o3d_pcn.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=200))
    o3d_pts.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=200))

    o3d_kpts = o3d_pcn.voxel_down_sample(voxel_size=radius)
    kpts = np.asarray(o3d_kpts.points)

    confs = cal_distribution(pts, kpts, radius=radius)
    nrmls = np.asarray(o3d_kpts.normals)
    if max(confs) != min(confs):
        confs = _normalize(confs)
    else:
        confs = np.ones(len(confs))
    if theta is not None:
        res_inx_list = []
        for i in range(len(kpts)):
            a = rm.angle_between_vectors(nrmls[i], cam_pos - kpts[i])
            if a > np.pi / 2:
                a = np.pi - rm.angle_between_vectors(nrmls[i], cam_pos - kpts[i])
            if a > theta:
                res_inx_list.append(i)
        kpts = np.asarray(kpts)[res_inx_list]
        nrmls = np.asarray(nrmls)[res_inx_list]
        confs = np.asarray(confs)[res_inx_list]

    if toggledebug:
        for i in range(len(confs)):
            gm.gen_sphere(kpts[i], radius=radius, rgba=[confs[i], 0, 1 - confs[i], .1]).attach_to(base)
            # if confs[i] < .4:
            #     gm.gen_arrow(kpts[i], kpts[i] + rm.unit_vector(nrmls[i]) * .02, rgba=[confs[i], 0, 1 - confs[i], 1],
            #                  thickness=.001).attach_to(base)
            gm.gen_arrow(kpts[1], kpts[1] + rm.unit_vector(nrmls[1]) * .02, rgba=[0, 0, 1, 1],
                         thickness=.001).attach_to(base)
    # kpts, nrmls, confs = extract_main_vec(kpts, nrmls, confs)
    # pts, nrmls, confs = extract_main_vec(pts, nrmls, confs, threshold=np.radians(30), toggledebug=toggledebug)

    return np.asarray(kpts)[np.argsort(confs)], \
           np.asarray(nrmls)[np.argsort(confs)], \
           np.asarray(confs)[np.argsort(confs)]


def opt_nbv_pcn(pts, model_name, load_model, cam_pos=(0, 0, 0), method='COBYLA'):
    nbv_opt = nbv_solver.NBVOptimizer(model_name=model_name, load_model=load_model, toggledebug=False)
    trans, rot, time_cost = nbv_opt.solve(pts, cam_pos, method=method)
    return trans, rot, time_cost


def cal_nbv_pcn_kpts(pts, pts_pcn, cam_pos=(0, 0, 0), theta=None, toggledebug=False):
    def _normalize(l):
        return [(v - min(l)) / (max(l) - min(l)) for v in l]

    _, _, trans = o3dh.registration_icp_ptpt(pts_pcn, pts, maxcorrdist=.02, toggledebug=False)
    pts_pcn = trans_pcd(pts_pcn, trans)
    show_pcd(pts_pcn, rgba=COLOR[1])
    show_pcd(pts, rgba=COLOR[0])
    kpts, kpts_rotseq = get_kpts_gmm(pts_pcn, n_components=16, show=False)
    confs = cal_distribution(pts, kpts)

    if max(confs) != min(confs):
        confs = _normalize(confs)
    else:
        confs = np.ones(len(confs))
    nrmls = kpts_rotseq[:, :, 0]
    if theta is not None:
        res_inx_list = []
        for i in range(len(kpts)):
            if rm.angle_between_vectors(nrmls[i], cam_pos - kpts[i]) > theta:
                res_inx_list.append(i)
        kpts = np.asarray(kpts)[res_inx_list]
        nrmls = np.asarray(nrmls)[res_inx_list]
        confs = np.asarray(confs)[res_inx_list]
    if toggledebug:
        for i in range(len(confs)):
            gm.gen_sphere(kpts[i], radius=.005, rgba=[confs[i], 0, 1 - confs[i], .3]).attach_to(base)
            gm.gen_arrow(kpts[i], kpts[i] + kpts_rotseq[i][:, 0] * .02, rgba=[confs[i], 0, 1 - confs[i], .3],
                         thickness=.001).attach_to(base)
    return np.asarray(kpts)[np.argsort(confs)], \
           np.asarray(nrmls)[np.argsort(confs)], \
           np.asarray(confs)[np.argsort(confs)]


def cal_coverage(pcd_partial, pcd_gt, voxel_size=.001, tor=.001, toggledebug=False):
    o3dpcd_gt = o3d_helper.nparray2o3dpcd(pcd_gt)
    downpcd_gt = o3dpcd_gt.voxel_down_sample(voxel_size=voxel_size)
    kdt_partial, _ = get_kdt(pcd_partial)
    cnt = 0
    for p in np.asarray(downpcd_gt.points):
        dist = get_min_dist(p, kdt_partial)
        if dist < tor:
            cnt += 1
        else:
            if toggledebug:
                gm.gen_sphere(p, radius=.001, rgba=(1, 0, 0, .2)).attach_to(base)
    return cnt / len(np.asarray(downpcd_gt.points))


def show_pcd(pcd, rgba=(1, 1, 1, 1)):
    gm.gen_pointcloud(pcd, rgbas=[rgba], pntsize=5).attach_to(base)


def show_pcd_withrgb(pcd, rgbas, show_percentage=1):
    n = int(1 / show_percentage)
    pcd = [p for i, p in enumerate(list(pcd)) if i % n == 0]
    rgbas = [c for i, c in enumerate(list(rgbas)) if i % n == 0]
    if len(rgbas[0]) == 3:
        rgbas = np.hstack((np.asarray(rgbas),
                           np.repeat(1, [len(rgbas)]).reshape((len(rgbas), 1))))
    gm.gen_pointcloud(np.asarray(pcd), rgbas=list(rgbas), pntsize=50).attach_to(base)


def show_pcdseq(pcdseq, rgba=(1, 1, 1, 1), time_sleep=.1):
    def __update(pcldnp, counter, pcdseq, task):
        if counter[0] >= len(pcdseq):
            counter[0] = 0
        if counter[0] < len(pcdseq):
            if pcldnp[0] is not None:
                pcldnp[0].detach()
            pcd = np.asarray(pcdseq[counter[0]])
            pcldnp[0] = gm.gen_pointcloud(pcd, rgbas=[rgba], pntsize=1)
            pcldnp[0].attach_to(base)
            counter[0] += 1
        else:
            counter[0] = 0
        return task.again

    counter = [0]
    pcldnp = [None]
    print(f'num of frames: {len(pcdseq)}')
    taskMgr.doMethodLater(time_sleep, __update, 'update', extraArgs=[pcldnp, counter, np.asarray(pcdseq)],
                          appendTask=True)


def show_pcdseq_withrgb(pcdseq, rgbasseq, time_sleep=.1):
    def __update(pcldnp, counter, pcdseq, rgbasseq, task):
        if counter[0] >= len(pcdseq):
            counter[0] = 0
        if counter[0] < len(pcdseq):
            if pcldnp[0] is not None:
                pcldnp[0].detach()
            pcd = np.asarray(pcdseq[counter[0]])
            rgbas = list(rgbasseq[counter[0]])
            if len(rgbas[0]) == 3:
                rgbas = np.hstack((rgbasseq[counter[0]],
                                   np.repeat(1, [len(pcd)]).reshape((len(pcd), 1))))
            pcldnp[0] = gm.gen_pointcloud(pcd, rgbas=list(rgbas), pntsize=1.5)
            pcldnp[0].attach_to(base)
            counter[0] += 1
        else:
            counter[0] = 0
        return task.again

    counter = [0]
    pcldnp = [None]
    print(f'num of frames: {len(pcdseq)}')
    taskMgr.doMethodLater(time_sleep, __update, 'update',
                          extraArgs=[pcldnp, counter, np.asarray(pcdseq), np.asarray(rgbasseq, dtype=object)],
                          appendTask=True)


def show_pcd_withrbt(pcd, rgba=(1, 1, 1, 1), rbtx=None, toggleendcoord=False):
    from localenv import envloader as el

    rbt = el.loadUr3e()
    env = el.Env_wrs(boundingradius=7.0)
    env.reparentTo(base)

    if rbtx is not None:
        for armname in ["lft_arm", "rgt_arm"]:
            tmprealjnts = rbtx.getjnts(armname)
            print(armname, tmprealjnts)
            rbt.fk(armname, tmprealjnts)

    rbt.gen_meshmodel(toggle_tcpcs=toggleendcoord).attach_to(base)
    gm.gen_pointcloud(pcd, rgbas=[rgba]).attach_to(base)


def show_cam(mat4):
    cam_cm = cm.CollisionModel(os.path.join(config.ROOT, 'obstacles', 'phoxi.stl'))
    cam_cm.set_homomat(mat4)
    cam_cm.set_rgba((.2, .2, .2, 1))
    cam_cm.attach_to(base)
    fov_cm = cm.CollisionModel(os.path.join(config.ROOT, 'obstacles', 'phoxi_fov.stl'))
    fov_cm.set_homomat(mat4)
    fov_cm.set_rgba((.8, .8, .8, .1))
    fov_cm.attach_to(base)
    laser_pos = mat4[:3, 3] - .175 * rm.unit_vector(mat4[:3, 0]) + .049 * rm.unit_vector(mat4[:3, 1]) + \
                .01 * rm.unit_vector(mat4[:3, 2])
    laser_cm = cm.CollisionModel(os.path.join(config.ROOT, 'obstacles', 'phoxi_laser.stl'))
    laser_cm.set_homomat(rm.homomat_from_posrot(laser_pos,
                                                np.dot(rm.rotmat_from_axangle((0, 0, 1), np.radians(22.5)),
                                                       mat4[:3, :3])))
    laser_cm.set_rgba((1, 0, 0, .1))
    laser_cm.attach_to(base)

    gm.gen_frame(mat4[:3, 3], mat4[:3, :3]).attach_to(base)


if __name__ == '__main__':
    from localenv import envloader as el

    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    objcm = el.loadObj("pentip.stl")

    # source_pcd = np.asarray(ts.sample_surface(objcm.objtrm, count=10000))
    # source = o3d_helper.nparray2o3dpcd(source_pcd[source_pcd[:, 2] > 5])
    # source.paint_uniform_color([0, 0.706, 1])
    # o3d.visualization.draw_geometries([source])

    # get_objpcd_partial(objcm, sample_num=10000, toggledebug=True)

    # inithomomat = pickle.load(
    #     open(el.root + "/graspplanner/graspmap/pentip_cover_objmat4_list.pkl", "rb"))[1070]
    #
    # for i, p in enumerate(pcd):
    #     base.pggen.plotArrow(base.render, spos=p, epos=p + 10 * pcd_normals[i])
    # base.run()
    get_objpcd_partial_bycam_pos(objcm, smp_num=10000, toggledebug=True)
    # get_objpcd_partial(objcm, objmat4=np.eye(4), sample_num=10000, toggledebug=True)

    # pcd = pickle.load(open(el.root + "/dataset/pcd/a_lft_0.pkl", "rb"))
    # amat = pickle.load(open(el.root + "/camcalib/data/phoxi_calibmat_0117.pkl", "rb"))
    # pcd = transform_pcd(remove_pcd_zeros(pcd), amat)
    # print(len(pcd))
    # obj = get_org_surface(pcd)
    # base.run()
