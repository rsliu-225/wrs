import copy
import itertools
import math
import random
import time

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn import linear_model

import modeling.collision_model as cm
import modeling.geometric_model as gm
import basis.trimesh.sample as ts
import utils.math_utils as mu
import basis.robot_math as rm
import basis.o3dhelper as o3d_helper
from basis import trimesh
from localenv import envloader as el
from sklearn.neighbors import KDTree


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
    p_list = np.asarray(p_list)
    p_narray = np.array(p_list[:, :dimension])
    kdt = KDTree(p_narray, leaf_size=100, metric='euclidean')
    print('time cost(kdt):', time.time() - time_start)
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
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def trans_p(p, transmat=None):
    if transmat is None:
        return p
    return trans_pcd(np.asarray([p]), transmat)[0]


def show_pcd(pcd, rgba=(1, 1, 1, 1)):
    gm.gen_pointcloud(pcd, rgbas=[rgba]).attach_to(base)


def show_pcd_withrgb(pcd, rgbas, show_percentage=1):
    n = int(1 / show_percentage)
    pcd = [p for i, p in enumerate(list(pcd)) if i % n == 0]
    rgbas = [c for i, c in enumerate(list(rgbas)) if i % n == 0]
    if len(rgbas[0]) == 3:
        rgbas = np.hstack((np.asarray(rgbas),
                           np.repeat(1, [len(rgbas)]).reshape((len(rgbas), 1))))
    gm.gen_pointcloud(np.asarray(pcd), rgbas=list(rgbas), pntsize=2).attach_to(base)


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
    rbt = el.loadUr3e()
    env = el.Env_wrs(boundingradius=7.0)
    env.reparentTo(base)

    if rbtx is not None:
        for armname in ["lft", "rgt"]:
            tmprealjnts = rbtx.getjnts(armname)
            print(armname, tmprealjnts)
            rbt.movearmfk(tmprealjnts, armname)

    rbt.gen_meshmodel(toggle_tcpcs=toggleendcoord).attach_to(base)
    gm.gen_pointcloud(pcd, rgbas=[rgba]).attach_to(base)


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
    pcd = trans_pcd(pcd, rm.homomat_from_posrot((0, 0, 0), rm.rodrigues((0, 0, 1), -rot_angle)))

    convexhull = trimesh.Trimesh(vertices=pcd)
    convexhull = convexhull.convex_hull
    obj = cm.CollisionModel(initor=convexhull)
    obj_w, obj_h = get_pcd_w_h(pcd)

    if origin == "tip":
        tip = get_pcd_tip(pcd, axis=0)
        origin_pos = center + \
                     trans_pcd(np.array([tip]), rm.homomat_from_posrot((0, 0, 0), rm.rodrigues((0, 0, 1), rot_angle)))[
                         0]
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
                base.pggen.plotArrow(base.render, spos=objpcd[i], epos=objpcd[i] + 10 * n)
                base.pggen.plotSphere(base.render, pos=objpcd[i], rgba=(1, 0, 0, 1))
        base.run()

    return objpcd, np.asarray(objpcd_nrmls)


def get_objpcd_partial(objcm, objmat4=np.eye(4), sample_num=100000, toggledebug=False):
    objpcd = np.asarray(ts.sample_surface(objcm.objtrm, count=sample_num))
    objpcd = trans_pcd(objpcd, objmat4)

    grid = {}
    for p in objpcd:
        x = round(p[0], 0)
        y = round(p[1], 0)
        if str((x, y)) in grid.keys():
            grid[str((x, y))].append(p)
        else:
            grid[str((x, y))] = [p]
    objpcd_new = []
    for k, v in grid.items():
        z_max = max(np.array(v)[:, 2])
        for p in v:
            objpcd_new.append([p[0], p[1], z_max])
    objpcd_new = np.array(objpcd_new)

    print("Length of org pcd", len(objpcd))
    print("Length of partial pcd", len(objpcd_new))

    if toggledebug:
        objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        objpcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([objpcd])

        objpcd_partial = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd_new))
        objpcd_partial.paint_uniform_color([0, 0.706, 1])
        o3d.visualization.draw_geometries([objpcd_partial])

        # pcddnp = base.pg.genpointcloudnp(objpcd)
        # pcddnp.reparentTo(base.render)

    return objpcd_new


def get_objpcd_partial_bycampos(objcm, objmat4=np.eye(4), sample_num=100000, cam_pos=np.array([860, 80, 1780]),
                                toggledebug=False):
    def __get_angle(x, y):
        lx = np.sqrt(x.dot(x))
        ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (lx * ly)
        angle = np.arccos(cos_angle)
        return angle * 360 / 2 / np.pi
        # return cos_angle

    def sigmoid(angle):
        return 1 / (1 + np.exp((angle - 90) / 90)) - 0.5

    objpcd = np.asarray(ts.sample_surface(objcm.objtrm, count=sample_num))
    objpcd_center = get_pcd_center(objpcd)
    face_num = len(objcm.objtrm.face_normals)

    objpcd_new = []
    area_list = objcm.objtrm.area_faces
    area_sum = sum(area_list)

    for i, n in enumerate(objcm.objtrm.face_normals):
        n = np.dot(n, objmat4[:3, :3])
        angle = __get_angle(n, np.array(cam_pos - objpcd_center))

        if angle > 90:
            continue
        else:
            objcm_temp = copy.deepcopy(objcm)
            # print(i, angle, sigmoid(angle))
            mask_temp = [False] * face_num
            mask_temp[i] = True
            objcm_temp.trimesh.update_faces(mask_temp)
            objpcd_new.extend(np.asarray(ts.sample_surface(objcm_temp.trimesh,
                                                           count=int(sample_num / area_sum * area_list[i] * sigmoid(
                                                               angle) * 100))))
    if len(objpcd_new) > sample_num:
        objpcd_new = random.sample(objpcd_new, sample_num)
    objpcd_new = np.array(objpcd_new)
    objpcd_new = trans_pcd(objpcd_new, objmat4)

    # print("Length of org pcd", len(objpcd))
    # print("Length of source pcd", len(objpcd_new))

    if toggledebug:
        objpcd = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd))
        objpcd.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([objpcd])

        objpcd_partial = o3d_helper.nparray2o3dpcd(copy.deepcopy(objpcd_new))
        objpcd_partial.paint_uniform_color([0, 0.706, 1])
        o3d.visualization.draw_geometries([objpcd_partial])

        # pcddnp = base.pg.genpointcloudnp(objpcd)
        # pcddnp.reparentTo(base.render)

    return objpcd_new


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


# def surface_interp(p, v, kdt_d3, inp=0.0005, max_nn=100):
#     pseq = []
#     rotseq = []
#     times = int(np.linalg.norm(v) / inp)
#     # v = np.asarray([v[0], v[1], 0])
#     knn = get_knn(p, kdt_d3, k=max_nn)
#     n = get_nrml_pca(knn)
#
#     for _ in range(times):
#         rotmat = rm.rotmat_between_vectors(np.asarray([0, 0, 1]), n)
#         v_cur = rm.unit_vector(np.dot(rotmat, v))
#         pt = np.asarray(p) + v_cur * inp
#         knn = get_knn(pt, kdt_d3, k=max_nn)
#         center = get_pcd_center(np.asarray(knn))
#         n_cur = get_nrml_pca(knn)
#         p_cur = pt - np.dot((pt - center), n_cur) * n_cur
#         if np.dot(n_cur, np.asarray([0, 0, 1])) < 0:
#             n_cur = -n_cur
#         pseq.append(p_cur)
#         rot = np.asarray([rm.unit_vector(n_cur), -rm.unit_vector(np.cross(n_cur, np.cross(n_cur, v_cur))),
#                           rm.unit_vector(np.cross(n_cur, v_cur))]).T
#         rotseq.append(rot)
#         p = p_cur
#         n = n_cur
#     return pseq, rotseq

def surface_interp(p, v, kdt_d3, inp=0.0005, max_nn=100):
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


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    objcm = el.loadObj("pentip.stl")

    source_pcd = np.asarray(ts.sample_surface(objcm.objtrm, count=10000))
    source = o3d_helper.nparray2o3dpcd(source_pcd[source_pcd[:, 2] > 5])
    # source.paint_uniform_color([0, 0.706, 1])
    # o3d.visualization.draw_geometries([source])

    # get_objpcd_partial(objcm, sample_num=10000, toggledebug=True)

    # inithomomat = pickle.load(
    #     open(el.root + "/graspplanner/graspmap/pentip_cover_objmat4_list.pkl", "rb"))[1070]
    #
    # get_normals(get_objpcd(objcm, sample_num=10000))
    # pcd, pcd_normals = get_objpcd_withnormals(objcm, sample_num=100000)
    # for i, p in enumerate(pcd):
    #     base.pggen.plotArrow(base.render, spos=p, epos=p + 10 * pcd_normals[i])
    # base.run()
    get_objpcd_partial_bycampos(objcm, sample_num=10000, toggledebug=True)

    # pcd = pickle.load(open(el.root + "/dataset/pcd/a_lft_0.pkl", "rb"))
    # amat = pickle.load(open(el.root + "/camcalib/data/phoxi_calibmat_0117.pkl", "rb"))
    # pcd = transform_pcd(remove_pcd_zeros(pcd), amat)
    # print(len(pcd))
    # obj = get_org_surface(pcd)
    # base.run()
