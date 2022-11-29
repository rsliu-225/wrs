import copy
import os
import random
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import interpolate
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KDTree

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import modeling.geometric_model as gm

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255

'''
basic func
'''


def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def trans_pos(pts, pos):
    return rm.homomat_transform_points(rm.homomat_from_posrot(-np.asarray(pos), np.eye(3)), pts)


def rot_new_orgin(pts, new_orgin, rot):
    trans_pts = trans_pos(pts, new_orgin)
    trans_pts = rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray([0, 0, 0]), rot), trans_pts)
    return trans_pos(trans_pts, -new_orgin).tolist()


def gen_random_homomat4(trans_diff=(.01, .01, .01), rot_diff=np.radians((10, 10, 10))):
    random_pos = np.asarray([random.uniform(-trans_diff[0], -trans_diff[0]),
                             random.uniform(-trans_diff[1], -trans_diff[1]),
                             random.uniform(-trans_diff[2], -trans_diff[2])])
    if rot_diff is None:
        random_rot = np.eye(3)
    else:
        random_rot = rm.rotmat_from_axangle((1, 0, 0), random.uniform(-rot_diff[0], -rot_diff[0])).dot(
            rm.rotmat_from_axangle((0, 1, 0), random.uniform(-rot_diff[1], -rot_diff[1]))).dot(
            rm.rotmat_from_axangle((0, 0, 1), random.uniform(-rot_diff[2], -rot_diff[2])))
    return rm.homomat_from_posrot(random_pos, random_rot)


def poly_inp(pseq, step=.001, toggledebug=False, kind="cubic"):
    length = np.sum(np.linalg.norm(np.diff(np.asarray(pseq), axis=0), axis=1))
    inp = interpolate.interp1d(pseq[:, 0], pseq[:, 1], kind=kind)
    inp_z = interpolate.interp1d(pseq[:, 0], pseq[:, 2], kind=kind)
    x = np.linspace(0, pseq[-1][0], int(length / step))
    y = inp(x)
    z = inp_z(x)
    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot3D(pseq[:, 0], pseq[:, 1], pseq[:, 2], color='red')
        ax.scatter3D(x, y, z, color='green')
        plt.show()

    return np.asarray(list(zip(x, y, z)))


def spl_inp(pseq, n=200, toggledebug=False):
    pseq = pseq.transpose()
    tck = interpolate.splprep(pseq, k=min([pseq.shape[1] - 1, 5]))[0]
    new = interpolate.splev(np.linspace(0, 1, n), tck, der=0)
    pts = np.asarray(new).transpose()

    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot(pseq[0], pseq[1], pseq[2], label='key points', lw=2, c='Dodgerblue')
        ax.plot(new[0], new[1], new[2], label='fit', lw=2, c='red')
        ax.legend()
        plt.show()
    return pts


def uni_length(pseq, goal_len):
    org_len = np.linalg.norm(np.diff(pseq, axis=0), axis=1).sum()
    return goal_len * np.asarray(pseq) / org_len


def get_rotseq_by_pseq_1d(pseq):
    rotseq = []
    for i in range(1, len(pseq)):
        v1 = pseq[i] - pseq[i - 1]
        y = np.asarray([0, 1, 0])
        n = np.cross(v1, y)
        if rm.angle_between_vectors(n, [0, 0, 1]) > np.pi / 2:
            n = -n
        rot = np.asarray([rm.unit_vector(v1), rm.unit_vector(y), rm.unit_vector(n)]).T
        rotseq.append(rot)
    rotseq = [rotseq[0]] + rotseq
    return np.asarray(rotseq)


def get_rotseq_by_pseq(pseq):
    rotseq = []
    n_pre = None
    kdt = KDTree(pseq, leaf_size=100, metric='euclidean')
    for i in range(1, len(pseq) - 1):
        v = pseq[i - 1] - pseq[i]

        # indices = kdt.query([pseq[i]], k=min([20, len(pseq) / 5]), return_distance=False)
        indices = kdt.query([pseq[i]], k=min([5, len(pseq) / 5]), return_distance=False)
        knn = pseq[indices][0]
        pcv, pcaxmat = rm.compute_pca(knn)
        n = pcaxmat[:, np.argmin(pcv)]

        if n_pre is not None:
            if rm.angle_between_vectors(n, n_pre) > np.pi / 2:
                n = -n
        n_pre = n
        x = np.cross(v, n)
        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v), rm.unit_vector(n)]).T
        rotseq.append(rot)
        # gm.gen_frame(pseq[i], rot, length=.02, thickness=.002).attach_to(base)
    rotseq = [rotseq[0]] + rotseq + [rotseq[-1]]
    return pseq, rotseq


def show_pseq(pseq, rgba=(1, 0, 0, 1), radius=0.0005, show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=radius).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=radius) \
                .attach_to(base)


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


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    import utils.pcd_utils as pcdu
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    kpts = sort_kpts(gmix.means_, seed=np.asarray([0, 0, 0]))

    if show:
        for i, p in enumerate(kpts[1:]):
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)

    kdt, _ = pcdu.get_kdt(objpcd)
    kpts_rotseq = []
    for i, p in enumerate(kpts[:-1]):
        knn = pcdu.get_knn(kpts[i], kdt, k=int(len(objpcd) / n_components))
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

    return kpts, np.asarray(kpts_rotseq)


'''
gen cm
'''


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

    return cm.CollisionModel(initor=objtrm, btwosided=True, name='obj')


'''
gen shape primitive
'''


def random_kpts(n=3, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append(((j + 1) * .02, random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


def random_kpts_sprl(num_kpts, z_max=.04, toggledebug=True):
    cir_num = random.uniform(1, 2)
    # theta_min = random.uniform(-2, 0) * np.pi
    # theta = np.append(np.linspace(theta_min, 0, int(num_kpts / 2)),
    #                   np.linspace(0, theta_min + 2 * cir_num * np.pi, int(num_kpts / 2) + 1)[1:])
    theta = np.linspace(0, 2 * cir_num * np.pi, num_kpts)
    z = np.linspace(0, z_max, num_kpts + random.randint(50, 200))[-num_kpts:]
    r = z
    x = r * np.sin(theta * random.uniform(.75, 1))
    y = r * np.cos(theta * random.uniform(.75, 1))
    if toggledebug:
        ax = plt.axes(projection='3d')
        ax.plot(x, y, z, label='parametric curve')
        ax.legend()
        plt.show()

    return np.asarray(list(zip(x, y, z))) - np.asarray([x[0], y[0], z[0]])


def random_rot_radians(n=3):
    rot_axial = []
    rot_radial = []
    for i in range(n):
        rot_axial.append(random.randint(10, 30) * random.choice([1, -1]))
        rot_radial.append(random.randint(0, 1) * random.choice([1, -1]))
    return np.radians(rot_axial), np.radians(rot_radial)


def gen_seed(num_kpts=4, max=.02, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, rand_wd=False):
    width = width + (np.random.uniform(0, 0.005) if rand_wd else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    flat_sec = [[0, width / 2], [0, -width / 2], [0, -width / 2], [0, width / 2]]
    success = False
    pseq, rotseq = [], []
    while not success:
        if num_kpts <= 5:
            kpts = random_kpts(num_kpts, max=max)
        else:
            kpts = random_kpts_sprl(num_kpts, z_max=max, toggledebug=toggledebug)
        if len(kpts) == 3:
            pseq = uni_length(poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
        elif len(kpts) != n:
            pseq = uni_length(spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
        else:
            pseq = uni_length(kpts, goal_len=length)
        pseq = np.asarray(pseq) - pseq[0]
        pseq, rotseq = get_rotseq_by_pseq(pseq)
        for i in range(len(rotseq) - 1):
            if rm.angle_between_vectors(rotseq[i][:, 2], rotseq[i + 1][:, 2]) > np.pi / 15:
                success = False
                break
            success = True
    # for i in range(len(pseq)):
    #     if i % 10 == 0:
    #         gm.gen_frame(pseq[i], rotseq[i], length=.02, thickness=.002).attach_to(base)
    return gen_swap(pseq, rotseq, cross_sec), gen_swap(pseq, rotseq, flat_sec), pseq, rotseq


'''
deform
'''


def gen_plate_ctr_pts(pts, goal_pseq, edge=0.0):
    org_len = np.linalg.norm(pts[:, 0].max() - pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_diff_uni = org_len * goal_diff / goal_diff.sum()
    # x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), num)
    x = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min()) - edge
    y_max = float(pts[:, 1].max()) + edge
    z_min = float(pts[:, 2].min()) - edge
    z_max = float(pts[:, 2].max()) + edge
    ctr_pts = [[x, y_min, z_min],
               [x, y_max, z_min],
               [x, y_max, z_max],
               [x, y_min, z_max]]
    for i in range(len(goal_diff_uni)):
        x += goal_diff_uni[i]
        ctr_pts.extend([[x, y_min, z_min],
                        [x, y_max, z_min],
                        [x, y_max, z_max],
                        [x, y_min, z_max]])

    return np.asarray(ctr_pts)


def gen_deformed_ctr_pts(ctr_pts, goal_pseq, rot_axial=None, rot_radial=None, show_ctrl_pts=False):
    goal_rotseq = get_rotseq_by_pseq_1d(goal_pseq)
    org_len = np.linalg.norm(ctr_pts[:, 0].max() - ctr_pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_pseq = org_len * goal_pseq / goal_diff.sum()

    org_kpts = ctr_pts.reshape((int(len(ctr_pts) / 4), 4, 3)).mean(axis=1)
    deformed_ctr_pts = []
    if len(ctr_pts) != len(goal_pseq) * 4:
        print('Wrong goal_diff size!', ctr_pts.shape, goal_pseq.shape)
        return None
    for i in range(len(goal_pseq)):
        if show_ctrl_pts:
            gm.gen_frame(goal_pseq[i], goal_rotseq[i], length=.01, thickness=.001).attach_to(base)
            gm.gen_frame(org_kpts[i], np.eye(3), length=.01, thickness=.001,
                         rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        transmat4 = np.dot(rm.homomat_from_posrot(goal_pseq[i], goal_rotseq[i]),
                           np.linalg.inv(rm.homomat_from_posrot(org_kpts[i], np.eye(3))))
        deformed_ctr_pts.extend(trans_pcd(ctr_pts[i * 4:(i + 1) * 4], transmat4))
        if show_ctrl_pts:
            for p in deformed_ctr_pts:
                gm.gen_sphere(p, radius=.001, rgba=(1, 0, 0, 1)).attach_to(base)

    if rot_axial is not None:
        deformed_ctr_pts_rot = []
        for i in range(len(rot_axial)):
            deformed_ctr_pts_rot.extend(rot_new_orgin(deformed_ctr_pts[i * 4:(i + 1) * 4],
                                                      goal_pseq[i],
                                                      rm.rotmat_from_axangle(goal_rotseq[i][:, 0], rot_axial[i])))
        deformed_ctr_pts = np.copy(deformed_ctr_pts_rot)
        if show_ctrl_pts:
            for p in deformed_ctr_pts:
                gm.gen_sphere(p, radius=.001, rgba=(1, 0, 1, 1)).attach_to(base)

    if rot_radial is not None:
        deformed_ctr_pts_rot = []
        for i in range(len(rot_radial)):
            deformed_ctr_pts_rot.extend(rot_new_orgin(deformed_ctr_pts[i * 4:(i + 1) * 4],
                                                      goal_pseq[i],
                                                      rm.rotmat_from_axangle(goal_rotseq[i][:, 1], rot_radial[i])))
        deformed_ctr_pts = np.copy(deformed_ctr_pts_rot)
        if show_ctrl_pts:
            for p in deformed_ctr_pts:
                gm.gen_sphere(p, radius=.001, rgba=(0, 0, 1, 1)).attach_to(base)

    if show_ctrl_pts:
        for p in ctr_pts:
            gm.gen_sphere(p, radius=.001, rgba=(1, 1, 0, 1)).attach_to(base)
        # for p in deformed_ctr_pts:
        #     gm.gen_sphere(p, radius=.001, rgba=(1, 0, 0, 1)).attach_to(base)
    return np.asarray(deformed_ctr_pts)


def deform_cm(objcm, goal_kpts, rot_axial, rot_radial, width=.008, thickness=0, rbf_radius=.05, show=False):
    from pygem import RBF
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    success = False
    objcm_deformed, objcm_gt = None, None
    while not success:
        vs = objcm.objtrm.vertices
        org_ctr_pts = gen_plate_ctr_pts(vs, goal_kpts)
        deformed_ctr_pts = gen_deformed_ctr_pts(org_ctr_pts, goal_kpts,
                                                rot_axial=rot_axial, rot_radial=rot_radial, show_ctrl_pts=show)
        rbf = RBF(original_control_points=org_ctr_pts, deformed_control_points=deformed_ctr_pts, radius=rbf_radius)
        new_vs = rbf(vs)
        objcm_deformed = cm.CollisionModel(initor=trm.Trimesh(vertices=np.asarray(new_vs), faces=objcm.objtrm.faces),
                                           btwosided=True, name='plate_deform')
        new_pts, _ = objcm_deformed.sample_surface(radius=.0005)

        kpts, kpts_rotseq = get_kpts_gmm(new_pts, rgba=(1, 1, 0, 1), n_components=16, show=False)
        objcm_gt = gen_swap(kpts, kpts_rotseq, cross_sec, toggledebug=False)
        for i in range(len(kpts_rotseq) - 1):
            if rm.angle_between_vectors(kpts_rotseq[i][:, 1], kpts_rotseq[i + 1][:, 1]) > np.pi / 3:
                success = False
                break
            success = True
    if show:
        for i, rot in enumerate(kpts_rotseq):
            gm.gen_frame(kpts[i], kpts_rotseq[i], thickness=.001, length=.02).attach_to(base)

        gm.gen_pointcloud(new_pts).attach_to(base)
        # objcm_deformed.set_rgba((.7, .7, 0, 1))
        # objcm_deformed.attach_to(base)
        objcm_gt.set_rgba((1, 1, 0, 1))
        objcm_gt.attach_to(base)

    return objcm_deformed, objcm_gt, kpts, kpts_rotseq


'''
adaptor
'''


def o3dmesh2cm(o3dmesh):
    objtrm = trm.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles)
    objcm = cm.CollisionModel(objtrm)
    return objcm


def cm2o3dmesh(objcm, wnormal=False):
    o3dmesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                        triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))
    if wnormal:
        o3dmesh.compute_vertex_normals()
    return o3dmesh


def nparray2o3dpcd(pcd_narray, nrmls_narry=None, estimate_normals=False):
    pcd_narray = np.asarray(pcd_narray)
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(pcd_narray[:, :3])
    if nrmls_narry is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nrmls_narry[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


'''
gen pcd
'''


def get_objpcd_partial_rh(objcm, img_size=(50, 50), granurity=0.2645 / 1000):
    pcd = []
    mask = np.zeros(img_size)

    gm.gen_stick(spos=np.asarray((0, 0, 0)), epos=np.asarray((img_size[0] * granurity, 0, 0)),
                 rgba=(1, 0, 0, 1), thickness=.001).attach_to(base)
    gm.gen_stick(spos=np.asarray((0, 0, 0)), epos=np.asarray((0, img_size[1] * granurity, 0)),
                 rgba=(0, 1, 0, 1), thickness=.001).attach_to(base)
    for x in range(img_size[0]):
        for y in range(img_size[1]):
            pf = np.asarray([x * granurity, y * granurity, 1])
            pt = np.asarray([x * granurity, y * granurity, -1])
            try:
                v, _ = objcm.ray_hit(pf, pt, option="closest")
                pcd.append(v)
                mask[x, y] = 1
            except:
                continue
    cv2.imshow('mask', np.asarray(mask))
    cv2.waitKey(0)
    return np.asarray(pcd), mask


def get_objpcd_partial_sample(objcm, objmat4=np.eye(4), smp_num=100000, cam_pos=np.array([.86, .08, 1.78]),
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
        gm.gen_pointcloud(objpcd_partial).attach_to(base)

    return objpcd_partial


def cal_kpts_conf(o3dpcd_i, kpts, kpts_rotseq, tor=.5, radius=.005):
    conf = []
    colors = []
    kdt_i = o3d.geometry.KDTreeFlann(o3dpcd_i)
    pcd_i = np.asarray(o3dpcd_i.points)
    for p in np.asarray(kpts):
        k, _, _ = kdt_i.search_radius_vector_3d(p, radius)
        # print(k, len(pcd_i) / len(kpts))
        if k < (len(pcd_i) / len(kpts)) * tor:
            conf.append(0)
            colors.append([1, 0, 0, .3])
        else:
            conf.append(1)
            colors.append([0, 1, 0, .3])

    return [kpts, kpts_rotseq, conf]


def get_objpcd_partial_o3d(objcm, objcm_gt, rot, rot_center, pseq=None, rotseq=None,
                           path='./', f_name='', resolusion=(1280, 720), ext_name='.pcd',
                           rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6),
                           occ_vt_ratio=1, noise_vt_ratio=1, noise_cnt=random.randint(0, 5),
                           visible_threshold=np.pi / 3,
                           add_noise=False, add_occ=False, add_rnd_occ=True, add_noise_pts=True,
                           savemesh=False, savedepthimg=False, savergbimg=False, savekpts=True, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'partial/')):
        os.mkdir(os.path.join(path, 'partial/'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    o3dmesh = cm2o3dmesh(objcm, wnormal=True)
    o3dmesh_gt = cm2o3dmesh(objcm_gt, wnormal=False)
    o3dmesh.rotate(rot, center=rot_center)
    o3dmesh_gt.rotate(rot, center=rot_center)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f_name + f'_tmp{ext_name}'), do_render=False,
                                  convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f_name + f'_tmp{ext_name}'))
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    o3dpcd_nrml = np.asarray(o3dpcd.normals)
    try:
        vis_idx = np.argwhere(np.arccos(abs(o3dpcd_nrml.dot(np.asarray([0, 0, 1])))) < visible_threshold).flatten()
        # print(len(o3dpcd_nrml), len(vis_idx), vis_idx)
        o3dpcd = o3dpcd.select_by_index(vis_idx)
        o3dpcd_org = copy.deepcopy(o3dpcd)

        if add_rnd_occ:
            o3dpcd = add_random_occ(o3dpcd, occ_ratio_rng=rnd_occ_ratio_rng)
            o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'), o3dpcd)
        if add_occ:
            o3dpcd = add_random_occ_by_nrml(o3dpcd, occ_ratio_rng=nrml_occ_ratio_rng)
            o3dpcd = add_random_occ_by_vt(o3dpcd, np.asarray(o3dmesh.vertices),
                                          edg_radius=5e-4, edg_sigma=5e-4, ratio=occ_vt_ratio)
            o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'), o3dpcd)
        if add_noise:
            o3dpcd = add_guassian_noise_by_vt(o3dpcd, np.asarray(o3dmesh.vertices), np.asarray(o3dmesh.vertex_normals),
                                              noise_mean=1e-3, noise_sigma=1e-4, ratio=noise_vt_ratio)
            o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'), o3dpcd)
        if add_noise_pts:
            o3dpcd = add_noise_pts_by_vt(o3dpcd, noise_cnt=noise_cnt, size=.01)
            o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'), o3dpcd)

        o3dpcd = resample(o3dpcd, smp_num=2048)
        o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'), o3dpcd)
        save_complete_pcd(f_name, o3dmesh_gt, path=path, method='possion', smp_num=2048)

        if savemesh:
            if not os.path.exists(os.path.join(path, 'mesh/')):
                os.mkdir(os.path.join(path, 'mesh/'))
            o3d.io.write_triangle_mesh(os.path.join(path, 'mesh', f'{f_name}.ply'), o3dmesh)
        if savergbimg:
            if not os.path.exists(os.path.join(path, 'rgbimg/')):
                os.mkdir(os.path.join(path, 'rgbimg/'))
            vis.capture_screen_image(os.path.join(path, 'rgbimg', f'{f_name}.jpg'), do_render=False)
        if savedepthimg:
            if not os.path.exists(os.path.join(path, 'depthimg/')):
                os.mkdir(os.path.join(path, 'depthimg/'))
            depthimg = np.asarray(vis.capture_depth_float_buffer()) * 1000
            cv2.imwrite(os.path.join(path, 'depthimg', f'{f_name}.jpg'), depthimg)
        if savekpts:
            if not os.path.exists(os.path.join(path, 'kpts/')):
                os.mkdir(os.path.join(path, 'kpts/'))
            if pseq is None or rotseq is None:
                o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(path, 'complete', f'{f_name}{ext_name}'))
                pseq, rotseq = get_kpts_gmm(np.asarray(o3dpcd_gt.points), n_components=16, show=False)
            else:
                pseq = trans_pcd(pseq, rm.homomat_from_posrot(pos=rot_center, rot=rot))
                rotseq = [np.dot(rot, r) for r in rotseq]
            pseq, rotseq, conf = cal_kpts_conf(o3dpcd, pseq, rotseq, tor=.5, radius=.005)
            # pickle.dump([pseq, rotseq], open(os.path.join(path, 'kpts', f_name + '.pkl'), 'wb'))
            pickle.dump([pseq, rotseq, conf], open(os.path.join(path, 'kpts', f'{f_name}.pkl'), 'wb'))
        vis.destroy_window()
    except:
        vis.destroy_window()
        return 0

    if toggledebug:
        o3dpcd = o3d.io.read_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'))
        o3dpcd_gt = o3d.io.read_point_cloud(os.path.join(path, 'complete', f'{f_name}{ext_name}'))
        o3dpcd_org = nparray2o3dpcd(np.asarray(o3dpcd_org.points))
        o3dpcd_org.paint_uniform_color(COLOR[0])
        o3dpcd.paint_uniform_color(COLOR[0])
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3d.visualization.draw_geometries([o3dmesh])
        o3d.visualization.draw_geometries([o3dpcd_gt])
        o3d.visualization.draw_geometries([o3dpcd_org])
        o3d.visualization.draw_geometries([o3dpcd])
        print(len(o3dpcd.points), len(o3dpcd_gt.points))
    os.remove(os.path.join(path, f_name + f'_tmp{ext_name}'))
    return 1


def get_objpcd_partial_o3d_vctrl(objcm, path='./', f_name='', resolusion=(1280, 720), ext_name='.pcd',
                                 occ_vt_ratio=1, noise_vt_ration=1, add_noise=False, add_occ=False, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'partial/')):
        os.mkdir(os.path.join(path, 'partial/'))
    if not os.path.exists(os.path.join(path, 'mesh/')):
        os.mkdir(os.path.join(path, 'mesh/'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    o3dmesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                        triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))
    o3dmesh.compute_vertex_normals()
    vis.add_geometry(o3dmesh)

    ctr = o3d.visualization.ViewControl()
    vis.get_render_option().load_from_json("./renderoption.json")

    init_param = ctr.convert_to_pinhole_camera_parameters()
    w, h = 4000, 3000
    K = np.asarray([[0.744375, 0.0, 0.0],
                    [0.0, 0.744375, 0.0],
                    [0.4255, 0.2395, 1.0]])
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    init_param.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(init_param)
    vis.poll_events()

    ctr.rotate(10, 0)
    image = vis.capture_screen_float_buffer()
    cv2.imshow('', cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)

    vis.capture_depth_point_cloud(os.path.join(path, f_name + f'_partial_org{ext_name}'), do_render=False,
                                  convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f_name + f'_partial_org{ext_name}'))
    if add_occ:
        o3dpcd = add_random_occ_by_nrml(o3dpcd, occ_ratio_rng=(.3, .6))
        o3dpcd = add_random_occ_by_vt(o3dpcd, np.asarray(o3dmesh.vertices),
                                      edg_radius=1e-3, edg_sigma=3e-4, ratio=occ_vt_ratio)
        o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}_partial{ext_name}'), o3dpcd)
    if add_noise:
        o3dpcd = add_guassian_noise_by_vt(o3dpcd, np.asarray(o3dmesh.vertices), np.asarray(o3dmesh.vertex_normals),
                                          noise_mean=1e-4, noise_sigma=1e-4, ratio=noise_vt_ration)
        o3d.io.write_point_cloud(os.path.join(path, 'partial', f'{f_name}_partial{ext_name}'), o3dpcd)
    o3d.io.write_triangle_mesh(os.path.join(path, 'mesh', f_name + '.ply'), o3dmesh)

    # vis.capture_screen_image(os.path.join(path, f_name + '.png'), do_render=False)
    save_complete_pcd(f_name, o3dmesh, path=path)
    vis.destroy_window()
    # o3d.visualization.draw_geometries([o3dpcd], point_show_normal=True)
    if toggledebug:
        o3dpcd_org = o3d.io.read_point_cloud(os.path.join(path, f_name + f'_partial_org{ext_name}'))
        o3dpcd = o3d.io.read_point_cloud(os.path.join(path, 'partial', f'{f_name}{ext_name}'))
        o3dpcd_org.paint_uniform_color([0, 0.706, 1])
        o3dpcd.paint_uniform_color([0.706, 0, 1])
        # print(o3dpcd_org.get_center())
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_org])
    os.remove(os.path.join(path, f_name + f'_partial_org{ext_name}'))
    return o3dpcd


def get_objpcd_full_sample(objcm, radius=.0005, smp_num=100000, toggledebug=False):
    objpcd, _ = objcm.sample_surface(radius=radius, nsample=smp_num)
    print("Length of org pcd", len(objpcd))

    if toggledebug:
        gm.gen_pointcloud(objpcd).attach_to(base)

    return objpcd


def get_objpcd_full_sample_o3d(o3dmesh, smp_num=16384, toggledebug=False, method='uniform'):
    if method == 'uniform':
        o3dpcd = o3dmesh.sample_points_uniformly(number_of_points=smp_num)
    else:
        o3dpcd = o3dmesh.sample_points_poisson_disk(number_of_points=smp_num)
    if toggledebug:
        gm.gen_pointcloud(o3dpcd.points).attach_to(base)
    return o3dpcd


'''
pre-processing
'''


def noisy_mesh(o3dmesh, noise=.0005):
    print('create noisy mesh')
    vertices = np.asarray(o3dmesh.vertices)
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    o3dmesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3dmesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([o3dmesh])
    return o3dmesh


def add_random_occ(o3dpcd, occ_ratio_rng=(.05, .1)):
    kdt = KDTree(np.asarray(o3dpcd.points), leaf_size=100, metric='euclidean')
    seed = random.choice(o3dpcd.points)
    _, indices = kdt.query([seed], k=int(len(o3dpcd.points) * random.uniform(occ_ratio_rng[0], occ_ratio_rng[1])),
                           return_distance=True)
    pcd = np.delete(np.asarray(o3dpcd.points), indices[0], axis=0)
    return o3dh.nparray2o3dpcd(pcd)


def add_random_occ_narry(pcd, occ_ratio_rng=(.05, .1)):
    kdt = KDTree(pcd, leaf_size=100, metric='euclidean')
    seed = random.choice(pcd)
    _, indices = kdt.query([seed], k=int(len(pcd) * random.uniform(occ_ratio_rng[0], occ_ratio_rng[1])),
                           return_distance=True)
    pcd = np.delete(np.asarray(pcd), indices[0], axis=0)
    return pcd


def add_random_occ_by_nrml(o3dpcd, occ_ratio_rng=(.3, .6)):
    kdt = KDTree(np.asarray(o3dpcd.points), leaf_size=100, metric='euclidean')
    seed = random.choice(range(len(o3dpcd.points)))
    _, indices = kdt.query([np.asarray(o3dpcd.points)[seed]],
                           k=int(len(o3dpcd.points) * random.uniform(occ_ratio_rng[0], occ_ratio_rng[1])),
                           return_distance=True)
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))

    nrmls = np.asarray(o3dpcd.normals)[indices[0]]
    nrml_0 = np.asarray(o3dpcd.normals)[seed]

    del_indices = []
    for i, v in enumerate(nrmls):
        a = rm.angle_between_vectors(v, nrml_0)
        if a < np.random.normal(np.pi / 2, np.pi / 18):
            del_indices.append(indices[0][i])
    pcd = np.delete(np.asarray(o3dpcd.points), del_indices, axis=0)

    return o3dh.nparray2o3dpcd(pcd)


def add_random_occ_by_vt(o3dpcd, vts, edg_radius=1e-3, edg_sigma=3e-4, ratio=1.0):
    kdt = KDTree(np.asarray(o3dpcd.points), leaf_size=100, metric='euclidean')
    seeds = random.choices(vts, k=random.choice(range(int(len(vts) * ratio))))
    del_indices = []
    for seed in seeds:
        indices, dists = kdt.query_radius(np.asarray([seed]),
                                          r=random.uniform(0, edg_radius + edg_sigma),
                                          return_distance=True, count_only=False)
        if len(indices[0]) == 0:
            continue
        pts = np.asarray(o3dpcd.points)[indices[0]]
        for i, p in enumerate(pts):
            l = np.linalg.norm(p - seed)
            if l < np.random.normal(np.mean(dists[0]), np.mean(dists[0]) / 5):
                del_indices.append(indices[0][i])
    pcd = np.delete(np.asarray(o3dpcd.points), del_indices, axis=0)

    return o3dh.nparray2o3dpcd(pcd)


def add_guassian_noise_by_vt(o3dpcd, vts, nrmls, noise_mean=1e-4, noise_sigma=1e-4, ratio=1.0):
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    pcd = np.asarray(o3dpcd.points)
    kdt = KDTree(pcd, leaf_size=100, metric='euclidean')
    # seeds = random.choices(vts, k=random.choice(range(len(vts))))
    inx_seeds = random.choices(range(len(vts)), k=random.choice(range(int(len(vts) * ratio))))
    diff = np.zeros(pcd.shape)
    for inx in inx_seeds:
        if len(pcd) > 100:
            dists, indices = kdt.query(np.asarray([vts[inx]]), k=random.randint(100, min(len(pcd), 500)),
                                       return_distance=True)
        else:
            dists, indices = kdt.query(np.asarray([vts[inx]]), k=random.randint(1, len(pcd)),
                                       return_distance=True)
        if len(indices[0]) == 0:
            continue
        dist_inv = (1 / dists[0]) / np.linalg.norm((1 / dists[0]))
        noise = np.repeat(np.random.normal(dist_inv * noise_mean, noise_sigma), 3).reshape(len(dists[0]), 3)
        diff[indices[0]] = nrmls[inx] * noise

    pcd = pcd + diff

    return o3dh.nparray2o3dpcd(pcd)


def add_noise_pts_by_vt(o3dpcd, noise_cnt=3, size=.01):
    if noise_cnt == 0:
        return o3dpcd
    for p in random.choices(np.asarray(o3dpcd.points), k=noise_cnt):
        p = np.asarray(p)
        vts_n = [
            p,
            p + np.asarray([random.uniform(-size, size), random.uniform(-size, size), random.uniform(-size, size)]),
            p + np.asarray([random.uniform(-size, size), random.uniform(-size, size), random.uniform(-size, size)])]
        tmp_gm = gm.GeometricModel(initor=trm.Trimesh(vertices=np.asarray(vts_n),
                                                      faces=np.asarray([[0, 1, 2]])), btwosided=False)
        o3dpcd += nparray2o3dpcd(tmp_gm.sample_surface(radius=random.uniform(.001, .002))[0])

    return o3dpcd


def resample(o3dpcd, smp_num=8192):
    pcd = list(np.asarray(o3dpcd.points))
    # print('Input length:', len(pcd))
    while len(pcd) != smp_num:
        pcd = list(np.asarray(o3dpcd.points))
        if len(pcd) > smp_num:
            if int(len(pcd) / smp_num) > 1:
                o3dpcd = o3dpcd.uniform_down_sample(int(len(pcd) / smp_num))
            o3dpcd = o3dpcd.random_down_sample(smp_num / len(np.asarray(o3dpcd.points)))
        else:
            remain = smp_num % len(pcd)
            for i in range(int(smp_num / len(pcd)) - 1):
                pcd = pcd + pcd
            pcd = pcd + pcd[:remain]
            o3dpcd = o3dh.nparray2o3dpcd(np.asarray(pcd))

    return o3dpcd


'''
view control
'''


def custom_draw_geometry_with_rotation(geo):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([geo], rotate_view)


'''
file io
'''


# Input: pcd file path
def show_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])
    return len(pcd.points)


# Ouput pcd in Numpy Array
def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def read_o3dpcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


# Input is Numpy Array
def show_pcd_pts(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def get_uniq_id(name, fact):
    parts = name.split("_")
    return int(parts[0]) * fact + int(parts[1])


def save_complete_pcd(name, mesh, path="./", method='uniform', smp_num=16384):
    path = os.path.join(path, 'complete/')
    if not os.path.exists(path):
        os.mkdir(path)
    exist = False
    # for file in os.listdir(path):
    #     try:
    #         if name.split("_")[0] == file.split("_")[0] and name.split("_")[2] == file.split("_")[2]:
    #             exist = True
    #     except:
    #         continue
    # if not exist:
    o3d.io.write_point_cloud(path + name + '.pcd', get_objpcd_full_sample_o3d(mesh, method=method, smp_num=smp_num))
