import copy
import math
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy import interpolate
from sklearn.neighbors import KDTree
from sklearn.mixture import GaussianMixture

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import modeling.geometric_model as gm


def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def trans_pos(pts, pos):
    return rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray(pos), np.eye(3)), pts)


def rot_new_orgin(pts, new_orgin, rot):
    trans_pts = trans_pos(pts, new_orgin)
    trans_pts = rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray([0, 0, 0]), rot), trans_pts)
    return trans_pos(trans_pts, -new_orgin).tolist()


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
    for i in range(1, len(pseq) - 1):
        v1 = pseq[i - 1] - pseq[i]
        v2 = pseq[i] - pseq[i + 1]
        n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
        if n_pre is not None:
            if rm.angle_between_vectors(n, n_pre) > np.pi / 2:
                n = -n
        n_pre = n
        x = np.cross(v1, n)

        rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(n)]).T
        rotseq.append(rot)
    rotseq = [rotseq[0]] + rotseq + [rotseq[-1]]
    return rotseq


def show_pseq(pseq, rgba=(1, 0, 0, 1), radius=0.0005, show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=radius).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=radius) \
                .attach_to(base)


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    if show:
        for p in gmix.means_:
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)


'''
adaptor
'''


def o3dmesh2cm(o3dmesh):
    objtrm = trm.Trimesh(vertices=o3dmesh.vertices, faces=o3dmesh.triangles)
    objcm = cm.CollisionModel(objtrm)
    return objcm


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


def get_objpcd_partial_o3d(objcm, rot, rot_center, path='./', f_name='', resolusion=(1280, 720), toggledebug=False,
                           add_noise=False, add_occ=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'partial/')):
        os.mkdir(os.path.join(path, 'partial/'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    ctr = o3d.visualization.ViewControl()
    o3dmesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                        triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))
    o3dmesh.rotate(rot, center=rot_center)
    if add_noise:
        o3dmesh = noisy_mesh(o3dmesh)
    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f_name + '_partial_org.pcd'), do_render=False,
                                  convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f_name + '_partial_org.pcd'))
    if add_occ:
        o3dpcd = add_random_occ_by_nrml(o3dpcd)
        o3d.io.write_point_cloud(os.path.join(path, 'partial', f_name + '_partial.pcd'), o3dpcd)
    # o3d.io.write_triangle_mesh(os.path.join(path, f_name + '.ply'), o3dmesh)
    # vis.capture_screen_image(os.path.join(path, f_name + '.png'), do_render=False)
    save_complete_pcd(f_name, o3dmesh, path=path)
    vis.destroy_window()
    # o3d.visualization.draw_geometries([o3dpcd], point_show_normal=True)
    if toggledebug:
        o3dpcd_org = o3d.io.read_point_cloud(os.path.join(path, f_name + '_partial_org.pcd'))
        o3dpcd = o3d.io.read_point_cloud(os.path.join(path, 'partial', f_name + '_partial.pcd'))
        o3dpcd_org.paint_uniform_color([0, 0.706, 1])
        o3dpcd.paint_uniform_color([0, 0.706, 1])
        # print(o3dpcd_org.get_center())
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_org])
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
    # pts = np.asarray(o3dpcd.points)[indices[0]]
    # pcv, pcaxmat = rm.compute_pca(pts)
    # inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    # nrml_0 = pcaxmat[:, inx[0]]
    nrml_0 = np.asarray(o3dpcd.normals)[seed]

    del_indices = []
    for i, v in enumerate(nrmls):
        a = rm.angle_between_vectors(v, nrml_0)
        if a < np.pi / 2:
            del_indices.append(indices[0][i])
    pcd = np.delete(np.asarray(o3dpcd.points), del_indices, axis=0)

    return o3dh.nparray2o3dpcd(pcd)


'''
file io
'''


def save_complete_pcd(name, mesh, path="./"):
    path = os.path.join(path, 'complete/')
    if not os.path.exists(path):
        os.mkdir(path)
    o3d.io.write_point_cloud(path + name + '_complete.pcd', get_objpcd_full_sample_o3d(mesh))
