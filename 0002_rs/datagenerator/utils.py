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
    return rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray(pos), np.eye(3)), pts)


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


def cubic_inp(pseq, step=.001, toggledebug=False, kind="cubic"):
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


def get_objpcd_partial_o3d(objcm, rot, rot_center, path='./', f_name='', resolusion=(1280, 720), ext_name='.pcd',
                           occ_vt_ratio=1, noise_vt_ration=1, add_noise=False, add_occ=False, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, 'partial/')):
        os.mkdir(os.path.join(path, 'partial/'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    ctr = o3d.visualization.ViewControl()
    o3dmesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                        triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))
    o3dmesh.compute_vertex_normals()
    o3dmesh.rotate(rot, center=rot_center)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
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
    # o3d.io.write_triangle_mesh(os.path.join(path, f_name + '.ply'), o3dmesh)
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
        o3d.visualization.draw_geometries([o3dpcd])
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
        dists, indices = kdt.query(np.asarray([vts[inx]]), k=random.randint(100, min(len(pcd),500)), return_distance=True)
        if len(indices[0]) == 0:
            continue
        dist_inv = (1 / dists[0]) / np.linalg.norm((1 / dists[0]))
        noise = np.repeat(np.random.normal(dist_inv * noise_mean, noise_sigma), 3).reshape(len(dists[0]), 3)
        diff[indices[0]] = nrmls[inx] * noise

    pcd = pcd + diff

    return o3dh.nparray2o3dpcd(pcd)


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


# Input is Numpy Array
def show_pcd_pts(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def get_uniq_id(name, fact):
    parts = name.split("_")
    return int(parts[0]) * fact + int(parts[1])


def save_complete_pcd(name, mesh, path="./"):
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
    o3d.io.write_point_cloud(path + name + '_complete.pcd', get_objpcd_full_sample_o3d(mesh))
