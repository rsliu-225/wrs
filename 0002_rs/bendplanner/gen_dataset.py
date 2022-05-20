import math
import random

import cv2
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
import copy


def gen_sgl_curve(pseq, step=.001, toggledebug=False):
    length = np.sum(np.linalg.norm(np.diff(np.asarray(pseq), axis=0), axis=1))
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

    return np.asarray(list(zip(x, y, z)))


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
    return rotseq


def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def show_pseq(pseq, rgba=(1, 0, 0, 1), radius=0.0005, show_stick=False):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=radius).attach_to(base)
    if show_stick:
        for i in range(0, len(pseq) - 1):
            gm.gen_stick(spos=np.asarray(pseq[i]), epos=np.asarray(pseq[i + 1]), rgba=rgba, thickness=radius) \
                .attach_to(base)


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


def get_objpcd_partial_o3d(objcm):
    vis = o3d.visualization.Visualizer()
    objpcd = o3dh.nparray2o3dpcd(np.asarray(objcm.sample_surface(radius=.001)[0]))
    o3d.geometry.TriangleMesh.
    o3d.visualization.draw_geometries(
        [objpcd], width=400, height=400, point_show_normal=True
    )
    vis.add_geometry(objpcd)
    vis.capture_depth_point_cloud("tst.pcd", do_render=True)


def get_objpcd_partial(objcm, img_size=(200, 200), granurity=0.2645 / 1000):
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


if __name__ == '__main__':
    import visualization.panda.world as wd

    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    width = .005
    thickness = .0015
    pseq = gen_sgl_curve(pseq=np.asarray([[0, 0, 0], [.018, .03, 0], [.06, .06, 0], [.12, 0, 0]]))
    rotseq = get_rotseq_by_pseq(pseq)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    objcm = gen_swap(pseq, rotseq, cross_sec)
    objcm.set_rotmat(rm.rotmat_from_axangle((0, 1, 0), np.pi / 4))
    # objpcd = get_objpcd_partial_sample(objcm, cam_pos=cam_pos)
    # objpcd, mask = get_objpcd_partial(objcm, granurity=.0005)
    objpcd = get_objpcd_partial_o3d(objcm)

    gm.gen_pointcloud(objpcd).attach_to(base)
    objcm.set_rgba((1, 1, 1, 1))
    objcm.attach_to(base)
    base.run()
