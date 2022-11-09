import os.path
import random

import numpy
import numpy as np

from _tst_gen_dataset import *
import data_utils as utl
from gen_multiview import *
import open3d as o3d
import pickle
from multiprocessing import Process

# PATH = 'E:/liu/dataset_2048_flat/'
# PATH = 'E:/liu/dataset_2048_prim_v10/'
PATH = 'D:/nbv_mesh'


def random_kts(n=3, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append(((j + 1) * .02, random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


def random_rot_radians(n=3):
    rot_axial = []
    rot_radial = []
    for i in range(n):
        rot_axial.append(random.randint(10, 30) * random.choice([1, -1]))
        rot_radial.append(random.randint(0, 1) * random.choice([1, -1]))
    return np.radians(rot_axial), np.radians(rot_radial)


def gen_seed(num_kpts=4, max=.02, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, rand_wid=False):
    width = width + (np.random.uniform(0, 0.005) if rand_wid else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    flat_sec = [[0, width / 2], [0, -width / 2], [0, -width / 2], [0, width / 2]]
    success = False
    pseq, rotseq = [], []
    while not success:
        kpts = random_kts(num_kpts, max=max)
        if len(kpts) == 3:
            pseq = utl.uni_length(utl.poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
        # elif len(kpts) == 4:
        #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
        else:
            pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
        pseq = np.asarray(pseq) - pseq[0]
        pseq, rotseq = utl.get_rotseq_by_pseq(pseq)
        for i in range(len(rotseq) - 1):
            if rm.angle_between_vectors(rotseq[i][:, 2], rotseq[i + 1][:, 2]) > np.pi / 15:
                success = False
                break
            success = True
    return utl.gen_swap(pseq, rotseq, cross_sec), utl.gen_swap(pseq, rotseq, flat_sec)


def init_gen(cat, num_kpts, max, i, length=.2, path=PATH, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, cat)):
        os.mkdir(os.path.join(path, cat))
        os.mkdir(os.path.join(path, cat, 'mesh'))
        os.mkdir(os.path.join(path, cat, 'prim'))

    objcm, objcm_gt = gen_seed(num_kpts, max=max, n=100, length=length, rand_wid=False)
    f_name = str(i).zfill(4)
    o3dmesh = utl.cm2o3dmesh(objcm, wnormal=False)
    o3dmesh_gt = utl.cm2o3dmesh(objcm_gt, wnormal=False)
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'mesh', f_name + '.ply'), o3dmesh)
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'prim', f_name + '.ply'), o3dmesh_gt)
    if toggledebug:
        o3dmesh_gt.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([o3dmesh, o3dmesh_gt])


def init_gen_deform(cat, num_kpts, i, path=PATH, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, cat)):
        os.mkdir(os.path.join(path, cat))
        os.mkdir(os.path.join(path, cat, 'mesh'))
        os.mkdir(os.path.join(path, cat, 'prim'))

    if cat[:4] == 'tmpl':
        objcm = cm.CollisionModel(f'../obstacles/template.stl')
    else:
        objcm = cm.CollisionModel(f'../obstacles/plate.stl')

    if num_kpts == 3:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.08 + random.uniform(-.02, .02), 0, random.uniform(.015, .02)],
                                [.16, 0, 0]])
    elif num_kpts == 4:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .04), 0, random.uniform(-.015, .015)],
                                [.12 + random.uniform(-.04, .02), 0, random.uniform(-.015, .015)],
                                [.16, 0, 0]])
    else:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .01), 0, random.uniform(-.005, .005)],
                                [.08 + random.uniform(-.01, .01), 0, random.uniform(-.015, .015)],
                                [.12 + random.uniform(-.01, .02), 0, random.uniform(-.005, .005)],
                                [.16, 0, 0]])
    rot_axial, rot_radial = random_rot_radians(num_kpts)
    rbf_radius = random.uniform(.05, .2)
    objcm_deformed, objcm_gt = utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=rbf_radius)
    f_name = str(i).zfill(4)
    o3dmesh = utl.cm2o3dmesh(objcm_deformed, wnormal=False)
    o3dmesh_gt = utl.cm2o3dmesh(objcm_gt, wnormal=False)
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'mesh', f_name + '.ply'), o3dmesh)
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'prim', f_name + '.ply'), o3dmesh_gt)
    if toggledebug:
        o3dmesh_gt.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([o3dmesh, o3dmesh_gt])


if __name__ == '__main__':
    num = 100
    for i in range(num):
        print(i)
        init_gen('bspl', num_kpts=4, max=random.choice([.01, .02, .03, .04]), i=i)
        init_gen_deform('plat', num_kpts=random.choice([3, 4, 5]), i=i)
        init_gen_deform('tmpl', num_kpts=random.choice([3, 4, 5]), i=i)
