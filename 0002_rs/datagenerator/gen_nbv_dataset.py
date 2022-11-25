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


def init_gen(cat, num_kpts, max, i, length=.2, path=PATH, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, cat)):
        os.mkdir(os.path.join(path, cat))
        os.mkdir(os.path.join(path, cat, 'mesh'))
        os.mkdir(os.path.join(path, cat, 'prim'))

    objcm, objcm_gt, _, _ = utl.gen_seed(num_kpts, max=max, n=100, length=length, rand_wd=False)
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
    rot_axial, rot_radial = utl.random_rot_radians(num_kpts)
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
