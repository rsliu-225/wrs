import math
import os
import random

import numpy as np
import open3d as o3d

import basis.robot_math as rm
import datagenerator.data_utils as utl
import modeling.collision_model as cm
import utils.pcd_utils as pcdu

# PATH = 'D:/nbv_mesh'
PATH = 'E:/liu/nbv_mesh'


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

    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    rot = random.choice(icomats)
    if cat[:4] == 'tmpl':
        objcm = cm.CollisionModel(f'../obstacles/template.stl')
    else:
        objcm = cm.CollisionModel(f'../obstacles/plate.stl')

    if num_kpts == 3:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.08 + random.uniform(-.02, .02), 0, random.uniform(.01, .04) * random.choice([-1, 1])],
                                [.16, 0, 0]])
    elif num_kpts == 4:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .04), 0, random.uniform(.01, .03) * random.choice([-1, 1])],
                                [.12 + random.uniform(-.04, .02), 0, random.uniform(.01, .03) * random.choice([-1, 1])],
                                [.16, 0, 0]])
    else:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .01), 0, random.uniform(-.01, .01)],
                                [.08 + random.uniform(-.01, .01), 0, random.uniform(-.02, .02)],
                                [.12 + random.uniform(-.01, .02), 0, random.uniform(-.01, .01)],
                                [.16, 0, 0]])

    rot_axial, rot_radial = utl.random_rot_radians(num_kpts)
    rbf_radius = random.uniform(.05, .2)
    objcm_deformed, objcm_gt, _, _ = utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=rbf_radius)
    f_name = str(i).zfill(4)
    o3dmesh = utl.cm2o3dmesh(objcm_deformed, wnormal=False)
    o3dmesh_gt = utl.cm2o3dmesh(objcm_gt, wnormal=False)
    o3dmesh.rotate(rot, center=(0, 0, 0))
    o3dmesh_gt.rotate(rot, center=(0, 0, 0))
    # o3d.visualization.draw_geometries([o3dmesh])
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'mesh', f_name + '.ply'), o3dmesh)
    o3d.io.write_triangle_mesh(os.path.join(path, cat, 'prim', f_name + '.ply'), o3dmesh_gt)
    # mesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f_name + '.ply'))
    # o3d.visualization.draw_geometries([mesh])

    if toggledebug:
        o3dmesh_gt.paint_uniform_color([1, 1, 0])
        o3d.visualization.draw_geometries([o3dmesh, o3dmesh_gt])


if __name__ == '__main__':
    num = 100
    for i in range(60, num):
        print(i)
        # init_gen('bspl', num_kpts=4, max=random.choice([.01, .02, .03, .04]), i=i)
        init_gen_deform('plat', num_kpts=random.choice([3, 4, 5]), i=i)
        # init_gen_deform('tmpl', num_kpts=random.choice([3, 4, 5]), i=i)
