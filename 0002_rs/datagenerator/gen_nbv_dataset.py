import math
import os
import random

import numpy as np
import open3d as o3d
import basis.o3dhelper as o3dh

import basis.robot_math as rm
import datagenerator.data_utils as utl
import modeling.collision_model as cm
import modeling.geometric_model as gm
import utils.pcd_utils as pcdu
import visualization.panda.world as wd

PATH = 'D:/nbv_mesh'


# PATH = 'E:/liu/nbv_mesh'


def init_gen(cat, num_kpts, max, i, length=.2, path=PATH, rand_prim=False, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, cat)):
        os.mkdir(os.path.join(path, cat))
        os.mkdir(os.path.join(path, cat, 'mesh'))
        os.mkdir(os.path.join(path, cat, 'prim'))

    objcm, objcm_gt, _, _ = utl.gen_seed(num_kpts, max=max, n=100, length=length, rand_prim=rand_prim,
                                         toggledebug=False)
    f_name = str(i).zfill(4)
    o3dmesh = utl.cm2o3dmesh(objcm, wnormal=False)
    o3dmesh_gt = utl.cm2o3dmesh(objcm_gt, wnormal=False)
    if not toggledebug:
        o3d.io.write_triangle_mesh(os.path.join(path, cat, 'mesh', f_name + '.ply'), o3dmesh)
        o3d.io.write_triangle_mesh(os.path.join(path, cat, 'prim', f_name + '.ply'), o3dmesh_gt)
    else:
        o3dmesh_gt.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([o3dmesh, o3dmesh_gt])


def init_gen_deform(cat, num_kpts, i, path=PATH, toggledebug=False):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, cat)):
        os.mkdir(os.path.join(path, cat))
        os.mkdir(os.path.join(path, cat, 'mesh'))
        os.mkdir(os.path.join(path, cat, 'prim'))
    f_name = str(i).zfill(4)
    if not os.path.exists(os.path.join(path, cat, 'mesh', f_name + '.ply')):
        # icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
        # icomats = [x for row in icomats for x in row]
        # icomats = [rot for rot in icomats if rm.angle_between_vectors(rot[:, 0], (1, 0, 0)) < np.pi]
        rot = rm.rotmat_from_axangle((0, 1, 0), random.uniform(-np.pi / 4, np.pi / 4)) \
            .dot(rm.rotmat_from_axangle((1, 0, 0), random.uniform(-np.pi, np.pi))) \
            .dot(rm.rotmat_from_axangle((0, 1, 0), random.uniform(-np.pi / 4, np.pi / 4)))
        if cat[:4] == 'tmpl':
            objcm = cm.CollisionModel(f'../obstacles/template.stl')
        else:
            objcm = cm.CollisionModel(f'../obstacles/plate.stl')

        if num_kpts == 3:
            goal_pseq = \
                np.asarray([[0, 0, 0],
                            [.08 + random.uniform(-.02, .02), 0, .04 * random.choice([-1, 1])],
                            [.16, 0, 0]])
        elif num_kpts == 4:
            goal_pseq = \
                np.asarray([[0, 0, 0],
                            [.04 + random.uniform(-.02, .04), 0, random.uniform(.01, .02) * random.choice([-1, 1])],
                            [.12 + random.uniform(-.04, .02), 0, random.uniform(.01, .02) * random.choice([-1, 1])],
                            [.16, random.uniform(-.02, .02), 0]])
        else:
            goal_pseq = \
                np.asarray([[0, 0, 0],
                            [.04 + random.uniform(-.02, .01), 0, random.uniform(-.01, .01)],
                            [.08 + random.uniform(-.01, .01), 0, random.uniform(.01, .02) * random.choice([-1, 1])],
                            [.12 + random.uniform(-.01, .02), 0, random.uniform(-.01, .01)],
                            [.16, 0, 0]])

        rot_axial, rot_radial = utl.random_rot_radians(num_kpts)
        rbf_radius = random.uniform(.05, .2)
        objcm_deformed, objcm_gt, _, _ = utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=rbf_radius)

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

    else:
        print('exist')


def show_mesh(cat, path=PATH):
    gm.gen_frame().attach_to(base)
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        mesh = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f))
        print(f)
        objcm = o3dh.o3dmesh2cm(mesh)
        objcm.attach_to(base)
    base.run()


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])

    num = 100
    for i in range(0, num):
        print(i)
        # init_gen('bspl_3', num_kpts=3, max=random.choice([.02, .03, .04]), i=i, toggledebug=False)
        # init_gen('bspl_4', num_kpts=4, max=random.choice([.02, .03, .04]), i=i, toggledebug=False)
        # init_gen('bspl_5', num_kpts=5, max=random.choice([.02, .03, .04]), i=i, toggledebug=False)
        # init_gen('rlen_3', num_kpts=3, max=random.choice([.02, .03, .04]), i=i, rand_prim=True, toggledebug=False)
        # init_gen('rlen_4', num_kpts=4, max=random.choice([.02, .03, .04]), i=i, rand_prim=True, toggledebug=False)
        # init_gen('rlen_5', num_kpts=5, max=random.choice([.02, .03, .04]), i=i, rand_prim=True, toggledebug=False)
        # init_gen_deform('plat', num_kpts=random.choice([3, 4, 5]), i=i)
        init_gen_deform('tmpl', num_kpts=random.choice([3]), i=i, toggledebug=False)
    # show_mesh('tmpl', path=PATH)
    # icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 6))
    # icomats = [x for row in icomats for x in row]
    # for rot in icomats:
    #     if rm.angle_between_vectors(rot[:, 0], (1, 0, 0)) < np.pi / 3:
    #         gm.gen_frame(pos=(0, 0, 0), rotmat=rot).attach_to(base)
    # base.run()
