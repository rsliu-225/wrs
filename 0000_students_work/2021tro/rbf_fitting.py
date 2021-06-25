import math

import numpy as np

import basis.robot_math as rm
import modeling.geometric_model as gm
import vision.depth_camera.rbf_surface as rbfs
import visualization.panda.world as wd
import math
import vision.depth_camera.surface.gaussian_surface as gs
import vision.depth_camera.surface.quadrantic_surface as qs

base = wd.World(cam_pos=np.array([.5, .1, .3]), lookat_pos=np.array([0, 0, 0.05]))
gm.gen_frame().attach_to(base)
tube_model = gm.GeometricModel(initor="./objects/bowl.stl")
tube_model.set_rgba([.3, .3, .3, .3])
tube_model.attach_to(base)
points, points_normals = tube_model.sample_surface(radius=.002, nsample=10000, toggle_option='normals')
sampled_points = []
for id, p in enumerate(points.tolist()):
    if np.dot(np.array([1, 0, 0]), points_normals[id]) > .3 and p[0] > 0:
        gm.gen_sphere(pos=p, radius=.001).attach_to(base)
        sampled_points.append(p)

# x - v
# y - u
rotmat_uv = rm.rotmat_from_euler(0, math.pi / 2, 0)
sampled_points = rotmat_uv.dot(np.array(sampled_points).T).T
surface = rbfs.RBFSurface(sampled_points[:, :2], sampled_points[:, 2])
z = surface.get_zdata([[20, 0]])
print(z)
gm.gen_sphere(pos=(10, 0, z[0]), rgba=(0, 1, 1, 1), radius=.5).attach_to(base)
surface_gm = surface.get_gometricmodel(rgba=(.7, .7, .3, 1))
surface_gm.set_rotmat(rotmat_uv.T)
surface_gm.attach_to(base)

surface = gs.MixedGaussianSurface(sampled_points[:, :2], sampled_points[:, 2], n_mix=1)
z = surface.get_zdata([[20, 0]])
print(z)
gm.gen_sphere(pos=(10, 0, z[0]), rgba=(0, 1, 1, 1), radius=.5).attach_to(base)
surface_gm = surface.get_gometricmodel(rgba=(.7, .3, .7, 1))
surface_gm.set_rotmat(rotmat_uv.T)
surface_gm.attach_to(base)

surface = qs.QuadraticSurface(sampled_points[:, :2], sampled_points[:, 2])
z = surface.get_zdata([[20, 0]])
print(z)
gm.gen_sphere(pos=(10, 0, z[0]), rgba=(0, 1, 1, 1), radius=.5).attach_to(base)
surface_gm = surface.get_gometricmodel(rgba=(.3, .3, .7, 1))
surface_gm.set_rotmat(rotmat_uv.T)
surface_gm.attach_to(base)

base.run()
