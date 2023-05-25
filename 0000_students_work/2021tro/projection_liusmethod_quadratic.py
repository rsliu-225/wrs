import math

import numpy as np
from scipy.spatial import cKDTree

import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import vision.depth_camera.surface.quadrantic_surface as qs
import vision.depth_camera.surface.rbf_surface as rbfs
# import vision.depth_camera.surface.gaussian_surface as gs
import visualization.panda.world as wd


def gen_circle(r=.02, interval=15):
    p_list = []
    for i in range(-180, 180, interval):
        p_list.append((r * math.sin(math.radians(i)), r * math.cos(math.radians(i))))
    return p_list


base = wd.World(cam_pos=np.array([-.3, -.7, .42]), lookat_pos=np.array([0, 0, 0]))
# gm.gen_frame().attach_to(base)
bowl_model = cm.CollisionModel(initor="./objects/cylinder2.stl")
bowl_model.set_pos((-.08, -.03, -.1))

# bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
bowl_model.set_rgba([.3, .3, .3, .3])
# bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi, 0, 0))
bowl_model.attach_to(base)

pn_direction = np.array([0, 0, -1])

bowl_samples, bowl_sample_normals = bowl_model.sample_surface(toggle_option='normals', radius=.002)
selection = bowl_sample_normals.dot(-pn_direction) > .1
bowl_samples = bowl_samples[selection]
bowl_sample_normals = bowl_sample_normals[selection]
tree = cKDTree(bowl_samples)

pt_direction = rm.orthogonal_vector(pn_direction, toggle_unit=True)
tmp_direction = np.cross(pn_direction, pt_direction)
plane_rotmat = np.column_stack((pt_direction, tmp_direction, pn_direction))
homomat = np.eye(4)
homomat[:3, :3] = plane_rotmat
homomat[:3, 3] = np.array([-.07, -.03, .1])
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[1, 1, 1, .3])
twod_plane.attach_to(base)

circle_radius = .05
p_list = np.asarray(gen_circle(r=.04, interval=15))
p_list = p_list + homomat[:2, 3]
for p in p_list:
    gm.gen_sphere(pos=(p[0], p[1], .1), radius=.001).attach_to(base)

line_segs = np.asarray(
    [[(p_list[i][0], p_list[i][1], homomat[:3, 3][2]), (p_list[i + 1][0], p_list[i + 1][1], homomat[:3, 3][2])] for i in
     range(len(p_list) - 1)])

gm.gen_linesegs(line_segs).attach_to(base)
gm.gen_arrow(spos=line_segs[0][0], epos=line_segs[0][1], thickness=0.004, rgba=(0, 1, 0, 1)).attach_to(base)
# spt = homomat[:3,3]
spt = line_segs[0][0]
# gm.gen_stick(spt, spt + pn_direction * 10, rgba=[0,1,0,1]).attach_to(base)
# base.run()
gm.gen_dasharrow(spt, spt - pn_direction * .07, thickness=.004).attach_to(base)  # p0
cpt, cnrml = bowl_model.ray_hit(spt, spt + pn_direction * 10000, option='closest')
gm.gen_dashstick(spt, cpt, rgba=[.57, .57, .57, .7], thickness=0.003).attach_to(base)
gm.gen_sphere(pos=cpt, radius=.005).attach_to(base)
gm.gen_dasharrow(cpt, cpt - pn_direction * .07, thickness=.004).attach_to(base)  # p0
gm.gen_dasharrow(cpt, cpt + cnrml * .07, thickness=.004).attach_to(base)  # p0

angle = rm.angle_between_vectors(-pn_direction, cnrml)
vec = np.cross(-pn_direction, cnrml)
rotmat = rm.rotmat_from_axangle(vec, angle)
new_plane_homomat = np.eye(4)
new_plane_homomat[:3, :3] = rotmat.dot(homomat[:3, :3])
new_plane_homomat[:3, 3] = cpt
# twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=new_plane_homomat, rgba=[1, 1, 1, .3])
twod_plane.attach_to(base)

new_line_segs = line_segs
gm.gen_linesegs(new_line_segs).attach_to(base)
# gm.gen_arrow(spos=new_line_segs[0][0], epos=new_line_segs[0][1], thickness=0.004).attach_to(base)

t_cpt = cpt
last_normal = cnrml
# direction = rotmat.dot(pt_direction)
direction = line_segs[0][1] - line_segs[0][0]

# surface = qs.QuadraticSurface(bowl_samples[:, :2], bowl_samples[:, 2])
surface = rbfs.RBFSurface(bowl_samples[:, :2], bowl_samples[:, 2])
# surface = gs.MixedGaussianSurface(bowl_samples[:, :2], bowl_samples[:,2],n_mix=1)
surface_gm = surface.get_gometricmodel([[-.2, .06], [-.13, .08]], rgba=[.5, .7, 1, .5])
gm.GeometricModel(bowl_samples).attach_to(base)
surface_gm.attach_to(base)

for tick in range(0, 5):
# for tick in range(0, len(line_segs)):
    t_npt = cpt + direction
    # gm.gen_arrow(spos=t_npt, epos=t_npt + last_normal * .015, thickness=0.001).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .01)
    nearby_samples = bowl_samples[nearby_sample_ids]
    # gm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    nearby_samples_on_xy = plane_rotmat.T.dot((nearby_samples - plane_center).T).T
    surface = qs.QuadraticSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:, 2])
    # surface = rbfs.RBFSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:, 2])
    # surface = gs.MixedGaussianSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:,2],n_mix=1)
    t_npt_on_xy = plane_rotmat.T.dot(t_npt - plane_center)
    projected_t_npt_z_on_xy = surface.get_zdata(np.array([t_npt_on_xy[:2]]))
    projected_t_npt_on_xy = np.array([t_npt_on_xy[0], t_npt_on_xy[1], projected_t_npt_z_on_xy[0]])
    projected_point = plane_rotmat.dot(projected_t_npt_on_xy) + plane_center
    # surface_gm = surface.get_gometricmodel([[-.15, .15], [-.15, .15]], rgba=[.5, .7, 1, .3])
    # surface_gm.set_pos(plane_center)
    # surface_gm.set_rotmat(plane_rotmat)
    # surface_gm.attach_to(base)
    new_normal = rm.unit_vector(t_npt - projected_point)
    if np.dot(new_normal, np.asarray([0, 0, 1])) < 0:
        new_normal = -new_normal
    gm.gen_arrow(spos=projected_point, epos=projected_point + new_normal * .02, thickness=0.001).attach_to(base)
    angle = rm.angle_between_vectors(-pn_direction, new_normal)
    vec = np.cross(-pn_direction, new_normal)
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(line_segs[tick][1] - line_segs[tick][0])
    new_tmp_direction = new_rotmat.dot(tmp_direction)
    new_line_segs = [[cpt, projected_point]]
    gm.gen_linesegs(new_line_segs, rgba=[1, .6, 0, 1]).attach_to(base)
    cpt = projected_point
    # new_line_segs = [[cpt, cpt+direction*(.05-tick*.05/n)],
    #                  [cpt+direction*(.05-tick*.05/n), cpt+direction*(.05-tick*.05/n)+new_tmp_direction*.05]]
    last_normal = new_normal
    # break

base.run()

base.run()
