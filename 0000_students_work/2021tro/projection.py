import math

import numpy as np
from scipy.spatial import cKDTree

import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd

base = wd.World(cam_pos=np.array([-.2, -.7, .42]), lookat_pos=np.array([0, 0, 0]))
# gm.gen_frame().attach_to(base)
bowl_model = cm.CollisionModel(initor="./objects/bowl.stl")
bowl_model.set_rgba([.3, .3, .3, .3])
bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi, 0, 0))
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
line_segs = [[homomat[:3, 3], homomat[:3, 3] + pt_direction * .05],
             [homomat[:3, 3] + pt_direction * .05, homomat[:3, 3] + pt_direction * .05 + tmp_direction * .05]]
print(line_segs)
gm.gen_linesegs(line_segs).attach_to(base)
gm.gen_arrow(spos=line_segs[0][0], epos=line_segs[0][1], thickness=0.004).attach_to(base)
spt = homomat[:3, 3]
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
twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=new_plane_homomat, rgba=[1, 1, 1, .3])
twod_plane.attach_to(base)
new_line_segs = [[cpt, cpt + rotmat.dot(pt_direction) * .05],
                 [cpt + rotmat.dot(pt_direction) * .05,
                  cpt + rotmat.dot(pt_direction) * .05 + rotmat.dot(tmp_direction) * .05]]
gm.gen_linesegs(new_line_segs).attach_to(base)
# gm.gen_arrow(spos=new_line_segs[0][0], epos=new_line_segs[0][1], thickness=0.004).attach_to(base)

t_cpt = cpt
last_normal = cnrml
direction = rotmat.dot(pt_direction)
tmp_direction = rotmat.dot(tmp_direction)
n = 3
for tick in range(1, n + 1):
    t_npt = cpt + direction * .05 / n
    gm.gen_arrow(spos=t_npt, epos=t_npt + last_normal * .015, thickness=0.001).attach_to(base)
    nearby_sample_ids = tree.query_ball_point(t_npt, .005)
    nearby_samples = bowl_samples[nearby_sample_ids]
    gm.GeometricModel(nearby_samples).attach_to(base)
    plane_center, plane_normal = rm.fit_plane(nearby_samples)
    plane_tangential = rm.orthogonal_vector(plane_normal)
    plane_tmp = np.cross(plane_normal, plane_tangential)
    plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = plane_center
    twod_plane = gm.gen_box(np.array([.2, .2, .001]), homomat=homomat, rgba=[.5, .7, 1, .3]).attach_to(base)
    projected_point = rm.project_to_plane(t_npt, plane_center, plane_normal)
    # gm.gen_stick(t_npt, projected_point, thickness=.002).attach_to(base)
    new_normal = rm.unit_vector(t_npt - projected_point)
    gm.gen_arrow(spos=projected_point, epos=projected_point + new_normal * .015, thickness=0.001).attach_to(base)
    angle = rm.angle_between_vectors(last_normal, new_normal)
    vec = rm.unit_vector(np.cross(last_normal, new_normal))
    new_rotmat = rm.rotmat_from_axangle(vec, angle)
    direction = new_rotmat.dot(direction)
    tmp_direction = new_rotmat.dot(tmp_direction)
    cpt = projected_point
    new_line_segs = [[cpt, cpt + direction * (.05 - tick * .05 / n)],
                     [cpt + direction * (.05 - tick * .05 / n),
                      cpt + direction * (.05 - tick * .05 / n) + tmp_direction * .05]]
    gm.gen_linesegs(new_line_segs).attach_to(base)
    # break

base.run()

base.run()
