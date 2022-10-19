import numpy as np

from basis.trimesh.creation import icosphere
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm
import visualization.panda.world as wd

# from huri.core.base_boost import zoombase, boost_base

img_name = "ico"
base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
tm = icosphere(subdivisions=3)
# for i in tm.vertices:
#     gm.gen_sphere(i, radius=.01).attach_to(base)
theta = np.radians(45)
origin = gm.gen_sphere(radius=.00001)
origin.attach_to(base)
for v_id, v in enumerate(tm.vertices):
    o_v_vector = v - np.array([0, 0, 0])
    rgba = [1, 0, 0, 1]
    # print(rm.angle_between_vectors(o_v_vector, np.array([0, 0, 1])))
    # if rm.angle_between_vectors(o_v_vector, np.array([0, 0, 1])) < theta:
    #     rgba = [0, 1, 0, 1]
    # random_color = rm.random_rgba()
    # gm.gen_arrow(spos=center, epos=center + center_normal * 0.3, rgba=rgba, thickness=.03).attach_to(base)
    gm.gen_sphere(v, radius=.03, rgba=rgba).attach_to(origin)
icosphere_cm = cm.CollisionModel(tm)
icosphere_cm.set_rgba([1, 1, 1, 1])
# icosphere_cm.objpdnp.setBin("fixed",10)

icosphere_cm.attach_to(base)
icosphere_cm.show_cdmesh()

selection_vector = gm.gen_arrow(np.array([0, 0, 0]), np.array([0, 0, 1]), thickness=.1)
selection_vector.set_rgba(np.array([31/255, 191/255, 31/255, 1]))

vertices, vertex_normals, faces = icosphere_cm.extract_rotated_vvnf()
objwm = gm.WireFrameModel(cm.da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
objwm.attach_to(base)
icosphere_cm.set_scale([.995, .995, .995])
# base = boost_base(base)

# base.boost.add_task(zoombase, args=[base, np.array([0.8, 0, 0.8])], timestep=0.2)
# base.boost.add_task(lambda task: base.boost.screen_shot(f"{img_name}_1"), timestep=0.3)
# base.boost.add_task(lambda task: origin.detach(), timestep=0.4)
# base.boost.add_task(lambda task: icosphere_cm.detach(), timestep=0.4)
# base.boost.add_task(lambda task: objwm.detach(), timestep=0.4)
# base.boost.add_task(lambda task: selection_vector.attach_to(base), timestep=0.5)
# base.boost.add_task(lambda task: base.boost.screen_shot(f"{img_name}_2"), timestep=0.6)
# base.boost.add_task(lambda task: exit(), timestep=0.6)
base.run()