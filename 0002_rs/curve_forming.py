import math

import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm

base = wd.World(cam_pos=[0, 0, 10], lookat_pos=[0, 0, 0])


def draw_plane(p, n):
    pt_direction = rm.orthogonal_vector(n, toggle_unit=True)
    tmp_direction = np.cross(n, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, n))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array(p)
    gm.gen_box(np.array([.5, .5, .001]), homomat=homomat, rgba=[1, 1, 1, .3]).attach_to(base)


def draw_arc(theta, r, spos, sdir):
    gm.gen_sphere(pos=spos, rgba=[1, 0, 0, 1]).attach_to(base)
    gm.gen_arrow(spos=spos, epos=spos + sdir).attach_to(base)
    gm.gen_box()
    alpha = rm.angle_between_vectors(sdir, np.asarray([1, 1, 0]))
    p_tmp = [spos[0] + 2 * r * math.sin(theta / 2) * math.cos(alpha - theta / 2),
             spos[1] + 2 * r * math.sin(theta / 2) * math.sin(alpha - theta / 2), 0]
    gm.gen_sphere(pos=np.asarray(p_tmp), rgba=[1, 0, 0, 1]).attach_to(base)
    draw_plane(spos, sdir)


draw_arc(theta=math.pi / 2, r=.1, spos=np.array([0, 0, 0]), sdir=np.array([.1, .1, 0]))
base.run()
