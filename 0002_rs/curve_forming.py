import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import utils.math_utils as mu
from sympy import symbols, diff, solve, exp

base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])


def draw_plane(p, n):
    pt_direction = rm.orthogonal_vector(n, toggle_unit=True)
    tmp_direction = np.cross(n, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, n))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array(p)
    gm.gen_box(np.array([.5, .5, .001]), homomat=homomat, rgba=[1, 1, 1, .3]).attach_to(base)


def draw_arc(center, r, theta, lift_angle):
    for a in np.arange(0, theta, math.pi / 90):
        if lift_angle == 0:
            p = (r * math.cos(a), r * math.sin(a), 0)
        else:
            p = (r * math.cos(a), r * math.sin(a), a * r / math.tan(lift_angle))
        gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, 1], radius=0.001).attach_to(base)
    draw_plane(center, np.asarray([0, 0, 1]))


def cal_tail(r_side, r_center, d, m):
    A = np.mat([[r_side + m, -r_center], [1, 1]])
    b = np.mat([0, d]).T
    l_center, l_side = np.asarray(np.linalg.solve(A, b))

    return l_center[0], l_side[0], \
           np.sqrt(l_center[0] ** 2 - r_center ** 2) + np.sqrt(l_side[0] ** 2 - (r_side + m) ** 2)


def cal_safe_margin(r_side, r_base, d):
    return np.degrees(2 * np.arcsin((r_side + r_base) / (2 * d)))


def cal_start_margin(l_center, r_center):
    return np.degrees(2 * np.arccos(r_center / l_center))


# r_side, r_center = 13 / 2, 15 / 2
# d = 20.5
# l_center, l_side, l = cal_tail(r_side, r_center, d, 2)
# print(l)
# print(l_center, l_side)
# a = cal_safe_margin(r_side, r_center, d)
# print(a)
# b = cal_start_margin(l_center, r_center)
# print(b)

gm.gen_frame(thickness=.001).attach_to(base)
gm.gen_stick(spos=np.asarray([0, 0, 0]), epos=np.asarray([0, 0, .2]), thickness=.026, sections=180,
             rgba=[.7, .7, .7, .5]).attach_to(base)
draw_arc(center=np.asarray([0, 0, 0]), r=.013, theta=math.pi / 4, lift_angle=math.pi / 3)
base.run()

