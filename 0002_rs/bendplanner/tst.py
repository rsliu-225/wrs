import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
from scipy import interpolate
from scipy.optimize import minimize
import basis.o3dhelper as o3dh
import time
import matplotlib.pyplot as plt


def normed_distance_along_path(polyline_x, polyline_y, polyline_z):
    polyline = np.asarray([polyline_x, polyline_y, polyline_z])
    distance = np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=1) ** 2, axis=0)))
    return np.insert(distance, 0, 0) / distance[-1]


def average_distance_between_polylines(pts1, pts2, toggledebug=False):
    x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]
    s1 = normed_distance_along_path(x1, y1, z1)
    s2 = normed_distance_along_path(x2, y2, z2)

    interpol_xyz1 = interpolate.interp1d(s1, [x1, y1, z1])
    xyz1_on_2 = interpol_xyz1(s2)

    node_to_node_distance = np.sqrt(np.sum((xyz1_on_2 - [x2, y2, z2]) ** 2, axis=0))
    if toggledebug:
        ax = plt.axes(projection='3d')

        ax.scatter3D(x1, y1, z1, color='red')
        ax.plot3D(x1, y1, z1, 'red')

        ax.scatter3D(x2, y2, z2, color='green')
        ax.plot3D(x2, y2, z2, 'green')

        ax.scatter3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], color='black')
        ax.plot3D(xyz1_on_2[0], xyz1_on_2[1], xyz1_on_2[2], 'black', linestyle='dotted')
        plt.show()

    return node_to_node_distance.mean(), xyz1_on_2


def gen_circle(r):
    pts = []
    for a in np.arange(0, 2 * math.pi, math.pi / 90):
        pts.append([r * math.cos(a), r * math.sin(a), 0])
    return pts


pts1 = gen_circle(2)
pts2 = [[0, 2, 0], [1, 1, 0], [0.5, -1, 0], [-0.5, -1, 0], [-1, 1, 0]]

pts1 = np.asarray(pts1)
pts2 = np.asarray(pts2)

# res, interp_pts = average_distance_between_polylines(pts1, pts2, toggledebug=True)
# print(res)

ax = plt.axes(projection='3d')

