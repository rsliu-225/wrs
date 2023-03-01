import math
import warnings as wns
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import config
import motionplanner.robot_helper as rbt_helper
import modeling.geometric_model as gm
import basis.robot_math as rm
import copy
import utils.pcd_utils as pcdu
import localenv.envloader as el

import nbv.nbv_utils as nu
import datagenerator.data_utils as du
import bendplanner.bend_utils as bu
import open3d as o3d
import pcn.inference as pcn
from scipy.spatial import KDTree
import basis.o3dhelper as o3dh

TB = True
COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


class NBVOptimizer(object):
    def __init__(self, env=None, model_name='pcn', load_model='pcn_emd_rec/best_emd_network.pth', toggledebug=TB):
        self.model_name, self.load_model = model_name, load_model
        self.env = env
        self.toggledebug = toggledebug
        self.rot_center = (0, 0, 0)
        self.result = None
        self.cons = []
        b = (-np.pi, -np.pi)
        tb = (-.1, .1)
        self.bnds = (tb, tb, tb, b, b, b)

        self.seedjntagls = None
        self.tgtpos = None
        self.tgtrot = None

        self.jnts = []
        self.rot_err = []
        self.pos_err = []
        self.jd_list = []  # joints displacement
        self.mp_list = []  # manipulability
        self.sr_list = []  # angle between line of sight
        self.wo_list = []  # wrist obstruction
        self.obj_list = []

        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = [], [], []
        self.o3dpcd_o, self.o3dmesh, self.o3dpcd_nbv = None, None, None
        self.campos = None

    def objctive(self, x):
        conf_sum = 0
        rot = rm.rotmat_from_euler(x[0], x[1], x[2])
        o3dpcd_tmp = nu.gen_partial_o3dpcd(self.o3dmesh, trans=x[3:], rot=rot, rot_center=self.rot_center)
        o3dpcd_tmp.paint_uniform_color((0, 1, 0))
        kdt_nbv = o3d.geometry.KDTreeFlann(self.o3dpcd_nbv)
        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dpcd_i, self.o3dmesh])
        # o3d.visualization.draw_geometries([o3dpcd_tmp, self.o3dpcd_nbv])
        pcd_tmp = np.asarray(o3dpcd_tmp.points)
        _, _, trans = o3dh.registration_icp_ptpt(pcd_tmp, np.asarray(self.o3dpcd_nbv.points),
                                                 maxcorrdist=.02, toggledebug=False)
        pcd_tmp = pcdu.trans_pcd(pcd_tmp, trans)

        for p in pcd_tmp:
            _, idx, _ = kdt_nbv.search_knn_vector_3d(p, 1)
            if np.linalg.norm(p - self.nbv_pts[idx]) < .01 and self.nbv_conf[idx] < .2:
                conf_sum += 1 - (self.nbv_conf[idx])
        self.obj_list.append(conf_sum)
        # print(x, conf_sum)

        return -conf_sum

    def update_known(self, pcd_i, campos):
        def _sigmoid(x):
            s = 1 / (1 + np.exp(1 - 2 * x))
            return s

        width = .008
        thickness = .0015
        cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

        self.campos = campos

        pcd_o = pcn.inference_sgl(pcd_i, self.model_name, self.load_model, toggledebug=False)
        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = \
            pcdu.cal_nbv_pcn(pcd_i, pcd_o, cam_pos=campos, theta=None, toggledebug=True)
        # print(self.nbv_conf)
        # self.nbv_conf = _sigmoid(self.nbv_conf)
        # print(self.nbv_conf)
        self.o3dpcd_o = du.nparray2o3dpcd(pcd_o)
        self.o3dpcd_i = du.nparray2o3dpcd(pcd_i)
        self.o3dpcd_nbv = du.nparray2o3dpcd(np.asarray(self.nbv_pts))
        self.o3dpcd_o.paint_uniform_color(COLOR[2])
        self.o3dpcd_i.paint_uniform_color(COLOR[0])
        self.o3dpcd_nbv.colors = o3d.utility.Vector3dVector([[c, 0, 1 - c] for c in self.nbv_conf])

        kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_o, rgba=(1, 1, 0, 1), n_components=15)
        inp_pseq = nu.nurbs_inp(kpts)
        inp_rotseq = pcdu.get_rots_wkpts(pcd_o, inp_pseq, k=250, show=True, rgba=(1, 0, 0, 1))
        self.o3dmesh = du.cm2o3dmesh(bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008))
        self.o3dmesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dpcd_i, self.o3dmesh], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dmesh], mesh_show_back_face=True)
        # print(self.nbv_conf)
        # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=.05)
        # mesh_cylinder.translate(self.nbv_pts[0])
        # mesh_cylinder.rotate(rm.rotmat_between_vectors(np.asarray(self.campos) - x[3:], self.nbv_nrmls[0]))

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, pcd_i, campos, method='SLSQP'):
        """

        :param seedjntagls:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(pcd_i, campos)
        init_rot = rm.rotmat_between_vectors(np.asarray(self.campos) - self.nbv_pts[0], self.nbv_nrmls[0])
        sol = minimize(self.objctive, np.asarray(list(rm.rotmat_to_euler(init_rot)) + list(self.nbv_pts[0])),
                       method=method, bounds=self.bnds, constraints=self.cons)
        time_cost = time.time() - time_start
        print("time cost", time_cost, sol.success)

        if self.toggledebug:
            # print(sol)
            self.__debug()

        if sol.success:
            rot = rm.rotmat_from_euler(sol.x[0], sol.x[1], sol.x[2])
            trans = sol.x[3:]
        else:
            rot = rm.rotmat_between_vectors(np.asarray(self.campos) - self.nbv_pts[0], self.nbv_nrmls[0])
            rot = np.linalg.inv(rot)
            trans = self.nbv_pts[0]

        return trans, rot, time_cost

    def plot_vlist(self, ax, vlist, label=None, title=None, show=True):
        ax.plot(range(len(vlist)), vlist, label=label)
        if title is not None:
            ax.set_title(title)
        if show:
            plt.show()

    def __debug(self):
        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(321)
        self.plot_vlist(ax1, self.obj_list, title="objective", show=False)

        plt.show()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    nbs_opt = NBVOptimizer()
    base.run()
