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
    def __init__(self, rbt, max_a=np.pi / 6, max_dist=1, env=None, armname="arm", releemat4=np.eye(4),
                 model_name='pcn', load_model='pcn_emd_rec/best_emd_network.pth', toggledebug=TB):
        self.rbt = rbt
        self.armname = armname
        self.model_name, self.load_model = model_name, load_model
        self.env = env
        self.armname = armname
        self.releemat4 = releemat4
        self.toggledebug = toggledebug
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)
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

        self.max_a = max_a
        self.max_dist = max_dist

        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = [], [], []
        self.init_eepos, self.init_eerot, self.init_eemat4 = None, None, None
        self.campos = None

    def objctive(self, x):
        # self.jnts.append(x)
        # self.rbth.goto_armjnts(x)
        conf_sum = 0
        rot = rm.rotmat_from_euler(x[0], x[1], x[2])
        o3dpcd_tmp = nu.gen_partial_o3dpcd(self.o3dmesh, trans=x[3:], rot=rot, rot_center=self.rot_center)
        o3dpcd_tmp = o3dpcd_tmp.voxel_down_sample(voxel_size=0.005)
        o3dpcd_tmp.paint_uniform_color((0, 1, 0))
        kdt_nbv = o3d.geometry.KDTreeFlann(self.o3dpcd_nbv)

        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dpcd_i, self.o3dmesh, self.mesh_cylinder],
        #                                   mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([o3dpcd_tmp, self.o3dpcd_nbv], mesh_show_back_face=True)
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

    def update_known(self, seedjntagls, pcd_i, campos):
        def _sigmoid(x):
            s = 1 / (1 + np.exp(1 - 2 * x))
            return s

        width = .008
        thickness = .0015
        cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

        self.campos = campos
        self.seedjntagls = seedjntagls
        self.init_eepos, self.init_eerot = self.rbt.get_gl_tcp()
        self.init_eemat4 = rm.homomat_from_posrot(self.init_eepos, self.init_eerot)

        pcd_o = pcn.inference_sgl(pcd_i, self.model_name, self.load_model, toggledebug=False)
        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = \
            pcdu.cal_pcn(pcd_i, pcd_o, cam_pos=campos, theta=None, toggledebug=True)
        # print(self.nbv_conf)
        # self.nbv_conf = _sigmoid(self.nbv_conf)
        # print(self.nbv_conf)
        self.o3dpcd_o = du.nparray2o3dpcd(pcd_o)
        self.o3dpcd_i = du.nparray2o3dpcd(pcd_i)
        self.o3dpcd_nbv = du.nparray2o3dpcd(np.asarray(self.nbv_pts))
        self.o3dpcd_o.paint_uniform_color(COLOR[2])
        self.o3dpcd_i.paint_uniform_color(COLOR[0])
        self.o3dpcd_nbv.colors = o3d.utility.Vector3dVector([[c, 0, 1 - c] for c in self.nbv_conf])

        kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_o, rgba=(1, 1, 0, 1), n_components=10)
        inp_pseq = nu.kpts2bspl(kpts)
        inp_rotseq = pcdu.get_rots_wkpts(pcd_o, inp_pseq, k=250, show=True, rgba=(1, 0, 0, 1))
        self.o3dmesh = du.cm2o3dmesh(bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008))
        self.o3dmesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dpcd_i, self.o3dmesh], mesh_show_back_face=True)
        # print(self.nbv_conf)
        # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=.05)
        # mesh_cylinder.translate(self.nbv_pts[0])
        # mesh_cylinder.rotate(rm.rotmat_between_vectors(np.asarray(self.campos) - x[3:], self.nbv_nrmls[0]))

    def con_rot(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        n_new = pcdu.trans_pcd([self.nbv_nrmls[0]], transmat4)[0]
        p_new = pcdu.trans_pcd([self.nbv_pts[0]], transmat4)[0]
        err = rm.angle_between_vectors(n_new, self.campos - p_new)
        self.rot_err.append(err)
        return self.max_a - err

    def con_dist(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.linalg.norm(np.asarray(eepos) - self.campos)
        self.pos_err.append(err)
        return self.max_dist - err

    def con_diff_x(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[0] - self.init_eepos[0])
        return .1 - err

    def con_diff_y(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[1] - self.init_eepos[1])
        return .1 - err

    def con_diff_z(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[2] - self.init_eepos[2])
        return .1 - err

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, seedjntagls, pcd_i, campos, method='SLSQP'):
        """

        :param seedjntagls:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(seedjntagls, pcd_i, campos)
        init_rot = rm.rotmat_between_vectors(np.asarray(self.campos) - self.nbv_pts[0], self.nbv_nrmls[0])
        sol = minimize(self.objctive, np.asarray(list(rm.rotmat_to_euler(init_rot)) + list(self.nbv_pts[0])),
                       method=method, bounds=self.bnds, constraints=self.cons)

        # print("time cost", time.time() - time_start, sol.success)

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

        # self.rbth.goto_armjnts(sol.x)
        # eepos, eerot = self.rbt.get_gl_tcp()
        # eemat4 = rm.homomat_from_posrot(eepos, eerot)
        # transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        # gm.gen_frame(eepos, eerot).attach_to(base)
        # gm.gen_frame(self.init_eepos, self.init_eerot).attach_to(base)

        return trans, rot

    def __debug(self):
        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(321)
        self.rbth.plot_vlist(ax1, self.obj_list, title="objective", show=False)
        ax2 = plt.subplot(322)
        self.rbth.plot_vlist(ax2, self.rot_err, title="normal to cam angle", show=False)
        ax3 = plt.subplot(323)
        self.rbth.plot_vlist(ax3, self.jd_list, title="joints displacement", show=False)
        ax4 = plt.subplot(324)
        self.rbth.plot_vlist(ax4, self.mp_list, title="manipulability", show=False)
        ax5 = plt.subplot(325)
        self.rbth.plot_vlist(ax5, self.sr_list, title="line of sight", show=False)
        ax6 = plt.subplot(326)
        self.rbth.plot_vlist(ax6, self.wo_list, title="wrist obstruction", show=False)
        # ax6 = plt.subplot(326)
        # self.rbth.plot_armjnts(ax6, self.jnts, show=False)
        plt.show()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import robot_sim.robots.xarm_shuidi.xarm_shuidi as xarm_shuidi

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    rbt = el.loadXarm(showrbt=True)
    nbs_opt = NBVOptimizer(rbt)
    base.run()
