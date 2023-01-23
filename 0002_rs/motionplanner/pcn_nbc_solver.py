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
import modeling.collision_model as cm
import basis.trimesh as trm
import basis.o3dhelper as o3dh

TB = True


def rbt2o3dmesh(rbt):
    rbtcm_list = rbt.gen_meshmodel().cm_list[3:]
    vertices = []
    vertex_normals = []
    faces = []
    for tmp_cm in rbtcm_list:
        tmp_vertices, tmp_vertex_normals, tmp_faces = tmp_cm.extract_rotated_vvnf()
        tmp_faces = np.asarray(tmp_faces) + len(vertices)
        vertices.extend(tmp_vertices)
        vertex_normals.extend(tmp_vertex_normals)
        faces.extend(list(tmp_faces))
    rbt_cm = cm.CollisionModel(initor=trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                                                  vertex_normals=np.asarray(vertex_normals)),
                               btwosided=True)
    return o3dh.cm2o3dmesh(rbt_cm)


class PCNNBCOptimizer(object):
    def __init__(self, rbt, max_a=np.pi / 6, max_dist=1, env=None, armname="arm", releemat4=np.eye(4), toggledebug=TB):
        self.rbt = rbt
        self.armname = armname
        self.env = env
        self.armname = armname
        self.releemat4 = releemat4
        self.toggledebug = toggledebug
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)
        self.rot_center = (0, 0, 0)
        self.result = None
        self.cons = []
        b = (-np.pi, np.pi)
        self.bnds = (b, b, b, b, b, b, b)

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
        self.o3dpcd_o, self.o3dmesh, self.o3dpcd_nbv = None, None, None
        self.campos = None

    def objctive(self, x, toggledebug=True):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.5)
        cam_mesh.translate(self.campos)

        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = np.linalg.inv(self.init_eemat4).dot(eemat4)
        rbt_o3dmesh = rbt2o3dmesh(self.rbt)
        rbt_o3dmesh.transform(np.linalg.inv(self.init_eemat4))
        conf_sum = 0
        o3dpcd_tmp = nu.gen_partial_o3dpcd(self.o3dmesh, toggledebug=toggledebug,
                                           trans=transmat4[:3, 3], rot=transmat4[:3, :3],
                                           rot_center=self.rot_center, othermesh=[rbt_o3dmesh])
        o3dpcd_tmp = o3dpcd_tmp.voxel_down_sample(voxel_size=0.005)
        pcd_tree = o3d.geometry.KDTreeFlann(o3dpcd_tmp)
        _, radius_vis_idx, _ = pcd_tree.search_radius_vector_3d(self.campos, .8)
        o3dpcd_tmp = o3dpcd_tmp.select_by_index(radius_vis_idx)

        pts = np.asarray([p for p in np.asarray(o3dpcd_tmp.points)
                          if rm.angle_between_vectors(self.campos - p, self.campos) < np.pi / 3])
        o3dpcd_tmp = du.nparray2o3dpcd(pts)

        o3dpcd_tmp.paint_uniform_color((0, 1, 0))
        kdt_nbv = o3d.geometry.KDTreeFlann(self.o3dpcd_nbv)

        if toggledebug:
            o3d.visualization.draw_geometries([rbt_o3dmesh, o3dpcd_tmp, self.o3dmesh, coord])
            o3d.visualization.draw_geometries([o3dpcd_tmp, self.o3dpcd_nbv, coord])
            # o3d.visualization.draw_geometries([o3dpcd_tmp, self.o3dpcd_nbv], mesh_show_back_face=True)

        for p in np.asarray(o3dpcd_tmp.points):
            _, idx, _ = kdt_nbv.search_knn_vector_3d(p, 1)
            if np.linalg.norm(p - self.nbv_pts[idx]) < .01 and self.nbv_conf[idx] < .5:
                conf_sum += 1 - (self.nbv_conf[idx])
        self.obj_list.append(conf_sum)
        print(x, conf_sum)
        return -conf_sum

    def update_known(self, seedjntagls, pcd_i, campos):
        model_name = 'pcn'
        load_model = 'pcn_emd_all/best_cd_p_network.pth'

        width = .008
        thickness = .002
        cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
        pcd_i = pcdu.trans_pcd(pcd_i, np.linalg.inv(self.releemat4))
        self.campos = campos
        self.seedjntagls = seedjntagls
        self.init_eepos, self.init_eerot = self.rbt.get_gl_tcp()
        self.init_eemat4 = rm.homomat_from_posrot(self.init_eepos, self.init_eerot)

        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        pcd_i = pcdu.trans_pcd(pcd_i, self.releemat4)
        pcd_o = pcdu.trans_pcd(pcd_o, self.releemat4)
        self.nbv_pts, self.nbv_nrmls, self.nbv_conf = pcdu.cal_pcn(pcd_i, pcd_o, cam_pos=campos, theta=None,
                                                                   toggledebug=True)
        self.o3dpcd_o = du.nparray2o3dpcd(pcd_o)
        self.o3dpcd_nbv = du.nparray2o3dpcd(np.asarray(self.nbv_pts))
        self.o3dpcd_nbv.colors = o3d.utility.Vector3dVector([[c, 0, 1 - c] for c in self.nbv_conf])
        # self.o3dmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(self.o3dpcd_o, .005)
        kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_o, rgba=(1, 1, 0, 1), n_components=16)
        inp_pseq = nu.kpts2bspl(kpts)
        inp_rotseq = pcdu.get_rots_wkpts(pcd_o, inp_pseq, k=200, show=True, rgba=(1, 0, 0, 1))
        self.o3dmesh = du.cm2o3dmesh(bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008))
        self.o3dmesh.compute_vertex_normals()

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
        # self.addconstraint(self.con_rot, condition="ineq")
        # self.addconstraint(self.con_dist, condition="ineq")
        self.addconstraint(self.con_diff_x, condition="ineq")
        self.addconstraint(self.con_diff_y, condition="ineq")
        self.addconstraint(self.con_diff_z, condition="ineq")
        sol = minimize(self.objctive, seedjntagls, method=method, bounds=self.bnds, constraints=self.cons)

        print("time cost", time.time() - time_start, sol.success)

        if self.toggledebug:
            # print(sol)
            self.__debug()

        if sol.success:
            return sol.x
        else:
            return None

        # self.rbth.goto_armjnts(sol.x)
        # eepos, eerot = self.rbt.get_gl_tcp()
        # eemat4 = rm.homomat_from_posrot(eepos, eerot)
        # transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        # gm.gen_frame(eepos, eerot).attach_to(base)
        # gm.gen_frame(self.init_eepos, self.init_eerot).attach_to(base)

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
    seedjntagls = rbt.get_jnt_values(component_name='arm')
    pcn_nbc_opt = PCNNBCOptimizer(rbt)
    pcn_nbc_opt.solve(seedjntagls, pcd_i, campos, method='SLSQP')
    base.run()
