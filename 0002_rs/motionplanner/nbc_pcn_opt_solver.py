import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import minimize

import basis.robot_math as rm
import bendplanner.bend_utils as bu
import datagenerator.data_utils as du
import motionplanner.robot_helper as rbt_helper
import nbv.nbv_utils as nu
import pcn.inference as pcn
import utils.pcd_utils as pcdu

TB = True


class PCNNBCOptimizer(object):
    def __init__(self, rbt, max_dist=1, env=None, armname="arm", releemat4=np.eye(4), toggledebug=TB):
        self.rbt = rbt
        self.rbt.jaw_to(jawwidth=0)
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
        self.ref_list = []
        self.obj_list = []

        self.max_dist = max_dist
        self.conf_tresh = .5

        self.pts_nbv, self.nrmls_nbv, self.nbv_conf = [], [], []
        self.init_eepos, self.init_eerot, self.init_eemat4 = None, None, None
        self.o3dpcd_o, self.o3dmesh, self.o3dpcd_nbv = None, None, None
        self.laser_pos, self.cam_pos, self.cam_mat4 = (0, 0, 0), (0, 0, 0), np.eye(3)

    def objective(self, x):
        self.jnts.append(x)
        self.rbth.goto_armjnts(x)
        rbt_o3dmesh = nu.rbt2o3dmesh(self.rbt, link_num=10, show_nrml=False)
        if self.toggledebug:
            rbt_o3dmesh.compute_vertex_normals()
        conf_sum = 0

        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = np.linalg.inv(self.init_eemat4).dot(eemat4)

        rbt_o3dmesh.transform(np.linalg.inv(self.init_eemat4))
        o3dpcd_tmp_origin = \
            nu.gen_partial_o3dpcd(self.o3dmesh, toggledebug=False, othermesh=[rbt_o3dmesh],
                                  trans=transmat4[:3, 3], rot=transmat4[:3, :3], rot_center=self.rot_center,
                                  fov=True, vis_threshold=np.radians(45),
                                  cam_mat4=self.cam_mat4)
        o3dpcd_tmp_origin.paint_uniform_color(nu.COLOR[5])

        if self.toggledebug:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
            coord_tmp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
            coord_tmp.transform(transmat4)
            cam_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=.005)
            cam_mesh.translate(self.cam_pos)
            coord_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
            coord_cam.transform(self.cam_mat4)
            rbt_o3dmesh_all = nu.rbt2o3dmesh(self.rbt, link_num=3, show_nrml=True)
            rbt_o3dmesh_all.transform(np.linalg.inv(self.init_eemat4))
            o3dmesh_tmp = copy.deepcopy(self.o3dmesh)
            o3dmesh_tmp.transform(transmat4)
            o3dpcd_nxt = copy.deepcopy(o3dpcd_tmp_origin)
            o3dpcd_nxt.transform(transmat4)
            o3d.visualization.draw_geometries(
                [rbt_o3dmesh_all, o3dpcd_nxt, o3dmesh_tmp, coord, coord_tmp, cam_mesh, coord_cam])
            o3d.visualization.draw_geometries([o3dpcd_tmp_origin, self.o3dpcd_nbv, coord])

        if len(np.asarray(o3dpcd_tmp_origin.points)) == 0:
            self.obj_list.append(conf_sum)
            return 0

        kdt_tmp = o3d.geometry.KDTreeFlann(o3dpcd_tmp_origin)
        nrmls_tmp = np.asarray(o3dpcd_tmp_origin.normals)
        for i in range(len(self.pts_nbv)):
            if self.nbv_conf[i] > self.conf_tresh:
                continue
            _, idx, _ = kdt_tmp.search_radius_vector_3d(self.pts_nbv[i], .01)
            conf_sum += (1 - (self.nbv_conf[i])) * len(idx) / 10
            # for j in idx:
            #     a = rm.angle_between_vectors(nrmls_tmp[j], self.cam_mat4[:3, 2])
            #     if a > np.pi / 9:
            #         conf_sum += (1 - (self.nbv_conf[i])) / 10

            # if len(idx) > 50:
            #     conf_sum += 1 - (self.nbv_conf[i])
        self.obj_list.append(conf_sum)
        if self.toggledebug:
            print(x, conf_sum)
        return -conf_sum

    def update_known(self, seedjntagls, pcd_i, cam_mat4, conf_tresh):
        model_name = 'pcn'
        load_model = 'pcn_emd_all/best_cd_p_network.pth'

        width = .008
        thickness = .002
        cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
        pcd_i = pcdu.trans_pcd(pcd_i, np.linalg.inv(self.releemat4))
        self.conf_tresh = conf_tresh
        self.seedjntagls = seedjntagls
        self.init_eepos, self.init_eerot = self.rbt.get_gl_tcp()
        self.init_eemat4 = rm.homomat_from_posrot(self.init_eepos, self.init_eerot)
        self.cam_mat4 = np.dot(np.linalg.inv(self.init_eemat4), cam_mat4)
        self.cam_pos = self.cam_mat4[:3, 3]
        self.laser_pos = self.cam_mat4[:3, 3] - .175 * rm.unit_vector(self.cam_mat4[:3, 0]) \
                         + .049 * rm.unit_vector(self.cam_mat4[:3, 1]) \
                         + .01 * rm.unit_vector(self.cam_mat4[:3, 2])

        pcd_o = pcn.inference_sgl(pcd_i, model_name, load_model, toggledebug=False)
        pcd_i_inhnd = pcdu.trans_pcd(pcd_i, self.releemat4)
        pcd_o_inhnd = pcdu.trans_pcd(pcd_o, self.releemat4)
        self.pts_nbv, self.nrmls_nbv, self.nbv_conf = \
            pcdu.cal_nbv_pcn(pcd_i_inhnd, pcd_o_inhnd, cam_pos=self.cam_pos, icp=True, theta=None, toggledebug=True)
        self.o3dpcd_o = du.nparray2o3dpcd(pcd_o_inhnd)
        self.o3dpcd_nbv = du.nparray2o3dpcd(np.asarray(self.pts_nbv))
        self.o3dpcd_nbv.colors = o3d.utility.Vector3dVector([[c, 0, 1 - c] for c in self.nbv_conf])
        kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_o_inhnd, rgba=(1, 1, 0, 1), n_components=16)
        inp_pseq = nu.nurbs_inp(kpts)
        inp_rotseq = pcdu.get_rots_wkpts(pcd_o_inhnd, inp_pseq, k=200, show=True, rgba=(1, 0, 0, 1))
        self.o3dmesh = du.cm2o3dmesh(bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008))
        # self.o3dmesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([self.o3dpcd_o, self.o3dmesh])

    def con_dist(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.linalg.norm(np.asarray(eepos) - self.cam_pos)
        self.pos_err.append(err)
        return self.max_dist - err

    def con_diff_x(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.asarray(eepos)[0] - self.init_eepos[0]
        if err > 0:
            return .1 - err
        if err <= 0:
            return .05 + err

    def con_diff_y(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = abs(np.asarray(eepos)[1] - self.init_eepos[1])
        return .12 - err

    def con_diff_z(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        err = np.asarray(eepos)[2] - self.init_eepos[2]
        if err > 0:
            return .1 - err
        if err <= 0:
            return .04 + err

    def con_cost(self, x):
        w_e = np.linalg.norm(x - self.seedjntagls)
        return 1 - w_e

    def con_collision(self, x):
        flag = self.rbth.is_selfcollided(x)
        print(flag, .5 - flag)
        return .5 - flag

    def con_reflection(self, x):
        self.rbth.goto_armjnts(x)
        eepos, eerot = self.rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot)
        transmat4 = eemat4.dot(np.linalg.inv(self.init_eemat4))
        # n_new = transmat4[:3, :3].dot(self.nrml_nbv)
        # err = rm.angle_between_vectors(n_new, self.cam_mat4[:3, 1])
        # err = min([err, np.pi - err])
        pts_new = pcdu.trans_pcd(self.pts_nbv, transmat4)
        nrmls_new = np.asarray([transmat4[:3, :3].dot(n) for n in self.nrmls_nbv])
        err = min([min(rm.angle_between_vectors(p - self.laser_pos, nrmls_new[i]),
                       np.pi - rm.angle_between_vectors(p - self.laser_pos, nrmls_new[i]))
                   for i, p in enumerate(pts_new)])
        self.ref_list.append(np.degrees(err))
        return err - np.pi / 9

    def con_manipulability(self, x):
        self.rbth.goto_armjnts(x)
        return 100 - self.rbt.manipulability(component_name='arm')

    def addconstraint(self, constraint, condition="ineq"):
        self.cons.append({'type': condition, 'fun': constraint})

    def solve(self, seedjntagls, pcd_i, cam_mat4, conf_tresh=.2, method='SLSQP'):
        """

        :param seedjntagls:
        :param method: 'SLSQP' or 'COBYLA'
        :return:
        """
        time_start = time.time()
        self.update_known(seedjntagls, pcd_i, cam_mat4, conf_tresh)
        self.addconstraint(self.con_dist, condition="ineq")
        self.addconstraint(self.con_diff_x, condition="ineq")
        self.addconstraint(self.con_diff_y, condition="ineq")
        self.addconstraint(self.con_diff_z, condition="ineq")

        sol = minimize(self.objective, seedjntagls, method=method, bounds=self.bnds, constraints=self.cons)
        time_cost = time.time() - time_start
        print("time cost", time_cost, sol.success)

        if self.toggledebug:
            print(sol)
            self.__debug()

        if sol.success:
            self.rbth.goto_armjnts(sol.x)
            eepos, eerot = self.rbt.get_gl_tcp()
            eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(self.releemat4)
            transmat4 = np.linalg.inv(self.init_eemat4).dot(eemat4)
            # gm.gen_frame(eepos, eerot).attach_to(base)
            # gm.gen_frame(self.init_eepos, self.init_eerot).attach_to(base)
            print(self.obj_list[0], self.obj_list[-1])
            return sol.x, transmat4, sol.fun, time_cost
        else:
            return None, None, None, time_cost

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
