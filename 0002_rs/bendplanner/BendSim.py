import copy
import math
import random

import numpy as np

import basis.robot_math as rm
import basis.trimesh as trm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import modeling.collision_model as cm
import modeling.geometric_model as gm
import utils.panda3d_utils as p3u


def draw_plane(p, n):
    pt_direction = rm.orthogonal_vector(n, toggle_unit=True)
    tmp_direction = np.cross(n, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, n))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array(p)
    gm.gen_box(np.array([.2, .2, .0005]), homomat=homomat, rgba=[1, 1, 1, .3]).attach_to(base)


class BendSim(object):
    def __init__(self, pseq=None, rotseq=None, show=False, granularity=np.pi / 90, cm_type='stick'):
        # bending device prop
        self.r_side = bconfig.R_SIDE
        self.r_center = bconfig.R_CENTER
        self.r_base = bconfig.R_BASE
        self.c2c_dist = bconfig.C2C_DIST
        self.init_l = bconfig.INIT_L

        # bending meterial prop
        self.thickness = bconfig.THICKNESS
        self.width = bconfig.WIDTH
        self.cm_type = cm_type
        self.stick_sec = 5

        # bending device prop
        self.bend_r = self.r_center + self.thickness / 2
        self.punch_pillar_init = self.cal_starta() / 2
        self.slot_w = self.c2c_dist * np.cos(self.punch_pillar_init / 2) - self.r_center - self.r_side - .001
        self.bender = cm.CollisionModel(f'{config.ROOT}/obstacles/bender_smp.stl', cdprimit_type='polygons',
                                        name='bender')
        self.bender.set_rgba((.5, .5, .5, 1))

        # bending result
        self.objcm = None
        if pseq is None:
            self.pseq = [[self.bend_r, 0, 0]]
            self.rotseq = [np.eye(3)]
        else:
            self.pseq = pseq
            self.pseq.append((0, self.pseq[-1][1] + bconfig.INIT_L, 0))
            self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
            self.rotseq = rotseq
            self.rotseq.append(np.eye(3))

        # gen pillars
        sections = 90
        self.pillar_center = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H]),
                                          epos=np.asarray([0, 0, bconfig.PILLAR_H]),
                                          thickness=self.r_center * 2, sections=sections,
                                          rgba=[.9, .9, .9, 1])
        self.pillar_center_inner = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H]),
                                                epos=np.asarray([0, 0, bconfig.PILLAR_H]),
                                                thickness=self.r_center * 2 - .001, sections=sections,
                                                rgba=[.7, .7, .7, .7])
        self.pillar_dieside = cm.gen_stick(spos=np.asarray([self.c2c_dist, 0, -bconfig.PILLAR_H]),
                                           epos=np.asarray([self.c2c_dist, 0, bconfig.PILLAR_H]),
                                           thickness=self.r_side * 2, sections=sections,
                                           rgba=[.9, .9, .9, 1])
        self.pillar_dieside_inner = cm.gen_stick(spos=np.asarray([self.c2c_dist, 0, -bconfig.PILLAR_H]),
                                                 epos=np.asarray([self.c2c_dist, 0, bconfig.PILLAR_H]),
                                                 thickness=self.r_side * 2 - .001, sections=sections,
                                                 rgba=[.7, .7, .7, .7])
        self.pillar_punch = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H]),
                                         epos=np.asarray([0, 0, bconfig.PILLAR_H]),
                                         thickness=self.r_side * 2, sections=sections,
                                         rgba=[.7, .7, .7, 1])
        self.pillar_punch_end = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H]),
                                             epos=np.asarray([0, 0, bconfig.PILLAR_H]),
                                             thickness=self.r_side * 2, sections=sections,
                                             rgba=[.7, .7, 0, .7])
        self.pillar_punch.set_homomat(
            rm.homomat_from_posrot([self.c2c_dist * np.cos(self.punch_pillar_init),
                                    self.c2c_dist * np.sin(self.punch_pillar_init), 0], np.eye(3)))
        # others
        self.granularity = granularity
        # obstacles
        self.__staticobslist = [self.bender, self.pillar_center_inner, self.pillar_dieside_inner]
        if show:
            gm.gen_frame(thickness=.001, length=.05, alpha=.1).attach_to(base)
            for a in np.arange(0, 2 * math.pi, math.pi / 360):
                gm.gen_sphere(pos=[self.c2c_dist * math.cos(a), self.c2c_dist * math.sin(a), 0], radius=.0001,
                              rgba=(.7, .7, .7, .2)).attach_to(base)
            self.pillar_dieside.attach_to(base)
            self.pillar_punch.attach_to(base)
            self.pillar_center.attach_to(base)
            for obj in self.__staticobslist:
                obj.attach_to(base)
            # self.bender.show_cdprimit()

    def reset(self, pseq, rotseq, extend=True):
        self.pseq = list(copy.deepcopy(pseq))
        self.rotseq = list(copy.deepcopy(rotseq))
        if extend:
            # self.pseq.append((0, self.pseq[-1][1] + 2 * np.pi * self.bend_r, 0))
            self.pseq.append((0, self.pseq[-1][1] + .6 * np.pi * self.bend_r, 0))
            # self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
            self.rotseq.append(np.eye(3))
        self.update_cm()

    def set_r_center(self, r):
        self.r_center = r
        self.bend_r = self.r_center + self.thickness / 2
        self.slot_w = self.c2c_dist * np.cos(self.punch_pillar_init / 2) - self.r_center - self.r_side - .001
        self.init_l = self.bend_r * np.pi

    def set_stick_sec(self, stick_sec):
        self.stick_sec = stick_sec

    def staticobs_list(self):
        return self.__staticobslist

    def add_staticobs(self, objcm):
        self.__staticobslist.append(objcm)

    def reset_staticobs(self):
        self.__staticobslist = [self.bender, self.pillar_center_inner, self.pillar_dieside_inner]

    def move_posrot(self, transmat4):
        # pillar_dieside = self.pillar_dieside.localframe()
        # print(pillar_dieside)
        self.pillar_dieside.set_homomat(transmat4)
        self.pillar_dieside_inner.set_homomat(transmat4)
        self.pillar_center.set_homomat(transmat4)
        self.pillar_center_inner.set_homomat(transmat4)
        self.bender.set_homomat(transmat4)

    def cal_minarm(self):
        A = np.mat([[self.r_side + self.thickness, -self.r_center], [1, 1]])
        b = np.mat([0, self.c2c_dist]).T
        l_center, l_side = np.asarray(np.linalg.solve(A, b))

        return l_center[0], l_side[0], \
            np.sqrt(l_center[0] ** 2 - self.r_center ** 2) + \
            np.sqrt(l_side[0] ** 2 - (self.r_side + self.thickness) ** 2)

    def cal_minlim(self):
        return 2 * np.arcsin((self.r_side + self.r_base) / (2 * self.c2c_dist))

    def cal_starta(self):
        l_center, _, _ = self.cal_minarm()
        return 2 * np.arccos(self.r_center / l_center)

    def cal_start(self, dir=1, toggledebug=False):
        plate_angle = self.cal_plate_die_angle(dir, toggledebug)
        if plate_angle is None:
            print('No collided point found (die & plate)!')
            return None, None
        punch_sangle = self.cal_punch_angle(dir, copy.deepcopy(self.objcm), toggledebug)
        if punch_sangle is None:
            print('No collided point found (punch & plate)!')
            return None, None
        return punch_sangle, plate_angle

    def cal_plate_die_angle(self, dir, toggledebug=False):
        range_plate = np.linspace(0, np.pi, 360) if dir == 1 else np.linspace(0, -np.pi, 360)
        step_plate = range_plate[1] - range_plate[0]
        for i in range_plate:
            tmp_p = np.asarray([self.bend_r * math.cos(i), self.bend_r * math.sin(i), 0])
            transmat4 = rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((0, 0, 1), step_plate))
            self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
            self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])
            self.update_cm()
            # self.show(show_frame=True)
            objcm_bend = copy.deepcopy(self.objcm)
            is_collided, collided_pts = self.pillar_dieside.is_mcdwith(objcm_bend, toggle_contacts=True)
            if is_collided:
                if toggledebug:
                    bu.show_pseq(collided_pts, radius=.0004)
                    objcm_bend.set_rgba(rgba=[0, 0, 1, .7])
                    objcm_bend.attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray(tmp_p),
                                     rgba=[.7, .7, .7, .7],
                                     thickness=.0005).attach_to(base)
                return i
        return None

    def cal_punch_angle(self, dir, objcm, toggledebug=False):
        range_punch = np.linspace(self.punch_pillar_init, 2 * np.pi - self.punch_pillar_init, 360) if dir == 1 \
            else np.linspace(-self.punch_pillar_init, -2 * np.pi + - self.punch_pillar_init, 360)
        for i in range_punch:
            tmp_p = np.asarray([self.c2c_dist * math.cos(i), self.c2c_dist * math.sin(i), 0])
            self.pillar_punch.set_homomat(rm.homomat_from_posrot(tmp_p, np.eye(3)))
            self.pillar_punch.set_rgba(rgba=[0, .7, .7, .7])
            is_collided, collided_pts = self.pillar_punch.is_mcdwith(objcm, toggle_contacts=True)
            if is_collided:
                if toggledebug:
                    bu.show_pseq(collided_pts, radius=.0004)
                    copy.deepcopy(self.pillar_punch).attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray((collided_pts[-1][0], collided_pts[-1][1], 0)),
                                     rgba=[0, .7, .7, .7],
                                     thickness=.0005).attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray(tmp_p),
                                     rgba=[0, .7, .7, .7],
                                     thickness=.0005).attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray((self.c2c_dist, 0, 0)),
                                     rgba=[0, .7, .7, .7],
                                     thickness=.0005).attach_to(base)
                return i
        return None

    def _get_bended_pseq(self, start_inx, end_inx, bend_angle, lift_angle, toggledebug=True):
        tmp_pseq = []
        tmp_rotseq = []
        n = np.asarray([0, 0, 1])
        y = np.asarray([0, 1, 0])
        if bend_angle > 0:
            srange = np.linspace(0, bend_angle, int(bend_angle / self.granularity) + 1)
        else:
            srange = np.linspace(bend_angle, 0, int(-bend_angle / self.granularity) + 1)[::-1]
        pseq_org = self.pseq[start_inx:end_inx + 1]
        rotseq_org = self.rotseq[start_inx:end_inx + 1]

        step_l = abs(bend_angle * self.bend_r / np.cos(lift_angle)) / len(srange)
        init_a = round(np.arctan(pseq_org[0][1] / pseq_org[0][0]), 5)
        if bend_angle > 0 and init_a < 0:
            init_a = np.pi + init_a
        elif bend_angle < 0 and init_a > 0:
            init_a = np.pi - init_a
        for i, a in enumerate(srange):
            _, insert_inx, insert_pos, insert_rot = self.get_posrot_by_l(step_l * i, pseq_org, rotseq_org)
            p = np.asarray((self.bend_r * math.cos(init_a + a), self.bend_r * math.sin(init_a + a), insert_pos[2]))
            tmp_pseq.append(p)
            if i > 0:
                ext_dir = np.dot(insert_rot, y)
                v_tmp = np.dot(rm.rotmat_from_axangle(n, init_a), np.asarray([-np.sin(a), np.cos(a), 0]))
                if a < 0:
                    v_tmp = -v_tmp
                a = rm.angle_between_vectors(v_tmp - n * v_tmp, ext_dir - n * ext_dir)
                if np.cross(v_tmp - n * v_tmp, ext_dir - n * ext_dir)[2] > 0:
                    a = -a
            tmp_rotseq.append(rm.rotmat_from_axangle((0, 0, 1), a).dot(insert_rot))
            if toggledebug:
                gm.gen_sphere(np.asarray(tmp_pseq[-1]), rgba=[1, 0, 0, .5], radius=0.0002).attach_to(base)
                gm.gen_frame(np.asarray(tmp_pseq[-1]), tmp_rotseq[-1], length=.005,
                             thickness=.0004, alpha=.5).attach_to(base)
            if insert_inx == -1:
                break
        return tmp_pseq, tmp_rotseq

    def bend(self, bend_angle, lift_angle=0, bend_pos=None, toggledebug=False):
        arc_l = abs(bend_angle * self.bend_r / np.cos(lift_angle))
        if abs(lift_angle) > np.pi / 2:
            bend_pos = self._cal_length() - bend_pos - arc_l

        start_inx = self._insert_p(bend_pos, toggledebug=False)
        end_inx = self._insert_p(bend_pos + arc_l, toggledebug=False)
        tmp_pseq, tmp_rotseq = self._get_bended_pseq(start_inx, end_inx, bend_angle=bend_angle, lift_angle=lift_angle,
                                                     toggledebug=toggledebug)
        if toggledebug:
            gm.gen_sphere(self.pseq[start_inx], radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(self.pseq[start_inx], self.rotseq[start_inx], length=.01, thickness=.0004).attach_to(base)
            gm.gen_sphere(self.pseq[end_inx], radius=.0004, rgba=(0, 1, 1, 1)).attach_to(base)
            gm.gen_frame(self.pseq[end_inx], self.rotseq[end_inx], length=.01, thickness=.0004).attach_to(base)
        init_homomat = rm.homomat_from_posrot(tmp_pseq[0], tmp_rotseq[0])
        start_homomat = rm.homomat_from_posrot(self.pseq[start_inx], self.rotseq[start_inx])
        end_homomat = rm.homomat_from_posrot(self.pseq[end_inx], self.rotseq[end_inx])

        transmat4_mid = np.dot(start_homomat, np.linalg.inv(init_homomat))
        # gm.gen_frame(np.asarray([.05, 0, 0]), start_homomat[:3, :3], length=.01, thickness=.0004).attach_to(base)
        # gm.gen_frame(np.asarray([.05, 0, 0]), init_homomat[:3, :3], length=.01, thickness=.0004).attach_to(base)
        pseq_mid = rm.homomat_transform_points(transmat4_mid, tmp_pseq).tolist()
        rotseq_mid = [np.dot(transmat4_mid[:3, :3], r) for r in tmp_rotseq]

        transmat4_end = np.dot(rm.homomat_from_posrot(pseq_mid[-1], rotseq_mid[-1]), np.linalg.inv(end_homomat))
        pseq_end = rm.homomat_transform_points(transmat4_end, self.pseq[end_inx + 1:]).tolist()
        rotseq_end = [np.dot(transmat4_end[:3, :3], r) for r in self.rotseq[end_inx + 1:]]

        # gm.gen_frame(pseq_mid[-1], rotseq_mid[-1], length=.012, thickness=.0004,
        #              rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        # gm.gen_frame(pseq_end[0], rotseq_end[0], length=.01, thickness=.0004).attach_to(base)
        self.pseq = self.pseq[:start_inx] + pseq_mid + pseq_end
        self.rotseq = list(self.rotseq[:start_inx]) + list(rotseq_mid) + list(rotseq_end)
        self.update_cm()

    def _trans_pos(self, pts, pos):
        return rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray(pos), np.eye(3)), pts)

    def _rot_new_orgin(self, pts, new_orgin, rot):
        trans_pts = self._trans_pos(pts, new_orgin)
        trans_pts = rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray([0, 0, 0]), rot), trans_pts)
        return self._trans_pos(trans_pts, -new_orgin).tolist()

    def get_posrot_by_l(self, pos, pseq, rotseq):
        tmp_l = 0
        insert_inx, insert_pos, insert_rot = 0, pseq[0], rotseq[0]
        if pos == 0:
            return False, insert_inx, insert_pos, insert_rot
        for i in range(len(pseq) - 1):
            p1 = np.asarray(pseq[i])
            p2 = np.asarray(pseq[i + 1])
            r1 = rotseq[i]
            r2 = rotseq[i + 1]
            tmp_l += np.linalg.norm(p2 - p1)
            if tmp_l < pos:
                continue
            elif tmp_l > pos:
                insert_radio = (tmp_l - pos) / np.linalg.norm(p2 - p1)
                insert_pos = p2 - insert_radio * (p2 - p1)
                if (r1 == r2).all():
                    insert_rot = r1
                else:
                    rotmat_list = rm.rotmat_slerp(r1, r2, 10)
                    inx = np.floor(insert_radio * len(rotmat_list)) - 1
                    if inx > 9:
                        inx = 9
                    insert_rot = rotmat_list[int(inx)]
                insert_inx = i + 1
                return True, insert_inx, insert_pos, insert_rot
            else:
                insert_pos = pseq[i + 1]
                insert_rot = rotseq[i + 1]
                insert_inx = i + 1
                return False, insert_inx, insert_pos, insert_rot
        return False, -1, pseq[-1], rotseq[-1]

    def _insert_p(self, pos, toggledebug=False):
        flag, insert_inx, insert_pos, insert_rot = self.get_posrot_by_l(pos, self.pseq, self.rotseq)
        if flag:
            self.pseq = self.pseq[:insert_inx] + [insert_pos] + self.pseq[insert_inx:]
            self.rotseq = list(self.rotseq[:insert_inx]) + [insert_rot] + list(self.rotseq[insert_inx:])
        if toggledebug:
            gm.gen_sphere(insert_pos, radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(insert_pos, insert_rot, length=.01, thickness=.0004).attach_to(base)
        return insert_inx

    def update_cm(self):
        # ts = time.time()
        if self.cm_type == 'stick':
            vertices, faces = self.gen_stick(self.pseq[::-1], self.rotseq[::-1], self.thickness / 2,
                                             section=self.stick_sec)
        else:
            vertices, faces = self.gen_swap(self.pseq[::-1], self.rotseq[::-1])
        objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        self.objcm = cm.CollisionModel(initor=objtrm, btwosided=True, name='obj', cdprimit_type='surface_balls')
        # print('time cost(update cm):', time.time() - ts)

    def show(self, rgba=(1, 1, 1, 1), objmat4=None, show_frame=False, show_pseq=False, show_dashframe=False):
        self.update_cm()
        objcm, pseq, rotseq = copy.deepcopy(self.objcm), copy.deepcopy(self.pseq), copy.deepcopy(self.rotseq)
        if objmat4 is not None:
            transmat4 = np.dot(objmat4, np.linalg.inv(rm.homomat_from_posrot(self.pseq[0], self.rotseq[0])))
            pseq = rm.homomat_transform_points(transmat4, pseq).tolist()
            rotseq = [np.dot(transmat4[:3, :3], r) for r in rotseq]
            objmat4[:3, 3] = objmat4[:3, 3] - np.asarray([self.bend_r, 0, 0])
            objcm.set_homomat(objmat4)
        if show_frame:
            for i in range(len(pseq)):
                gm.gen_frame(pseq[i], rotseq[i], length=.006, thickness=.0004, alpha=1).attach_to(base)
        if show_dashframe:
            for i in range(len(pseq)):
                gm.gen_dashframe(pseq[i], rotseq[i], length=.006, thickness=.0004, alpha=1).attach_to(base)
            # gm.gen_frame(pseq[0], rotseq[0], length=.01, thickness=.0004, alpha=.5,
            #              rgbmatrix=np.asarray([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]])).attach_to(base)
        if show_pseq:
            for p in bu.linear_inp3d_by_step(pseq):
                gm.gen_sphere(p, radius=.0004, rgba=rgba).attach_to(base)
            gm.gen_sphere(self.pseq[0], radius=.0004, rgba=(1, 0, 0, 1)).attach_to(base)
            gm.gen_sphere(self.pseq[-1], radius=.0004, rgba=(0, 1, 0, 1)).attach_to(base)
        objcm.set_rgba(rgba)
        objcm.attach_to(base)

    def voxelize(self):
        # pcd_narry, _ = self.objcm.sample_surface(radius=.0005)
        # pcd = o3dh.nparray2o3dpcd(np.asarray(pcd_narry))
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=bconfig.THICKNESS / 2)
        return bu.voxelize(self.pseq[1:], self.rotseq[1:])

    def unshow(self):
        self.objcm.detach()

    def show_bendresseq(self, bendresseq, is_success=None):
        motioncounter = [0]
        if is_success is None:
            is_success = [True] * len(bendresseq)
        taskMgr.doMethodLater(.05, self._update_bendresseq, 'update_bendresseq',
                              extraArgs=[motioncounter, bendresseq, is_success], appendTask=True)

    def show_pullseq(self, resseq):
        motioncounter = [0]
        taskMgr.doMethodLater(.05, self._update_pull, 'update_pull',
                              extraArgs=[motioncounter, resseq], appendTask=True)

    def _reverse(self):
        self.pseq = self.pseq[::-1]
        self.rotseq = self.rotseq[::-1]

    def gen_by_bendseq(self, bendseq, h=0.0, cc=True, prune=False, toggledebug=False):
        is_success = [False] * len(bendseq)
        result = [[None]] * len(bendseq)
        fail_reason_list = []
        for i, bend in enumerate(bendseq):
            bend_dir = 0 if bend[0] < 0 else 1
            # bend_angle, lift_angle, rot_angle, bend_pos
            self.move_to_org(bend[3], dir=bend_dir, bend_angle=bend[0], lift_angle=bend[1], rot_angle=bend[2],
                             toggledebug=toggledebug)
            # self.show(rgba=(.7, .7, .7, .7), show_frame=True)
            pseq_init, rotseq_init = copy.deepcopy(self.pseq), copy.deepcopy(self.rotseq)
            if abs(bend[1]) > np.pi / 2:
                pseq_init, rotseq_init = pseq_init[::-1], rotseq_init[::-1]
            if cc:
                print('------------')
                print(bend)
                motor_sa, plate_a = self.cal_start(dir=0 if bend[0] < 0 else 1, toggledebug=toggledebug)
                pseq_init, rotseq_init = copy.deepcopy(self.pseq), copy.deepcopy(self.rotseq)
                if abs(bend[1]) > np.pi / 2:
                    pseq_init, rotseq_init = pseq_init[::-1], rotseq_init[::-1]
                objcm_init = copy.deepcopy(self.objcm)
                if motor_sa is None:
                    print('No collided point found (punch start & plate)!')
                    self.reset(pseq_init, rotseq_init, extend=False)
                    fail_reason_list.append('unbendable')
                    if prune:
                        break
                    else:
                        continue
                self.bend(bend[0], bend[1], bend[3], toggledebug=toggledebug)

                motor_eangle = self.cal_punch_angle(bend_dir, copy.deepcopy(self.objcm), toggledebug)
                if motor_eangle is None:
                    print('No collided point found (punch end & plate)!')
                    self.reset(pseq_init, rotseq_init, extend=False)
                    fail_reason_list.append('unbendable')
                    if prune:
                        break
                    else:
                        continue

                collided_pts = self.bender_cc([objcm_init, self.objcm.copy()])
                # if len(collided_pts) != 0:
                #     self.reset(pseq_init, rotseq_init, extend=False)
                #     fail_reason_list.append('collided')
                #     if prune:
                #         break
                #     else:
                #         continue

                if abs(bend[1]) > np.pi / 2:
                    self._reverse()
                pseq_end, rotseq_end = copy.deepcopy(self.pseq), copy.deepcopy(self.rotseq)
                is_success[i] = True
                print('plate angle:', np.degrees(plate_a))
                print('motor init:', np.degrees(motor_sa))
                print('motor finish:', np.degrees(motor_eangle))
                result[i] = [motor_sa, motor_eangle, plate_a,
                             pseq_init, rotseq_init, pseq_end, rotseq_end]
            else:
                self.bend(bend_angle=bend[0], lift_angle=bend[1], bend_pos=bend[3], toggledebug=toggledebug)
                if abs(bend[1]) > np.pi / 2:
                    self._reverse()
                is_success[i] = True
                result[i] = [0, self.punch_pillar_init, self.punch_pillar_init,
                             pseq_init, rotseq_init,
                             copy.deepcopy(self.pseq), copy.deepcopy(self.rotseq)]
        return is_success, result, fail_reason_list

    def bender_cc(self, objcm_list, toggledebug=False):
        collided_pts_all = []
        for objcm in objcm_list:
            for obj in self.__staticobslist:
                is_collided, collided_pts = obj.is_mcdwith(objcm, toggle_contacts=True)
                if is_collided:
                    collided_pts_all.extend(collided_pts)
        if len(collided_pts_all) != 0:
            print('Collide with bender')
            if toggledebug:
                bu.show_pseq(collided_pts_all, radius=.0004)
                self.show(rgba=(.7, 0, 0, .7))
        return collided_pts_all

    def move_to_org(self, l=bconfig.INIT_L, dir=1, lift_angle=0, rot_angle=0, bend_angle=0, toggledebug=False):
        arc_l = abs(bend_angle * self.bend_r / np.cos(lift_angle))
        if abs(lift_angle) < np.pi / 2:
            inx = self._insert_p(l, toggledebug=False)
        else:
            self._reverse()
            inx = self._insert_p(self._cal_length() - l - arc_l, toggledebug=False)

        init_homomat = rm.homomat_from_posrot([self.bend_r, 0, 0], np.eye(3))
        goal_homomat = rm.homomat_from_posrot(self.pseq[inx], self.rotseq[inx])
        transmat4 = np.dot(init_homomat, np.linalg.inv(goal_homomat))
        self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
        self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])
        if dir == 0:
            rot = rm.rotmat_from_axangle((0, 0, 1), np.pi)
            rot_angle = -rot_angle
            self.pseq = self._rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
            self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])
        # self.show(rgba=(.7, .7, .7, .7), show_dashframe=True)

        if rot_angle != 0:
            self._rot(rot_angle)
        if lift_angle != 0:
            self._lift(lift_angle)
        self.update_cm()
        if toggledebug:
            gm.gen_sphere(self.pseq[inx], radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(self.pseq[inx], self.rotseq[inx], length=.01, thickness=.0004).attach_to(base)
            self.show(rgba=(.7, .7, 0, .7), show_frame=True, show_pseq=True)

    def get_updown_primitive(self):
        rot = rm.rotmat_from_axangle((0, 0, 1), np.pi / 18)
        pseq = self._rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
        rotseq = np.asarray([rot.dot(r) for r in self.rotseq])

        vertices, faces = self.gen_stick(pseq, rotseq, r=self.thickness / 2, section=30)
        tmp_cm = cm.CollisionModel(initor=trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces)),
                                   btwosided=True, name='obj')
        tmp_cm.attach_to(base)

    def _cal_length(self):
        length = 0
        for i in range(len(self.pseq)):
            if i != 0:
                length += np.linalg.norm(np.asarray(self.pseq[i]) - np.asarray(self.pseq[i - 1]))
        return length

    def _lift(self, lift_angle):
        rot = rm.rotmat_from_axangle((1, 0, 0), lift_angle)
        trans_lift = rm.homomat_from_posrot((0, 0, 0), rot)
        self.pseq = rm.homomat_transform_points(trans_lift, self.pseq).tolist()
        self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])

    def _rot(self, rot_angle):
        rot = rm.rotmat_from_axangle((0, 1, 0), rot_angle)
        self.pseq = self._rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
        self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])

    def gen_surface(self, pseq, rotseq, toggledebug=False):
        vertices = []
        faces = []

        for i, p in enumerate(pseq):
            vertices.append(p + rotseq[i][:, 0] * self.thickness / 2 + rotseq[i][:, 2] * self.width / 2)
            vertices.append(p + rotseq[i][:, 0] * self.thickness / 2 - rotseq[i][:, 2] * self.width / 2)
        for i in range(2 * len(pseq) - 2):
            f = [i, i + 1, i + 2]
            if i % 2 == 0:
                f = f[::-1]
            faces.append(f)
        if toggledebug:
            for p in pseq:
                gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, 1], radius=0.0002).attach_to(base)
            tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
            tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
            tmp_cm.set_rgba((.7, .7, 0, .7))
            tmp_cm.attach_to(base)

        return np.asarray(vertices), np.asarray(faces)

    def gen_swap(self, pseq, rotseq, toggledebug=False):
        cross_sec = [[0, self.width / 2], [0, -self.width / 2],
                     [-self.thickness / 2, -self.width / 2], [-self.thickness / 2, self.width / 2]]
        objcm = bu.gen_swap(pseq, rotseq, cross_sec, toggledebug=toggledebug)
        return np.asarray(objcm.objtrm.vertices), np.asarray(objcm.objtrm.faces)

    def gen_stick(self, pseq, rotseq, r, section=5, toggledebug=False):
        vertices = []
        faces = []
        for i, p in enumerate(pseq):
            for a in np.linspace(-np.pi, np.pi, section + 1):
                vertices.append(p + rotseq[i][:, 0] * r * np.sin(a)
                                + rotseq[i][:, 2] * r * np.cos(a))
        for i in range((section + 1) * (len(pseq) - 1)):
            if i % (section + 1) == 0:
                for v in range(i, i + section):
                    faces.extend([[v, v + section + 1, v + section + 2], [v, v + section + 2, v + 1]])
        if toggledebug:
            bu.show_pseq(pseq, rgba=[1, 0, 0, 1], radius=0.0002)
            bu.show_pseq(vertices, rgba=[1, 1, 0, 1], radius=0.0002)
            tmp_trm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
            tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
            tmp_cm.set_rgba((.7, .7, 0, .7))
            tmp_cm.attach_to(base)

        return np.asarray(vertices), np.asarray(faces)

    def _update_pull(self, motioncounter, resseq, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            if motioncounter[0] < len(resseq):
                print('-------------')
                for pseq, rotseq in resseq[motioncounter[0]]:
                    self.reset(pseq, rotseq, extend=False)
                    # gm.gen_frame(pseq[0], rotseq[0], thickness=.001, length=.01).attach_to(base)
                    objcm_init = copy.deepcopy(self.objcm)
                    objcm_init.set_rgba((.7, .7, 0, .7))
                    objcm_init.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def _update_bendresseq(self, motioncounter, bendresseq, is_success, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                flag = is_success[motioncounter[0]]
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))
                # gm.gen_frame(pseq_init[0], rotseq_init[0], length=.02, thickness=.0005).attach_to(base)

                self.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self.objcm)
                if flag:
                    objcm_init.set_rgba((.7, 0, 0, .7))
                else:
                    objcm_init.set_rgba((.7, 0, .7, .7))
                objcm_init.attach_to(base)

                self.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self.objcm)
                if flag:
                    objcm_end.set_rgba((0, .7, 0, .7))
                else:
                    objcm_end.set_rgba((0, .7, .7, .7))
                objcm_end.attach_to(base)

                if init_a is not None:
                    tmp_p = np.asarray([self.c2c_dist * math.cos(init_a), self.c2c_dist * math.sin(init_a), 0])
                    self.pillar_punch.set_homomat(rm.homomat_from_posrot(tmp_p, np.eye(3)))
                    self.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                    self.pillar_punch.attach_to(base)

                if end_a is not None:
                    tmp_p = np.asarray([self.c2c_dist * math.cos(end_a), self.c2c_dist * math.sin(end_a), 0])
                    self.pillar_punch_end.set_homomat(rm.homomat_from_posrot(tmp_p, np.eye(3)))
                    self.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                    self.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def _gen_cc_bound(self, p, rot, l=.05):
        vertices, faces = self.gen_stick([p, p + rot[:3, 1] * l], [rot, rot], self.slot_w / 2, section=8)
        cctrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        return cm.CollisionModel(initor=cctrm, btwosided=True, name='obj')

    def get_pull_primitive(self, sl, el, toggledebug=False):
        s_inx = self._insert_p(sl)
        e_inx = self._insert_p(el)
        gm.gen_sphere(self.pseq[s_inx], radius=.001, rgba=(1, 0, 0, 1)).attach_to(base)
        gm.gen_sphere(self.pseq[e_inx], radius=.001, rgba=(0, 1, 0, 1)).attach_to(base)
        # gm.gen_frame(self.pseq[s_inx], self.rotseq[s_inx], thickness=.001, length=.01).attach_to(base)
        # gm.gen_frame(self.pseq[e_inx], self.rotseq[e_inx], thickness=.001, length=.01).attach_to(base)

        if s_inx < e_inx:
            pseq = self.pseq[s_inx:e_inx + 1]
            rotseq = self.rotseq[s_inx:e_inx + 1]
            l = .05
        else:
            pseq = self.pseq[e_inx:s_inx + 1][::-1]
            rotseq = self.rotseq[e_inx:s_inx + 1][::-1]
            e_inx, s_inx = s_inx, e_inx
            l = -.05

        cccm = self._gen_cc_bound(pseq[0], rotseq[0], l=l)
        i = 1
        j = 1
        key_pseq, key_rotseq = [], []
        key_idx = []
        while i < len(pseq) - 1:
            print(range(i, len(pseq) - 1))
            for j in range(i, len(pseq) - 1):
                vertices, faces = \
                    self.gen_stick(np.asarray([pseq[j - 1], pseq[j + 1]]), np.asarray([rotseq[j - 1], rotseq[j + 1]]),
                                   self.thickness / 2, section=4)
                strm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
                scm = cm.CollisionModel(initor=strm, btwosided=True, name='scm')
                scm.attach_to(base)
                is_collided, collided_pts = scm.is_mcdwith(cccm, toggle_contacts=True)
                if is_collided:
                    p = np.average(collided_pts, axis=0)
                    rot = rotseq[j]
                    key_pseq.append(p)
                    key_rotseq.append(rot)
                    key_idx.append(j)
                    # gm.gen_frame(p, rot, thickness=.001, length=.01).attach_to(base)
                    cccm = self._gen_cc_bound(p, rot, l=l)
                    cccm_show = self._gen_cc_bound(pseq[i], rotseq[i], l=np.linalg.norm(pseq[i] - p))
                    cccm_show.set_rgba((1, 1, 1, .5))
                    cccm_show.attach_to(base)
                    break

                j = len(pseq) - 1
            i = j + 1
        if toggledebug:
            for i in range(len(key_pseq)):
                gm.gen_frame(key_pseq[i], key_rotseq[i], thickness=.001, length=.01).attach_to(base)
        if l > 0:
            key_pseq = [self.pseq[s_inx]] + key_pseq + [self.pseq[e_inx]]
            key_rotseq = [self.rotseq[s_inx]] + key_rotseq + [self.rotseq[e_inx]]
        else:
            key_pseq = [self.pseq[e_inx]] + key_pseq + [self.pseq[s_inx]]
            key_rotseq = [self.rotseq[e_inx]] + key_rotseq + [self.rotseq[s_inx]]
        return key_pseq, key_rotseq

    def pull_sample(self, key_pseq, key_rotseq):
        self.add_staticobs(self.pillar_punch)
        goal_p = np.asarray([(self.slot_w / 2 + self.r_center) * np.cos(self.punch_pillar_init / 2),
                             (self.slot_w / 2 + self.r_center) * np.sin(self.punch_pillar_init / 2), 0])
        goal_rot = rm.rotmat_from_axangle((0, 0, 1), self.punch_pillar_init / 2)
        goal_homomat = rm.homomat_from_posrot(goal_p, goal_rot)
        org_pseq, org_rotseq = self.pseq.copy(), self.rotseq.copy()
        resseq = []
        for i in range(len(key_pseq)):
            resseq_tmp = []
            init_homomat = rm.homomat_from_posrot(key_pseq[i], key_rotseq[i])
            transmat4 = np.dot(goal_homomat, np.linalg.inv(init_homomat))
            self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
            self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])

            for rot_a in np.linspace(0, np.pi * 2, 10):
                for lift_a in np.linspace(-np.pi / 2, np.pi / 2, 10):
                    rot = rm.rotmat_from_axangle(goal_rot[:3, 1], rot_a) \
                        .dot(rm.rotmat_from_axangle(goal_rot[:3, 0], lift_a))
                    self.pseq = self._rot_new_orgin(self.pseq, -goal_p, rot)
                    self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])
                    self.update_cm()
                    collided_pts = self.bender_cc([self.objcm.copy()])
                    if len(collided_pts) == 0:
                        resseq_tmp.append([self.pseq.copy(), self.rotseq.copy()])
                    # resseq_tmp.append([self.pseq.copy(), self.rotseq.copy()])

            self.reset(org_pseq, org_rotseq, extend=False)
            resseq.append(resseq_tmp)
        self.reset_staticobs()
        # self.show_pullseq(resseq)
        # base.run()

        return resseq

    def pull(self, key_pseq, key_rotseq, rot_a):
        self.add_staticobs(self.pillar_punch)
        goal_p = np.asarray([(self.slot_w / 2 + self.r_center) * np.cos(self.punch_pillar_init / 2),
                             (self.slot_w / 2 + self.r_center) * np.sin(self.punch_pillar_init / 2), 0])
        goal_rot = rm.rotmat_from_axangle((0, 0, 1), self.punch_pillar_init / 2)
        goal_rot = rm.rotmat_from_axangle(goal_rot[:3, 1], rot_a).dot(goal_rot)
        goal_homomat = rm.homomat_from_posrot(goal_p, goal_rot)
        org_pseq, org_rotseq = self.pseq.copy(), self.rotseq.copy()
        resseq = []
        for i in range(len(key_pseq)):
            init_homomat = rm.homomat_from_posrot(key_pseq[i], key_rotseq[i])
            transmat4 = np.dot(goal_homomat, np.linalg.inv(init_homomat))
            self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
            self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])

            self.pseq = self._rot_new_orgin(self.pseq, -goal_p, goal_rot)
            self.rotseq = np.asarray([goal_rot.dot(r) for r in self.rotseq])
            self.update_cm()
            collided_pts = self.bender_cc([self.objcm.copy()])
            # if len(collided_pts) == 0:
            #     resseq.append([self.pseq.copy(), self.rotseq.copy()])
            # else:
            #     resseq.append(None)
            resseq.append([self.pseq.copy(), self.rotseq.copy()])

            self.reset(org_pseq, org_rotseq, extend=False)
        self.reset_staticobs()
        # self.show_pullseq([[r] for r in resseq])
        # base.run()

        return resseq

    def gen_random_bendset(self, n=5, max_bend=np.pi):
        bendset = []
        for i in range(n):
            if i > 0:
                bendset.append([random.uniform(.2, .98) * max_bend,
                                0,
                                random.uniform(0, 1) * np.pi * random.choice([-1, 1]),
                                bendset[-1][-1] + self.bend_r * abs(bendset[-1][0] / np.cos(bendset[-1][1]))
                                + random.uniform(self.bend_r * 2, self.bend_r * max_bend)])
            else:
                bendset.append([random.uniform(.2, .98) * max_bend,
                                0,
                                random.uniform(0, 1) * np.pi * random.choice([-1, 1]),
                                self.bend_r * np.pi + .1])
        return bendset


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.075, .1, .05], lookat_pos=[0, 0, 0])
    bs = BendSim(show=False, cm_type='plate', granularity=np.pi / 30)
    bs.set_stick_sec(180)
    cm.gen_stick(spos=np.asarray([0, 0, -.015]),
                 epos=np.asarray([0, 0, .015]),
                 thickness=.015, sections=180,
                 rgba=[1, 1, 1, 1]).attach_to(base)
    # bs.pillar_center = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H / 2.5]),
    #                                 epos=np.asarray([0, 0, bconfig.PILLAR_H / 2.5]),
    #                                 thickness=bs.r_center * 2, sections=90,
    #                                 rgba=[.9, .9, .9, 1]).attach_to(base)
    bendset = [
        # [np.radians(225), np.radians(0), np.radians(0), .04],
        # [np.radians(-90), np.radians(0), np.radians(0), .08],
        # [np.radians(-40), np.radians(0), np.radians(0), .08],
        # [np.radians(90), np.radians(0), np.radians(0), .1],
        # [np.radians(45), np.radians(0), np.radians(0), .04],
        # [np.radians(45), np.radians(0), np.radians(0), .04],
        # [np.radians(45), np.radians(30), np.radians(0), .04],
        [np.radians(45), np.radians(0), np.radians(45), .015],
        # [np.radians(90), np.radians(0), np.radians(0), .04],
        # [np.radians(180), np.radians(0), np.radians(0), .04],
        # [np.radians(-90), np.radians(0), np.radians(0), .08],
        # [np.radians(90), np.radians(0), np.radians(0), .1],

    ]
    # bendseq = pickle.load(open('./penta_bendseq.pkl', 'rb'))
    # bendset = bs.gen_random_bendset(5)
    # print(bendset)
    bs.reset([(0, 0, 0), (0, max(np.asarray(bendset)[:, 3]) - .01, 0)], [np.eye(3), np.eye(3)])

    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    # bs.show(rgba=(.7, .7, 0, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))
    bs.show(rgba=(.7, .7, 0, .7), show_frame=True)
    # bu.visualize_voxel([bs.voxelize()], colors=['r'])

    # bs.get_updown_primitive()

    # bs.move_to_org(.04)
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    # key_pseq, key_rotseq = bs.get_pull_primitive(.12, .04, toggledebug=True)
    # resseq = bs.pull(key_pseq, key_rotseq, np.pi)
    # bs.move_to_org(.04)
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)

    base.run()
