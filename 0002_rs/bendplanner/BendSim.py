import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import utils.math_utils as mu
import basis.trimesh as trm
import modeling.collision_model as cm
import bend_utils as bu
import bender_config as bconfig


def draw_plane(p, n):
    pt_direction = rm.orthogonal_vector(n, toggle_unit=True)
    tmp_direction = np.cross(n, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, n))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array(p)
    gm.gen_box(np.array([.2, .2, .0005]), homomat=homomat, rgba=[1, 1, 1, .3]).attach_to(base)


class BendSim(object):
    def __init__(self, pseq=None, rotseq=None, show=False, granularity=np.pi / 90):
        # bending device prop
        self.r_side = bconfig.R_SIDE
        self.r_center = bconfig.R_CENTER
        self.r_base = bconfig.R_BASE
        self.c2c_dist = bconfig.C2C_DIST

        # bending meterial prop
        self.thickness = bconfig.THICKNESS
        self.width = bconfig.WIDTH

        # bending device prop
        self.bend_r = self.r_center + self.thickness / 2
        self.punch_pillar_init = self.cal_start_margin() / 2

        # bending result
        self.objcm = None
        if pseq is None:
            self.pseq = [[self.bend_r, 0, 0]]
            self.rotseq = [np.eye(3)]
        else:
            self.pseq = pseq
            self.pseq.append((0, self.pseq[-1][1] + 2 * bconfig.INIT_L, 0))
            self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
            self.rotseq = rotseq
            self.rotseq.append(np.eye(3))

        # gen pillars
        sections = 360
        self.pillar_center = cm.gen_stick(spos=np.asarray([0, 0, -.02]),
                                          epos=np.asarray([0, 0, .02]),
                                          thickness=self.r_center * 2 - .0001, sections=sections, rgba=[.7, .7, .7, .7])
        self.pillar_dieside = cm.gen_stick(spos=np.asarray([self.c2c_dist, 0, -.02]),
                                           epos=np.asarray([self.c2c_dist, 0, .02]),
                                           thickness=self.r_side * 2, sections=sections, rgba=[.7, .7, .7, .7])
        self.pillar_punch = cm.gen_stick(spos=np.asarray([0, 0, -.02]),
                                         epos=np.asarray([0, 0, .02]),
                                         thickness=self.r_side * 2, sections=sections, rgba=[.7, .7, .7, .7])
        self.pillar_punch.set_homomat(
            rm.homomat_from_posrot([self.c2c_dist * np.cos(self.punch_pillar_init),
                                    self.c2c_dist * np.sin(self.punch_pillar_init), 0], np.eye(3)))
        if show:
            gm.gen_frame(thickness=.001, length=.05, alpha=.1).attach_to(base)
            for a in np.arange(0, 2 * math.pi, math.pi / 360):
                gm.gen_sphere(pos=[self.c2c_dist * math.cos(a), self.c2c_dist * math.sin(a), 0], radius=.0001,
                              rgba=(.7, .7, .7, .2)).attach_to(base)
            self.pillar_center.attach_to(base)
            self.pillar_dieside.attach_to(base)
            self.pillar_punch.attach_to(base)
        # others
        self.granularity = granularity

    def reset(self, pseq, rotseq):
        self.pseq = copy.deepcopy(pseq)
        self.pseq.append((0, self.pseq[-1][1] + 2 * np.pi * self.bend_r, 0))
        self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
        self.rotseq = copy.deepcopy(rotseq)
        self.rotseq.append(np.eye(3))
        self.objcm = None

    def cal_tail(self):
        A = np.mat([[self.r_side + self.thickness, -self.r_center], [1, 1]])
        b = np.mat([0, self.c2c_dist]).T
        l_center, l_side = np.asarray(np.linalg.solve(A, b))

        return l_center[0], l_side[0], \
               np.sqrt(l_center[0] ** 2 - self.r_center ** 2) + \
               np.sqrt(l_side[0] ** 2 - (self.r_side + self.thickness) ** 2)

    def cal_safe_margin(self):
        return 2 * np.arcsin((self.r_side + self.r_base) / (2 * self.c2c_dist))

    def cal_start_margin(self):
        l_center, _, _ = self.cal_tail()
        return 2 * np.arccos(self.r_center / l_center)

    def cal_startp(self, dir=1, toggledebug=False):
        plate_angle = None
        objcm_bend = None
        range_plate = np.linspace(0, np.pi, 360) if dir == 1 else np.linspace(0, -np.pi, 360)
        step_plate = range_plate[1] - range_plate[0]
        for i in range_plate:
            tmp_p = np.asarray([self.bend_r * math.cos(i), self.bend_r * math.sin(i), 0])
            transmat4 = rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((0, 0, 1), step_plate))
            self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
            self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])
            self.update_cm()
            # self.show()
            objcm_bend = copy.deepcopy(self.objcm)
            is_collided, collided_pts = self.pillar_dieside.is_mcdwith(objcm_bend, toggle_contacts=True)
            if is_collided:
                plate_angle = i
                if toggledebug:
                    bu.show_pseq(collided_pts, radius=.0004)
                    objcm_bend.set_rgba(rgba=[0, 0, 1, .7])
                    objcm_bend.attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray(tmp_p),
                                     rgba=[.7, .7, .7, .7],
                                     thickness=.0005).attach_to(base)
                break
        if plate_angle is None:
            print('No collided point found (die pillar & plate)!')
            return None, None, None
        print('plate angle:', np.degrees(plate_angle))

        range_punch = np.linspace(self.punch_pillar_init, np.pi, 360) if dir == 1 \
            else np.linspace(-self.punch_pillar_init, -np.pi, 360)
        for i in range_punch:
            tmp_p = np.asarray([self.c2c_dist * math.cos(i), self.c2c_dist * math.sin(i), 0])
            self.pillar_punch.set_homomat(rm.homomat_from_posrot(tmp_p, np.eye(3)))
            self.pillar_punch.set_rgba(rgba=[0, .7, .7, .7])
            is_collided, collided_pts = self.pillar_punch.is_mcdwith(objcm_bend, toggle_contacts=True)
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

                return self.pseq[0], self.rotseq[0], np.pi - abs(i) if i > 0 else abs(i) - np.pi
        print('No collided point found (punch pillar & plate)!')
        return None, None, None

    def __insert(self, pos, pseq, rotseq):
        tmp_l = 0
        insert_inx, insert_pos, insert_rot = 0, pseq[0], rotseq[0]
        if pos == 0:
            return insert_inx, insert_pos, insert_rot
        for i in range(len(pseq) - 1):
            p1 = np.asarray(pseq[i])
            p2 = np.asarray(pseq[i + 1])
            r1 = rotseq[i]
            r2 = rotseq[i + 1]
            tmp_l += np.linalg.norm(p2[:2] - p1[:2])
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
                return insert_inx, insert_pos, insert_rot
            else:
                insert_pos = pseq[i + 1]
                insert_rot = rotseq[i + 1]
                insert_inx = i + 1
                return insert_inx, insert_pos, insert_rot
        return -1, pseq[-1], rotseq[-1]

    def __get_bended_pseq(self, start_inx, end_inx, bend_angle, toggledebug=False):
        tmp_pseq = []
        tmp_rotseq = []
        if bend_angle > 0:
            rng = (0, bend_angle + self.granularity)
            srange = np.arange(rng[0], rng[1], self.granularity)
        else:
            rng = (bend_angle, self.granularity)
            srange = np.arange(rng[0], rng[1], self.granularity)[::-1]

        pseq_org = self.pseq[start_inx:end_inx + 1]
        rotseq_org = self.rotseq[start_inx:end_inx + 1]
        step_l = abs(bend_angle * self.bend_r) / len(srange)
        init_a = np.arctan(pseq_org[0][1] / pseq_org[0][0])
        stop = 0
        for i, a in enumerate(srange):
            # p = (r * math.cos(a), r * math.sin(a), a * r * math.tan(lift_angle))
            insert_inx, insert_pos, insert_rot = self.__insert(step_l * i, pseq_org, rotseq_org)
            p = (self.bend_r * math.cos(init_a + a), self.bend_r * math.sin(init_a + a), insert_pos[2])
            # print(np.degrees(init_a + a), p)
            if insert_inx == -1:
                stop += 1
            if stop > 1:
                break
            tmp_pseq.append(p)
            tmp_rotseq.append(rm.rotmat_from_axangle((0, 0, 1), a)
                              # .dot(rm.rotmat_from_axangle((0, 1, 0), rot_angle))
                              # .dot(rm.rotmat_from_axangle((1, 0, 0), lift_angle))
                              .dot(insert_rot))
            if toggledebug:
                gm.gen_sphere(np.asarray(tmp_pseq[-1]), rgba=[1, 0, 0, .5], radius=0.0002).attach_to(base)
                gm.gen_frame(np.asarray(tmp_pseq[-1]), tmp_rotseq[-1], length=.005,
                             thickness=.0004, alpha=.5).attach_to(base)

        return tmp_pseq, tmp_rotseq

    def bend(self, bend_angle, lift_angle=0, rot_angle=0, bend_pos=None, toggledebug=False):
        # rot_end = np.dot(rm.rotmat_from_axangle((1, 0, 0), lift_angle), rm.rotmat_from_axangle((0, 0, 1), bend_angle))
        rot_end = rm.rotmat_from_axangle((0, 0, 1), bend_angle)
        if abs(lift_angle) >= 90:
            print("lift angle should be in -90~90 degree!")
            return None, None
        if bend_pos is not None:
            arc_l = abs(bend_angle * self.bend_r / np.cos(lift_angle))
            start_inx = self.__insert_p(bend_pos, toggledebug=False)
            end_inx = self.__insert_p(bend_pos + arc_l, toggledebug=False)
            tmp_pseq, tmp_rotseq = self.__get_bended_pseq(start_inx, end_inx, bend_angle=bend_angle)
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
            org_pseq = copy.deepcopy(self.pseq)
            org_rotseq = copy.deepcopy(self.rotseq)
            self.pseq = self.pseq[:start_inx] + pseq_mid + pseq_end
            self.rotseq = self.rotseq[:start_inx] + rotseq_mid + rotseq_end
            self.update_cm()
            objcm = copy.deepcopy(self.objcm)
            is_collided, collided_pts = self.pillar_center.is_mcdwith(objcm, toggle_contacts=True)
            if is_collided:
                bu.show_pseq(collided_pts, radius=.0004)
                print('Bend result collid with center pillar!')
                self.show(rgba=(.7, 0, 0, .7))
                # base.run()
                self.pseq, self.rotseq = org_pseq, org_rotseq
                return False
            return True

        else:
            tmp_pseq, tmp_rotseq = \
                self.__get_bended_pseq(start_inx=-1, end_inx=-1, bend_angle=bend_angle)
            if bend_angle > 0:
                pos = np.asarray((0, 0, 0))
                self.pseq = tmp_pseq + rm.homomat_transform_points(rm.homomat_from_posrot(pos, rot_end),
                                                                   self.pseq).tolist()
            else:
                pos = np.asarray((-2 * self.bend_r, 0, 0))
                self.pseq = tmp_pseq + self.__rot_new_orgin(self.pseq, pos, rot_end)
            self.rotseq = tmp_rotseq + [np.dot(rot_end, org_rot) for org_rot in self.rotseq]

    def __trans_pos(self, pts, pos):
        return rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray(pos), np.eye(3)), pts)

    def __rot_new_orgin(self, pts, new_orgin, rot):
        trans_pts = self.__trans_pos(pts, new_orgin)
        trans_pts = rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray([0, 0, 0]), rot), trans_pts)
        return self.__trans_pos(trans_pts, -new_orgin).tolist()

    def __insert_p(self, pos, toggledebug=False):
        tmp_l = 0
        insert_inx, insert_pos, insert_rot = 0, self.pseq[0], self.rotseq[0]
        for i in range(len(self.pseq) - 1):
            p1 = np.asarray(self.pseq[i])
            p2 = np.asarray(self.pseq[i + 1])
            r1 = self.rotseq[i]
            r2 = self.rotseq[i + 1]
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
                self.pseq = self.pseq[:i + 1] + [insert_pos] + self.pseq[i + 1:]
                self.rotseq = list(self.rotseq[:i + 1]) + [insert_rot] + list(self.rotseq[i + 1:])
                break
            else:
                insert_pos = self.pseq[i + 1]
                insert_rot = self.rotseq[i + 1]
                insert_inx = i + 1
                break

        if toggledebug:
            gm.gen_sphere(insert_pos, radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(insert_pos, insert_rot, length=.01, thickness=.0004).attach_to(base)

        return insert_inx
        # print('error', pos)

    def feed(self, pos_diff):
        pos = np.asarray(pos_diff)
        self.pseq = rm.homomat_transform_points(rm.homomat_from_posrot(pos, np.eye(3)), self.pseq).tolist()
        # self.rotseq = [self.rotseq[0]] + self.rotseq
        # self.pseq = [[self.bend_r, 0, 0]] + self.pseq
        # print(len(self.pseq),len(self.rotseq))

    def update_cm(self, type='stick'):
        if type == 'stick':
            vertices, faces = self.__gen_stick()
        else:
            vertices, faces = self.__gen_surface()
        objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        self.objcm = cm.CollisionModel(initor=objtrm, btwosided=True)

    def show(self, rgba=(1, 1, 1, 1), objmat4=None, show_frame=False, show_pseq=False):
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
                gm.gen_frame(pseq[i], rotseq[i], length=.005, thickness=.0005, alpha=.5).attach_to(base)
        if show_pseq:
            for p in bu.linear_inp3d_by_step(pseq):
                gm.gen_sphere(p, radius=.0004, rgba=rgba).attach_to(base)
            # gm.gen_sphere(self.pseq[0], radius=.0004, rgba=(1, 0, 0, 1)).attach_to(base)
            # gm.gen_sphere(self.pseq[-1], radius=.0004, rgba=(0, 1, 0, 1)).attach_to(base)

        objcm.set_rgba(rgba)
        objcm.attach_to(base)

    def unshow(self):
        self.objcm.detach()

    def show_ani(self, motion_seq):
        motioncounter = [0]
        taskMgr.doMethodLater(2, self.__update_ani, "update",
                              extraArgs=[motioncounter, motion_seq], appendTask=True)

    def show_ani_bendseq(self, bendseq):
        motioncounter = [0]
        init_pseq = self.pseq
        init_rotseq = self.rotseq
        taskMgr.doMethodLater(2, self.__update_bendseq, "update",
                              extraArgs=[motioncounter, bendseq, init_pseq, init_rotseq], appendTask=True)

    def gen_by_motionseq(self, motionseq):
        # motion_seq = [
        #     ['f', (0, .01, 0)],
        #     ['b', np.radians(-60), np.radians(10)],
        # ]
        for motion in motionseq:
            if motion[0] == 'b':
                self.bend(motion[1], motion[2], motion[3])
            else:
                self.feed(motion[1])

    def gen_by_bendseq(self, bendseq, toggledebug=False):
        result_flag = [False] * len(bendseq)
        for i, bend in enumerate(bendseq):
            # bend_angle, lift_angle, rot_angle, bend_pos
            print(bend)
            flag = self.move_to_org(bend[3], dir=0 if bend[0] < 0 else 1, lift_angle=bend[1], rot_angle=bend[2],
                                    toggledebug=toggledebug)
            if flag:
                pos, rot, angle = self.cal_startp(dir=0 if bend[0] < 0 else 1, toggledebug=toggledebug)
                if pos is not None:
                    if toggledebug:
                        # hndmat4 = rm.homomat_from_posrot(pos, rot)
                        gm.gen_frame(pos, rot, length=.01, thickness=.001).attach_to(base)
                    print("motor init:", np.degrees(angle))
                    print("motor finish:", np.degrees(angle) + np.degrees(bend[0]))
                    bend_flag = self.bend(bend[0], bend[1], bend[2], bend[3], toggledebug=toggledebug)
                    result_flag[i] = bend_flag
                print('------------')
            else:
                continue
        return result_flag

    def cal_length(self):
        length = 0
        for i in range(len(self.pseq)):
            if i != 0:
                length += np.linalg.norm(np.asarray(self.pseq[i]) - np.asarray(self.pseq[i - 1]))
        return length

    def move_to_org(self, l, dir=1, lift_angle=0, rot_angle=0, toggledebug=False):
        inx = self.__insert_p(l, toggledebug=False)
        init_homomat = rm.homomat_from_posrot([self.bend_r, 0, 0], np.eye(3))
        goal_homomat = rm.homomat_from_posrot(self.pseq[inx], self.rotseq[inx])
        transmat4 = np.dot(init_homomat, np.linalg.inv(goal_homomat))
        self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
        self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])
        if dir == 0:
            rot = rm.rotmat_from_axangle((0, 0, 1), np.pi)
            rot_angle = -rot_angle
            self.pseq = self.__rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
            self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])
        if rot_angle != 0:
            self.__rot(rot_angle)
        if lift_angle != 0:
            self.__lift(lift_angle)
        self.update_cm()
        objcm = copy.deepcopy(self.objcm)
        is_collided, collided_pts = self.pillar_center.is_mcdwith(objcm, toggle_contacts=True)
        if is_collided:
            print('Bend init pos collid with center pillar!')
            self.show(rgba=(.7, 0, 0, .7))
            base.run()
            return False
        if toggledebug:
            gm.gen_sphere(self.pseq[inx], radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(self.pseq[inx], self.rotseq[inx], length=.01, thickness=.0004).attach_to(base)
            self.show(rgba=(.7, .7, 0, .7), show_frame=True)
        return True

    def __lift(self, lift_angle):
        rot = rm.rotmat_from_axangle((1, 0, 0), lift_angle)
        trans_lift = rm.homomat_from_posrot((0, 0, 0), rot)
        self.pseq = rm.homomat_transform_points(trans_lift, self.pseq).tolist()
        self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])

    def __rot(self, rot_angle):
        rot = rm.rotmat_from_axangle((0, 1, 0), rot_angle)
        self.pseq = self.__rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
        self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])

    def __gen_surface(self, toggledebug=False):
        tmp_vertices = []
        tmp_faces = []
        pseq = self.pseq[::-1]
        rotseq = self.rotseq[::-1]
        for i, p in enumerate(pseq):
            tmp_vertices.append(p + rotseq[i][:, 0] * self.thickness / 2 + rotseq[i][:, 2] * self.width / 2)
            tmp_vertices.append(p + rotseq[i][:, 0] * self.thickness / 2 - rotseq[i][:, 2] * self.width / 2)
        for i in range(2 * len(pseq) - 2):
            f = [i, i + 1, i + 2]
            if i % 2 == 0:
                f = f[::-1]
            tmp_faces.append(f)
        if toggledebug:
            for p in pseq:
                gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, 1], radius=0.0002).attach_to(base)
            tmp_trm = trm.Trimesh(vertices=np.asarray(tmp_vertices), faces=np.asarray(tmp_faces))
            tmp_cm = cm.CollisionModel(initor=tmp_trm, btwosided=True)
            tmp_cm.set_rgba((.7, .7, 0, .7))
            tmp_cm.attach_to(base)

        return np.asarray(tmp_vertices), np.asarray(tmp_faces)

    def __gen_stick(self):
        vertices = []
        faces = []
        for i in range(len(self.pseq) - 1):
            tmp_cm = cm.gen_stick(spos=np.asarray(self.pseq[i]),
                                  epos=np.asarray(self.pseq[i + 1]),
                                  thickness=self.thickness)
            tmp_faces = tmp_cm.objtrm.faces
            faces.extend(tmp_faces + np.ones(tmp_faces.shape) * len(vertices))
            vertices.extend(tmp_cm.objtrm.vertices)
        return np.array(vertices), np.array(faces)

    def __update_ani(self, motioncounter, motion_seq, task):
        if motioncounter[0] < len(motion_seq):
            if self.objcm is not None:
                self.objcm.detach()
            if motion_seq[motioncounter[0]][0] == 'b':
                bs.bend(motion_seq[motioncounter[0]][1], motion_seq[motioncounter[0]][2])
            else:
                bs.feed(motion_seq[motioncounter[0]][1])
            self.update_cm()
            self.objcm.attach_to(base)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
            self.reset([], [])
        return task.again

    # def clearbase(self, sceneflag=3):
    #     for i in base.render.children:
    #         if sceneflag > 0:
    #             sceneflag -= 1
    #         else:
    #             i.removeNode()

    def __update_bendseq(self, motioncounter, bendseq, init_pseq, init_rotseq, task):
        if motioncounter[0] < len(bendseq):
            self.pillar_punch.detach()
            flag = self.move_to_org(bendseq[motioncounter[0]][2], dir=1 if bendseq[motioncounter[0]][0] > 0 else 0,
                                    lift_angle=bendseq[motioncounter[0]][1])
            bs.cal_startp(toggledebug=True)
            objcm_init = copy.deepcopy(self.objcm)
            objcm_init.attach_to(base)
            self.pillar_punch.attach_to(base)
            bs.bend(bendseq[motioncounter[0]][0], bendseq[motioncounter[0]][1], bendseq[motioncounter[0]][2])
            self.update_cm()
            objcm_res = copy.deepcopy(self.objcm)
            objcm_res.attach_to(base)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
            self.reset(init_pseq, init_rotseq)
        return task.again


if __name__ == '__main__':
    import pickle

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    bendseq = [
        [np.radians(60), np.radians(0), np.radians(90), .02],
        [np.radians(60), np.radians(0), np.radians(0), .02],
        # [np.radians(-60), np.radians(0), np.radians(0), .02],
        # [np.radians(60), np.radians(0), np.radians(-90), .06],
        # [np.radians(40), np.radians(0), np.radians(0), .06],
        # [np.radians(-15), np.radians(0), np.radians(0), .08],
        # [np.radians(20), np.radians(0), np.radians(0), .1]
    ]
    # bendseq = pickle.load(open('./tmp_bendseq.pkl', 'rb'))[:5]

    bs = BendSim(show=True)
    bs.reset([(0, 0, 0), (0, bendseq[-1][3], 0)], [np.eye(3), np.eye(3)])
    bs.show(rgba=(.7, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))
    bs.show(rgba=(.7, .7, .7, .7), show_frame=True)

    result_flag = bs.gen_by_bendseq(bendseq, toggledebug=True)
    print('Result Flag:', result_flag)
    # bs.show_ani_bendseq(bendseq)
    bs.show(rgba=(0, 0, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1)), show_pseq=True, show_frame=True)
    bs.show(rgba=(0, 0, .7, .7), show_pseq=True, show_frame=True)

    # threshold = bs.cal_startp(toggledebug=True)
    # print(threshold)
    # threshold = bs.cal_startp(pos_l=.02, toggledebug=True)

    # bs.bend(np.radians(30), np.radians(0), insert_l=.015, toggledebug=True)
    # bs.show(rgba=(.7, .7, 0, .7))
    # print(bs.pseq)

    # init_pseq = [(0, 0, 0), (0, .01, 0), (0, .05, 0)]
    # init_rotseq = [np.eye(3), np.eye(3), np.eye(3)]
    # bs.reset(init_pseq, init_rotseq)
    # bs.show(rgba=(.7, 0, 0, .7))
    # print(bs.pseq)

    # bs.show_ani(motion_seq)
    base.run()
