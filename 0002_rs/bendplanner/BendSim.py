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


def draw_plane(p, n):
    pt_direction = rm.orthogonal_vector(n, toggle_unit=True)
    tmp_direction = np.cross(n, pt_direction)
    plane_rotmat = np.column_stack((pt_direction, tmp_direction, n))
    homomat = np.eye(4)
    homomat[:3, :3] = plane_rotmat
    homomat[:3, 3] = np.array(p)
    gm.gen_box(np.array([.2, .2, .0005]), homomat=homomat, rgba=[1, 1, 1, .3]).attach_to(base)


class BendSim(object):
    def __init__(self, thickness, width, pseq=None, rotseq=None, show=False):
        # bending device prop
        self.r_side = .013 / 2
        self.r_center = .02 / 2
        self.r_base = .015 / 2
        self.c2c_dist = .0235

        # bending meterial prop
        self.thickness = thickness
        self.width = width

        # bending device prop
        self.bend_r = self.r_center + self.thickness
        self.punch_pillar_init = self.cal_start_margin() / 2

        # bending result
        self.objcm = None
        if pseq is None:
            self.pseq = [[self.bend_r, 0, 0]]
            self.rotseq = [np.eye(3)]
        else:
            self.pseq = pseq
            self.pseq.append((0, self.pseq[-1][1] + 2 * np.pi * self.bend_r, 0))
            self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
            self.rotseq = rotseq
            self.rotseq.append(np.eye(3))

        # gen pillars
        sections = 360
        self.pillar_center = gm.gen_stick(spos=np.asarray([0, 0, -.02]),
                                          epos=np.asarray([0, 0, .02]),
                                          thickness=self.r_center * 2, sections=sections, rgba=[.7, .7, .7, .7])
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
            gm.gen_frame(thickness=.001, length=.05).attach_to(base)
            for a in np.arange(0, 2 * math.pi, math.pi / 360):
                gm.gen_sphere(pos=[self.c2c_dist * math.cos(a), self.c2c_dist * math.sin(a), 0], radius=.0001,
                              rgba=(.7, .7, .7, .2)).attach_to(base)
            self.pillar_center.attach_to(base)
            self.pillar_dieside.attach_to(base)
            self.pillar_punch.attach_to(base)

    def reset(self, pseq, rotseq):
        self.pseq = pseq
        self.pseq.append((0, self.pseq[-1][1] + 2 * np.pi * self.bend_r, 0))
        self.pseq = [np.asarray(p) + np.asarray([self.bend_r, 0, 0]) for p in self.pseq]
        self.rotseq = rotseq
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

    def cal_startp(self, pos_l, dir=1, lift_angle=0, toggledebug=False):
        self.move_to_org(pos_l, dir, lift_angle, toggledebug=toggledebug)
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
                    for p in collided_pts:
                        gm.gen_sphere(p, radius=.0004).attach_to(base)
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

        range_punch = np.linspace(self.punch_pillar_init, np.pi, 360) if dir == 1 \
            else np.linspace(-self.punch_pillar_init, -np.pi, 360)
        for i in range_punch:
            tmp_p = np.asarray([self.c2c_dist * math.cos(i), self.c2c_dist * math.sin(i), 0])
            self.pillar_punch.set_homomat(rm.homomat_from_posrot(tmp_p, np.eye(3)))
            self.pillar_punch.set_rgba(rgba=[0, .7, .7, .7])
            is_collided, collided_pts = self.pillar_punch.is_mcdwith(objcm_bend, toggle_contacts=True)
            if is_collided:
                if toggledebug:
                    for p in collided_pts:
                        gm.gen_sphere(p, radius=.0004).attach_to(base)
                    self.pillar_punch.attach_to(base)
                    gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                     epos=np.asarray((p[0], p[1], 0)),
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
                print(plate_angle, rm.angle_between_vectors(tmp_p, [-self.c2c_dist, 0, 0]))
                return self.pseq[0], self.rotseq[0], rm.angle_between_vectors(tmp_p, [-self.c2c_dist, 0, 0])
        print('No collided point found (punch pillar & plate)!')
        return None, None, None

    def __get_bended_pseq(self, center, r, rot_angle, lift_angle, step=math.pi / 180, toggledebug=False):
        tmp_pseq = []
        tmp_rotseq = []
        if rot_angle > 0:
            rng = (0, rot_angle + step)
        else:
            rng = (rot_angle, step)
            r = -r
            center = center + np.asarray([2 * (self.r_center + self.thickness), 0, 0])

        for a in np.arange(rng[0], rng[1], step):
            if lift_angle == 0:
                p = (r * math.cos(a), r * math.sin(a), 0)
            elif abs(lift_angle) >= 90:
                print("lift angle should be in -90~90 degree!")
                return None, None
            else:
                p = (r * math.cos(a), r * math.sin(a), a * r * math.tan(lift_angle))
            p = np.asarray(center) + np.asarray(p)
            tmp_pseq.append(p)
            tmp_rotseq.append(rm.rotmat_from_axangle((0, 0, 1), a).dot(rm.rotmat_from_axangle((1, 0, 0), lift_angle)))
            if toggledebug:
                gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, .5], radius=0.0002).attach_to(base)
        if rot_angle < 0:
            return tmp_pseq[::-1], tmp_rotseq[::-1]
        else:
            return tmp_pseq, tmp_rotseq

    def bend(self, rot_angle, lift_angle, insert_l=None, toggledebug=False):
        # rot_end = np.dot(rm.rotmat_from_axangle((1, 0, 0), lift_angle), rm.rotmat_from_axangle((0, 0, 1), rot_angle))
        rot_end = rm.rotmat_from_axangle((0, 0, 1), rot_angle)
        if insert_l is not None:
            tmp_pseq, tmp_rotseq = self.__get_bended_pseq(center=np.asarray([0, 0, 0]), r=self.bend_r,
                                                          rot_angle=rot_angle, lift_angle=0)
            arc_l = abs(rot_angle * self.bend_r / np.cos(lift_angle))
            start_inx = self.__insert_p(insert_l, toggledebug=False)
            end_inx = self.__insert_p(insert_l + arc_l, toggledebug=False)
            if toggledebug:
                gm.gen_sphere(self.pseq[start_inx], radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
                gm.gen_frame(self.pseq[start_inx], self.rotseq[start_inx], length=.01, thickness=.0004).attach_to(base)
                gm.gen_sphere(self.pseq[end_inx], radius=.0004, rgba=(0, 1, 1, 1)).attach_to(base)
                gm.gen_frame(self.pseq[end_inx], self.rotseq[start_inx], length=.01, thickness=.0004).attach_to(base)
            init_homomat = rm.homomat_from_posrot([self.bend_r, 0, 0], np.eye(3))
            start_homomat = rm.homomat_from_posrot(self.pseq[start_inx], self.rotseq[start_inx])
            org_end_homomat = rm.homomat_from_posrot(self.pseq[end_inx], self.rotseq[end_inx])

            # transmat4_start = rm.homomat_from_posrot((0, 0, 0), rot_start)
            # pseq_start = rm.homomat_transform_points(transmat4_start, self.pseq[:start_inx]).tolist()
            # rotseq_start = [np.dot(rot_start, r) for r in self.rotseq[:start_inx]]

            transmat4_mid = np.dot(start_homomat, np.linalg.inv(init_homomat))
            pseq_mid = rm.homomat_transform_points(transmat4_mid, tmp_pseq).tolist()
            rotseq_mid = [np.dot(self.rotseq[start_inx], r) for r in tmp_rotseq]

            new_end_homomat = rm.homomat_from_posrot(pseq_mid[-1], rotseq_mid[-1])
            transmat4_end = np.dot(new_end_homomat, np.linalg.inv(org_end_homomat))
            pseq_end = rm.homomat_transform_points(transmat4_end, self.pseq[end_inx:]).tolist()
            rotseq_end = [np.dot(transmat4_end[:3,:3], r) for r in self.rotseq[end_inx:]]

            self.pseq = self.pseq[:start_inx] + pseq_mid + pseq_end
            self.rotseq = self.rotseq[:start_inx] + rotseq_mid + rotseq_end

        else:
            tmp_pseq, tmp_rotseq = self.__get_bended_pseq(center=np.asarray([0, 0, 0]), r=self.bend_r,
                                                          rot_angle=rot_angle, lift_angle=lift_angle)
            if rot_angle > 0:
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

    def __insert_p(self, insert_l, toggledebug=False):
        tmp_l = 0
        for i in range(len(self.pseq) - 1):
            p1 = np.asarray(self.pseq[i])
            p2 = np.asarray(self.pseq[i + 1])
            r1 = self.rotseq[i]
            r2 = self.rotseq[i + 1]
            tmp_l += np.linalg.norm(p2 - p1)
            if tmp_l < insert_l:
                continue
            elif tmp_l > insert_l:
                insert_radio = (tmp_l - insert_l) / np.linalg.norm(p2 - p1)
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
            else:
                insert_pos = self.pseq[i]
                insert_rot = self.rotseq[i]
                insert_inx = i

            if toggledebug:
                gm.gen_sphere(insert_pos, radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
                gm.gen_frame(insert_pos, insert_rot, length=.01, thickness=.0004).attach_to(base)

            return insert_inx
        print('error', insert_l)

    def feed(self, pos_diff):
        pos = np.asarray(pos_diff)
        self.pseq = rm.homomat_transform_points(rm.homomat_from_posrot(pos, np.eye(3)), self.pseq).tolist()
        # self.rotseq = [self.rotseq[0]] + self.rotseq
        # self.pseq = [[self.bend_r, 0, 0]] + self.pseq
        # print(len(self.pseq),len(self.rotseq))

    def gen_surface(self, pseq, toggledebug=False):
        tmp_vertices = []
        tmp_faces = []
        pseq = pseq[::-1]
        for p in pseq:
            tmp_vertices.append(p + np.asarray([0, 0, self.width / 2]))
            tmp_vertices.append(p - np.asarray([0, 0, self.width / 2]))
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

    def update_cm(self):
        vertices, faces = self.gen_surface(self.pseq)
        objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        self.objcm = cm.CollisionModel(initor=objtrm, btwosided=True)

    def show(self, rgba=(1, 1, 1, 1), objmat4=None, show_frame=False, show_pseq=False):
        self.update_cm()
        objcm = copy.deepcopy(self.objcm)
        if show_frame:
            for i in range(len(self.pseq)):
                gm.gen_frame(self.pseq[i], self.rotseq[i], length=.005, thickness=.0005, alpha=.5).attach_to(base)
        if show_pseq:
            for p in bu.linear_inp3d(self.pseq):
                gm.gen_sphere(p, radius=.0004, rgba=rgba).attach_to(base)
            # gm.gen_sphere(self.pseq[0], radius=.0004, rgba=(1, 0, 0, 1)).attach_to(base)
            # gm.gen_sphere(self.pseq[-1], radius=.0004, rgba=(0, 1, 0, 1)).attach_to(base)
        if objmat4 is not None:
            objcm.set_homomat(objmat4)

        objcm.set_rgba(rgba)
        objcm.attach_to(base)

    def unshow(self):
        self.objcm.detach()

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
            bs.cal_startp(bendseq[motioncounter[0]][2], dir=1 if bendseq[motioncounter[0]][0] > 0 else 0,
                          lift_angle=bendseq[motioncounter[0]][1], toggledebug=True)
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
        for motion in motionseq:
            if motion[0] == 'b':
                self.bend(motion[1], motion[2])
            else:
                self.feed(motion[1])

    def gen_by_bendseq(self, bendseq, toggledebug=False):
        for bend in bendseq:
            # pos, rot, angle = \
            #     self.cal_startp(bend[2], dir=0 if bend[0] < 0 else 1, lift_angle=bend[1], toggledebug=toggledebug)
            self.bend(bend[0], bend[1], bend[2], toggledebug=toggledebug)

    def cal_length(self):
        length = 0
        for i in range(len(self.pseq)):
            if i != 0:
                length += np.linalg.norm(np.asarray(self.pseq[i]) - np.asarray(self.pseq[i - 1]))
        return length

    def move_to_org(self, l, dir=1, lift_angle=0, toggledebug=False):
        inx = self.__insert_p(l, toggledebug=False)
        init_homomat = rm.homomat_from_posrot([self.bend_r, 0, 0], np.eye(3))
        goal_homomat = rm.homomat_from_posrot(self.pseq[inx], self.rotseq[inx])
        transmat4 = np.dot(init_homomat, np.linalg.inv(goal_homomat))
        self.pseq = rm.homomat_transform_points(transmat4, self.pseq).tolist()
        self.rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in self.rotseq])
        if dir == 0:
            rot = rm.rotmat_from_axangle((0, 0, 1), np.pi)
            self.pseq = self.__rot_new_orgin(self.pseq, np.asarray((-self.bend_r, 0, 0)), rot)
            self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])
        if lift_angle != 0:
            rot = rm.rotmat_from_axangle((1, 0, 0), lift_angle)
            trans_lift = rm.homomat_from_posrot((0, 0, 0), rot)
            self.pseq = rm.homomat_transform_points(trans_lift, self.pseq).tolist()
            self.rotseq = np.asarray([rot.dot(r) for r in self.rotseq])
        self.update_cm()
        if toggledebug:
            gm.gen_sphere(self.pseq[inx], radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
            gm.gen_frame(self.pseq[inx], self.rotseq[inx], length=.01, thickness=.0004).attach_to(base)
            self.show(rgba=(.7, .7, 0, .7))


if __name__ == '__main__':
    import pickle

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    thickness = .0015
    # thickness = 0
    width = .002
    motion_seq = [
        ['f', (0, .01, 0)],
        ['b', np.radians(-60), np.radians(0)],
        ['f', (0, .01, 0)],
        ['b', np.radians(60), np.radians(-20)],
        ['f', (0, .02, 0)],
        ['b', np.radians(1), np.radians(0)],
    ]
    # bendseq = [
    #     [np.radians(-60), np.radians(0), .02],
    #     [np.radians(60), np.radians(0), .025],
    #     # [np.radians(40), np.radians(0), .06],
    #     # [np.radians(-15), np.radians(0), .08],
    #     # [np.radians(20), np.radians(0), .1]
    # ]
    bendseq = pickle.load(open('./tmp_bendseq.pkl', 'rb'))

    bs = BendSim(thickness, width, show=True)
    # bs.gen_by_motionseq(motion_seq)
    bs.reset([(0, 0, 0), (0, bendseq[-1][2], 0)], [np.eye(3), np.eye(3)])
    bs.show(rgba=(.7, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))

    bs.gen_by_bendseq(bendseq, toggledebug=True)
    # bs.show_ani_bendseq(bendseq)
    bs.show(rgba=(0, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)), show_pseq=True,
            show_frame=True)
    # base.run()
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
