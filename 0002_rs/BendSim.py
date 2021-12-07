import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import utils.math_utils as mu
import basis.trimesh as trm
import modeling.collision_model as cm


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
        self.bend_r = self.r_center + self.thickness

        # bending result
        self.objcm = None
        if pseq is None:
            self.pseq = [[self.bend_r, 0, 0]]
            self.rotseq = [np.eye(3)]
        else:
            self.pseq = [np.asarray(p) + np.asarray([self.r_center + self.thickness, 0, 0]) for p in pseq]
            self.rotseq = rotseq
        if show:
            gm.gen_frame(thickness=.001, length=.05).attach_to(base)
            # gen pillars
            self.pillar_center = gm.gen_stick(spos=np.asarray([0, 0, -.02]), epos=np.asarray([0, 0, .02]),
                                              thickness=self.r_center * 2, sections=180, rgba=[.7, .7, .7, .5])
            self.pillar_moveside = gm.gen_stick(spos=np.asarray([self.c2c_dist, 0, -.02]),
                                                epos=np.asarray([self.c2c_dist, 0, .02]),
                                                thickness=self.r_side * 2, sections=180, rgba=[.7, .7, .7, .5])
            a = np.radians(-40)
            self.pillar_fixside = gm.gen_stick(
                spos=np.asarray([self.c2c_dist * np.cos(a), self.c2c_dist * np.sin(a), -.02]),
                epos=np.asarray([self.c2c_dist * np.cos(a), self.c2c_dist * np.sin(a), .02]),
                thickness=self.r_side * 2, sections=180, rgba=[.7, .7, .7, .5])

            self.pillar_center.attach_to(base)
            self.pillar_moveside.attach_to(base)
            self.pillar_fixside.attach_to(base)

    def reset(self, pseq,rotseq):
        self.pseq = [np.asarray(p) + np.asarray([self.r_center + self.thickness, 0, 0]) for p in pseq]
        self.rotseq = rotseq
        self.objcm = None

    def cal_tail(self):
        A = np.mat([[self.r_side + self.thickness, -self.r_center], [1, 1]])
        b = np.mat([0, self.c2c_dist]).T
        l_center, l_side = np.asarray(np.linalg.solve(A, b))

        return l_center[0], l_side[0], \
               np.sqrt(l_center[0] ** 2 - self.r_center ** 2) + \
               np.sqrt(l_side[0] ** 2 - (self.r_side + self.thickness) ** 2)

    def cal_safe_margin(self):
        return np.degrees(2 * np.arcsin((self.r_side + self.r_base) / (2 * self.c2c_dist)))

    def cal_start_margin(self, l_center):
        return np.degrees(2 * np.arccos(self.r_center / l_center))

    def cal_startp(self, pos_l=None, toggledebug=False):
        objcm_copy = copy.deepcopy(self.objcm)
        if pos_l is not None:
            p, inx = self.__insert_p(pos_l)
            homomat = rm.homomat_from_posrot(self.pseq[inx], self.rotseq[inx])
            inithomomat = rm.homomat_from_posrot([self.r_center + self.thickness, 0, 0], np.eye(3))
            transmat4 = np.dot(inithomomat, np.linalg.inv(homomat))
            objcm_copy.set_homomat(transmat4)
        objcm_copy.set_rgba(rgba=(.7, .7, 0, .7))
        objcm_copy.attach_to(base)
        for i in np.linspace(0, 180, 360):
            tmp_p = np.asarray([self.c2c_dist * math.cos(np.radians(i)), self.c2c_dist * math.sin(np.radians(i)), 0])
            pillar = cm.gen_stick(spos=np.asarray([0, 0, -.02]) + tmp_p, epos=np.asarray([0, 0, .02]) + tmp_p,
                                  thickness=self.r_side * 2, sections=180, rgba=[0, .7, .7, .7])
            is_collided, collided_pts = pillar.is_mcdwith(objcm_copy, toggle_contacts=True)
            if is_collided:
                for p in collided_pts:
                    if toggledebug:
                        pillar.attach_to(base)
                        gm.gen_sphere(p, radius=.0004).attach_to(base)
                        gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                         epos=np.asarray((p[0], p[1], 0)),
                                         thickness=.0005).attach_to(base)
                        gm.gen_dashstick(spos=np.asarray((0, 0, 0)),
                                         epos=np.asarray(tmp_p),
                                         thickness=.0005).attach_to(base)

                    return np.degrees(rm.angle_between_vectors(tmp_p, [self.c2c_dist, 0, 0]))
        print('No collided point found!')
        return None

    def __get_bended_pseq(self, center, r, rot_angle, lift_angle, toggledebug=False):
        tmp_pseq = []
        tmp_rotseq = []
        if rot_angle > 0:
            rng = (0, rot_angle + math.pi / 90)
        else:
            rng = (rot_angle, math.pi / 90)
            r = -r
            center = center + np.asarray([2 * (self.r_center + self.thickness), 0, 0])

        for a in np.arange(rng[0], rng[1], math.pi / 90):
            if lift_angle == 0:
                p = (r * math.cos(a), r * math.sin(a), 0)
            elif abs(lift_angle) >= 90:
                print("lift angle should be in -90~90 degree!")
                return None
            else:
                p = (r * math.cos(a), r * math.sin(a), a * r * math.tan(lift_angle))
            p = np.asarray(center) + np.asarray(p)
            tmp_pseq.append(p)
            tmp_rotseq.append(rm.rotmat_from_axangle((0, 0, 1), a))
            if toggledebug:
                gm.gen_sphere(pos=np.asarray(p), rgba=[1, 0, 0, .5], radius=0.0002).attach_to(base)
        if rot_angle < 0:
            return tmp_pseq[::-1], tmp_rotseq[::-1]
        else:
            return tmp_pseq, tmp_rotseq

    def gen_cm(self, vertices, faces):
        objtrm = trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        objcm = cm.CollisionModel(initor=objtrm, btwosided=True)
        return objcm

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

    def bend(self, rot_angle, lift_angle, insert_l=None, toggledebug=False):
        bend_r = self.r_center + self.thickness
        tmp_pseq, tmp_rotseq = self.__get_bended_pseq(center=np.asarray([0, 0, 0]), r=bend_r,
                                                      rot_angle=rot_angle, lift_angle=lift_angle)
        rot = np.dot(rm.rotmat_from_axangle((1, 0, 0), lift_angle), rm.rotmat_from_axangle((0, 0, 1), rot_angle))
        if insert_l is not None:
            arc_l = rot_angle * bend_r / np.cos(lift_angle)
            start_p, start_inx = self.__insert_p(insert_l, toggledebug=toggledebug)
            end_p, end_inx = self.__insert_p(insert_l + arc_l, toggledebug=toggledebug)
            init_homomat = rm.homomat_from_posrot([bend_r, 0, 0], np.eye(3))
            start_homomat = rm.homomat_from_posrot(self.pseq[start_inx], self.rotseq[start_inx])
            org_end_homomat = rm.homomat_from_posrot(self.pseq[end_inx], self.rotseq[end_inx])

            transmat4_start = np.dot(start_homomat, np.linalg.inv(init_homomat))
            pseq_mid = rm.homomat_transform_points(transmat4_start, tmp_pseq).tolist()
            rotseq_mid = [np.dot(self.rotseq[start_inx], r) for r in tmp_rotseq]

            new_end_homomat = rm.homomat_from_posrot(pseq_mid[-1], rotseq_mid[-1])
            transmat4_remain = np.dot(new_end_homomat, np.linalg.inv(org_end_homomat))
            pseq_end = rm.homomat_transform_points(transmat4_remain, self.pseq[end_inx:]).tolist()
            rotseq_end = [np.dot(rot, r) for r in self.rotseq[end_inx:]]

            self.pseq = self.pseq[:start_inx] + pseq_mid + pseq_end
            self.rotseq = self.rotseq[:start_inx] + rotseq_mid + rotseq_end

        else:
            if rot_angle > 0:
                pos = np.asarray((0, 0, 0))
                self.pseq = tmp_pseq + rm.homomat_transform_points(rm.homomat_from_posrot(pos, rot), self.pseq).tolist()
            else:
                pos = np.asarray((-2 * bend_r, 0, 0))
                self.pseq = tmp_pseq + self.__rot_new_orgin(self.pseq, pos, rot)
            self.rotseq = tmp_rotseq + [np.dot(rot, org_rot) for org_rot in self.rotseq]

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
            # print(p1, p2)
            # print(np.linalg.norm(p2 - p1), tmp_l)
            if tmp_l < insert_l:
                continue
            else:
                if tmp_l > insert_l:
                    insert_radio = (tmp_l - insert_l) / np.linalg.norm(p2 - p1)
                    insert_pos = p2 - insert_radio * (p2 - p1)
                    if (r1 == r2).all():
                        insert_rot = r1
                    else:
                        rotmat_list = rm.rotmat_slerp(r1, r2, 10)
                        inx = np.floor(insert_radio * len(rotmat_list))-1
                        if inx>9:
                            inx=9
                        # print(tmp_l,insert_l)
                        # print(insert_radio,inx, int(inx))
                        insert_rot = rotmat_list[int(inx)]
                        # print(len(rotmat_list), int(insert_radio * len(rotmat_list)))
                    insert_inx = i + 1
                else:
                    insert_pos = self.pseq[i]
                    insert_rot = self.rotseq[i]
                    insert_inx = i
                # print(i)
                # print(self.pseq[:i])
                # print(self.pseq[i:])
                # print([insert_pos])
                self.pseq = self.pseq[:i + 1] + [insert_pos] + self.pseq[i + 1:]
                self.rotseq = self.rotseq[:i + 1] + [insert_rot] + self.rotseq[i + 1:]
                if toggledebug:
                    gm.gen_sphere(insert_pos, radius=.0004, rgba=(1, 1, 0, 1)).attach_to(base)
                    gm.gen_frame(insert_pos, insert_rot, length=.01, thickness=.0004).attach_to(base)
                return insert_pos, insert_inx
        print('error', insert_l)

    def feed(self, pos_diff):
        pos = np.asarray(pos_diff)
        self.pseq = [[self.r_center + self.thickness, 0, 0]] + \
                    rm.homomat_transform_points(rm.homomat_from_posrot(pos, np.eye(3)), self.pseq).tolist()
        self.rotseq = [self.rotseq[0]] + self.rotseq

    def update_cm(self):
        vertices, faces = self.gen_surface(self.pseq)
        self.objcm = self.gen_cm(vertices, faces)

    def show(self, rgba=(1, 1, 1, 1)):
        self.update_cm()
        for i in range(len(self.pseq)):
            gm.gen_frame(self.pseq[i], self.rotseq[i], length=.01, thickness=.0005, alpha=.1).attach_to(base)
        self.objcm.set_rgba(rgba)
        self.objcm.attach_to(base)

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
            self.reset([])
        return task.again

    def show_ani(self, motion_seq):
        motioncounter = [0]
        taskMgr.doMethodLater(2, self.__update_ani, "update",
                              extraArgs=[motioncounter, motion_seq], appendTask=True)

    def gen_by_motionseq(self, motion_seq):
        for motion in motion_seq:
            if motion[0] == 'b':
                self.bend(motion[1], motion[2])
            else:
                self.feed(motion[1])

    def cal_length(self):
        length = 0
        for i in range(len(self.pseq)):
            if i != 0:
                length += np.linalg.norm(np.asarray(self.pseq[i]) - np.asarray(self.pseq[i - 1]))
        return length


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    thickness = .0015
    width = .002
    motion_seq = [['b', np.radians(60), np.radians(0)],
                  ['f', (0, .01, 0)],
                  ['b', np.radians(-60), np.radians(0)],
                  ['f', (0, .01, 0)],
                  ]

    bs = BendSim(thickness, width, show=True)
    bs.gen_by_motionseq(motion_seq)
    # bs.show()
    # threshold = bs.cal_startp(toggledebug=True)
    # print(threshold)
    # threshold = bs.cal_startp(pos_l=.012, toggledebug=True)
    # print(threshold)
    bs.bend(np.radians(100), np.radians(0), insert_l=0)
    bs.show(rgba=(.7, .7, 0, .7))
    print(bs.pseq)

    init_pseq = [(0, 0, 0), (0, .01, 0), (0, .05, 0)]
    init_rotseq = [np.eye(3), np.eye(3), np.eye(3)]
    bs.reset(init_pseq, init_rotseq)
    bs.show(rgba=(.7, 0, 0, .7))
    print(bs.pseq)




    # bs.show_ani(motion_seq)
    base.run()
