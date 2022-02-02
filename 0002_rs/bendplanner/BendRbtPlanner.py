import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import bend_utils as bu
import bender_config as bconfig
import time
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import utils.panda3d_utils as p3u
import PremutationTree as p_tree
from scipy import interpolate
import matplotlib.pyplot as plt


class BendRbtPlanner(object):
    def __init__(self, bendsim, motionplanner):
        self.bs = bendsim
        self.rbt = motionplanner.rbt
        self.mp = motionplanner

    def reset_bs(self, pseq, rotseq, extend=True):
        self.bs.reset(pseq, rotseq, extend)

    def __ps2seg_max_dist(self, p1, p2, ps):
        p1_p = np.asarray([p1] * len(ps)) - np.asarray(ps)
        p2_p = np.asarray([p2] * len(ps)) - np.asarray(ps)
        p1_p_norm = np.linalg.norm(p1_p, axis=1)
        p2_p_norm = np.linalg.norm(p2_p, axis=1)
        p2_p1 = np.asarray([p2 - p1] * len(ps))
        dist_list = abs(np.linalg.norm(np.cross(p2_p1, p1_p), axis=1) / np.linalg.norm(p2 - p1))

        l1 = np.arccos(np.sum((p1_p / p1_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1))
        l2 = np.arccos(np.sum((p2_p / p2_p_norm.reshape((len(ps), 1))) * (p2_p1 / np.linalg.norm(p2 - p1)), axis=1))
        l1 = (l1[:] < math.pi / 2).astype(int)
        l2 = (l2[:] > math.pi / 2).astype(int)

        dist_list = np.multiply(p1_p_norm, l1) + np.multiply(p2_p_norm, l2) + np.multiply(dist_list, 1 - l1 - l2)
        max_dist = max(dist_list)

        return max_dist, list(dist_list).index(max_dist)

    def iter_fit(self, pseq, tor=.001, toggledebug=False):
        pseq = np.asarray(pseq)
        res_pids = [0, len(pseq) - 1]
        ptr = 0
        while ptr < len(res_pids) - 1:
            max_err, max_inx = self.__ps2seg_max_dist(pseq[res_pids[ptr]], pseq[res_pids[ptr + 1]],
                                                      pseq[res_pids[ptr]:res_pids[ptr + 1]])
            if max_err > tor:
                res_pids.append(max_inx + res_pids[ptr])
                res_pids = sorted(res_pids)
            else:
                ptr += 1
            if toggledebug:
                ax = plt.axes(projection='3d')
                bu.plot_pseq(ax, pseq)
                bu.plot_pseq(ax, bu.linear_inp3d_by_step(pseq[res_pids]))
                bu.plot_pseq(ax, pseq[res_pids])
                plt.show()
        print('Num. of fitting result:', len(res_pids))
        return pseq[res_pids]

    def pseq2bendset(self, res_pseq, mode='rot', bend_r=bconfig.R_BEND, init_l=bconfig.INIT_L, toggledebug=False):
        ax = plt.axes(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        tangent_pts = []
        bendseq = []
        pos = 0
        diff_list = []
        n_pre = None
        rot_a = 0
        lift_a = 0
        for i in range(1, len(res_pseq) - 1):
            v1 = res_pseq[i] - res_pseq[i - 1]
            v2 = res_pseq[i + 1] - res_pseq[i]
            pos += np.linalg.norm(v1)
            rot_n = np.cross(rm.unit_vector(v1), rm.unit_vector(v2))
            bend_a = rm.angle_between_vectors(v1, v2)
            if n_pre is not None:
                a = rm.angle_between_vectors(n_pre, rot_n)
            else:
                a = 0
            if mode == 'lift':
                if a > np.pi / 2:
                    a = np.pi - a
                    bend_a = -bend_a
                # if n_pre is not None and rm.angle_between_vectors(v1, np.cross(n_pre, rot_n)) > np.pi / 2:
                #     lift_a += a
                # else:
                #     lift_a -= a
            else:
                if n_pre is not None and rm.angle_between_vectors(v1, np.cross(n_pre, rot_n)) > np.pi / 2:
                    rot_a += a
                else:
                    rot_a -= a

            n_pre = rot_n
            l = (bend_r / np.tan((np.pi - abs(bend_a)) / 2)) / np.cos(abs(lift_a))
            arc = abs(bend_a) * bend_r
            bendseq.append([bend_a, lift_a, rot_a, pos + init_l - l - sum(diff_list)])
            diff_list.append(2 * l - arc)

            ratio_1 = l / np.linalg.norm(res_pseq[i] - res_pseq[i - 1])
            p1 = res_pseq[i] + (res_pseq[i - 1] - res_pseq[i]) * ratio_1
            ratio_2 = l / np.linalg.norm(res_pseq[i] - res_pseq[i + 1])
            p2 = res_pseq[i] + (res_pseq[i + 1] - res_pseq[i]) * ratio_2
            tangent_pts.append(p1)
            tangent_pts.append(p2)

            x = np.cross(v1, rot_n)
            rot = np.asarray([rm.unit_vector(x), rm.unit_vector(v1), rm.unit_vector(rot_n)]).T
            if toggledebug:
                gm.gen_frame(res_pseq[i - 1], rot, length=.02, thickness=.001).attach_to(base)
            bu.plot_frame(ax, res_pseq[i - 1], rot)
            if i == 1:
                init_rot = rot

        bu.plot_pseq(ax, res_pseq)
        bu.scatter_pseq(ax, res_pseq, s=5)
        bu.scatter_pseq(ax, tangent_pts, s=5)
        plt.show()

        return bendseq, init_rot

    def cal_rbtik(self, bendresseq, grasp, transmat4, max_fail=np.inf):
        armjntsseq = []
        fail_cnt = 0
        self.bs.move_posrot(transmat4)
        for bendres in bendresseq:
            init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres
            pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
            rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
            self.bs.reset(pseq_init, rotseq_init, extend=False)
            objcm_init = copy.deepcopy(self.bs.objcm)
            pseq_end = rm.homomat_transform_points(transmat4, pseq_end).tolist()
            rotseq_end = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_end])
            self.bs.reset(pseq_end, rotseq_end, extend=False)
            objcm_end = copy.deepcopy(self.bs.objcm)
            objpos = pseq_init[0]
            objrot = rotseq_init[0]
            armjnts = self.mp.get_armjnts_by_objmat4ngrasp(grasp, [objcm_init, objcm_end] + self.bs.staticobs_list(),
                                                           rm.homomat_from_posrot(objpos, objrot))
            # is_collided = self.rbt.lft_hnd.is_mesh_collided([objcm_init, objcm_end] + self.bs.staticobs_list())
            # print(is_collided)
            # if is_collided:
            #     self.mp.ah.show_armjnts()
            #     base.run()

            if armjnts is None:
                fail_cnt += 1
                if fail_cnt > max_fail:
                    break
            armjntsseq.append(armjnts)
        return armjntsseq

    def grasp_reasoning(self, g_list, bendresseq, transmat4=np.eye(4)):
        min_fail = np.inf
        armjntsseq = None
        g = None
        for i, g_tmp in enumerate(g_list):
            print(f'----------grasp_id: {i}----------')
            armjntsseq_tmp = brp.cal_rbtik(bendresseq, g_tmp, transmat4, max_fail=min_fail)
            fail_cnt = [str(v) for v in armjntsseq_tmp].count('None')
            print(min_fail, fail_cnt)
            if fail_cnt < min_fail:
                min_fail = fail_cnt
                g = g_tmp
                armjntsseq = armjntsseq_tmp
            if fail_cnt == 0:
                return g_tmp, armjntsseq
        return g, armjntsseq

    def pnp_plan(self, bendresseq, grasp, armjntsseq, transmat4):
        print(f'----------plan pick & place----------')
        pathseq = []
        for i, armjnts in enumerate(armjntsseq[:-1]):
            if armjnts is None or armjntsseq[i + 1] is None:
                pathseq.append(None)
                continue
            _, _, _, _, _, pseq_init, rotseq_init = bendresseq[i]
            _, _, _, pseq_end, rotseq_end, _, _ = bendresseq[i + 1]

            pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
            rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
            pseq_end = rm.homomat_transform_points(transmat4, pseq_end).tolist()
            rotseq_end = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_end])
            objmat4_end = np.linalg.inv(rm.homomat_from_posrot(pseq_init[0], rotseq_init[0])) \
                .dot(rm.homomat_from_posrot(pseq_end[0], rotseq_end[0]))

            self.reset_bs(pseq_init, rotseq_init, extend=False)
            objcm = copy.deepcopy(self.bs.objcm)
            path = self.mp.plan_picknplace(grasp, [np.eye(4), objmat4_end],
                                           objcm, use_msc=True, start=armjnts, goal=armjntsseq[i + 1],
                                           use_pickupprim=True, use_placedownprim=True,
                                           pickupprim_len=.1, placedownprim_len=.1)
            if path is not None:
                pathseq.append(path)
        return pathseq

    def load_bendresseq(self, f_name='./tmp_bendresseq.pkl'):
        return pickle.load(open(f_name, 'rb'))

    def show_bendresseq(self, bendresseq, transmat4):
        motioncounter = [0]
        taskMgr.doMethodLater(.05, self.__update, "update_bendresseq",
                              extraArgs=[motioncounter, bendresseq, transmat4], appendTask=True)

    def show_bendresseq_withrbt(self, bendresseq, transmat4, armjntsseq):
        motioncounter = [0]
        rbtmnp = [None, None]
        taskMgr.doMethodLater(.05, self.__update_rbt, "update_rbt_bendresseq",
                              extraArgs=[rbtmnp, motioncounter, bendresseq, transmat4, armjntsseq], appendTask=True)

    def show_motion_withrbt(self, bendresseq, transmat4, pathseq):
        motioncounter = [0]
        # rbtmnp = [None, None]
        # path = [None]
        taskMgr.doMethodLater(.05, self.__update_rbt_motion, "update_rbt_motion",
                              extraArgs=[motioncounter, bendresseq, transmat4, pathseq], appendTask=True)

    def __update(self, motioncounter, bendresseq, transmat4, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            self.bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
                rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
                pseq_end = rm.homomat_transform_points(transmat4, pseq_end).tolist()
                rotseq_end = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_end])

                self.bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self.bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, .7))
                objcm_init.attach_to(base)

                self.bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self.bs.objcm)
                objcm_end.set_rgba((0, .7, 0, .7))
                objcm_end.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(init_a), self.bs.c2c_dist * math.sin(init_a), 0])
                self.bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self.bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(end_a), self.bs.c2c_dist * math.sin(end_a), 0])
                self.bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self.bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def __update_rbt(self, rbtmnp, motioncounter, bendresseq, transmat4, armjntsseq, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            self.bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                if rbtmnp[0] is not None:
                    rbtmnp[0].detach()
                armjnts = armjntsseq[motioncounter[0]]
                if armjnts is not None:
                    rbt.fk(self.mp.armname, armjnts)
                    rbtmnp[0] = rbt.gen_meshmodel()
                    rbtmnp[0].attach_to(base)

                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
                rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
                pseq_end = rm.homomat_transform_points(transmat4, pseq_end).tolist()
                rotseq_end = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_end])

                self.bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self.bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, .7))
                objcm_init.attach_to(base)

                self.bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self.bs.objcm)
                objcm_end.set_rgba((0, .7, 0, .7))
                objcm_end.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(init_a), self.bs.c2c_dist * math.sin(init_a), 0])
                self.bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self.bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(end_a), self.bs.c2c_dist * math.sin(end_a), 0])
                self.bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self.bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def __update_rbt_motion(self, motioncounter, bendresseq, transmat4, pathseq, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj', 'auto'])
            self.bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq)-1:
                print('-------------')
                taskMgr.remove('update')
                path = pathseq[motioncounter[0]]
                if path is not None:
                    self.mp.ah.show_ani(path)

                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
                rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
                pseq_end = rm.homomat_transform_points(transmat4, pseq_end).tolist()
                rotseq_end = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_end])

                self.bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self.bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, .7))
                objcm_init.attach_to(base)

                self.bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self.bs.objcm)
                objcm_end.set_rgba((0, .7, 0, .7))
                objcm_end.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(init_a), self.bs.c2c_dist * math.sin(init_a), 0])
                self.bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self.bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self.bs.c2c_dist * math.cos(end_a), self.bs.c2c_dist * math.sin(end_a), 0])
                self.bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self.bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self.bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again


if __name__ == '__main__':
    import pickle
    import localenv.envloader as el

    gripper = rtqhe.RobotiqHE()
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()

    bs = b_sim.BendSim(show=True)
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    brp = BendRbtPlanner(bs, mp_lft)

    # goal_pseq = bu.gen_polygen(5, .05)
    goal_pseq = bu.gen_ramdom_curve(kp_num=4, length=.12, step=.0005, z_max=.02, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([(0, 0, 0), (0, .02, 0), (.02, .02, 0), (.02, .03, .02), (0, .03, 0), (0, .03, -.02)])

    # init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0)]
    # init_rotseq = [np.eye(3), np.eye(3)]
    # brp.reset_bs(init_pseq, init_rotseq)
    #
    # fit_pseq = brp.iter_fit(goal_pseq, tor=.0005, toggledebug=False)
    # init_bendseq, init_rot = brp.pseq2bendset(fit_pseq, toggledebug=False)
    # pickle.dump(init_bendseq, open('./tmp_bendseq.pkl', 'wb'))
    #
    # is_success, bendresseq = bs.gen_by_bendseq(init_bendseq, cc=True, toggledebug=False)
    # print('Result Flag:', is_success)
    #
    # goal_pseq, res_pseq = bu.align_with_goal(bs, goal_pseq, init_rot)
    # err, _ = bu.avg_distance_between_polylines(res_pseq, goal_pseq, toggledebug=True)
    # pickle.dump(bendresseq, open('./tmp_bendresseq.pkl', 'wb'))
    #
    # bu.show_pseq(bs.pseq, rgba=(1, 0, 0, 1))
    # bu.show_pseq(bu.linear_inp3d_by_step(goal_pseq), rgba=(0, 1, 0, 1))
    # bs.show(rgba=(.7, .7, .7, .7))

    bendseq = pickle.load(open('./tmp_bendseq.pkl', 'rb'))
    transmat4 = rm.homomat_from_posrot((.8, .3, .78 + 0.15175), np.eye(3))
    bendresseq = pickle.load(open('./tmp_bendresseq.pkl', 'rb'))

    grasp_list = mp_lft.load_all_grasp('stick')[370:380]
    print(len(grasp_list))

    grasp, armjntsseq = brp.grasp_reasoning(grasp_list, bendresseq, transmat4)
    pickle.dump([grasp, armjntsseq], open('./tmp_armjntsseq.pkl', 'wb'))
    grasp, armjntsseq = pickle.load(open('./tmp_armjntsseq.pkl', 'rb'))
    # brp.show_bendresseq_withrbt(bendresseq, transmat4, armjntsseq)

    pathseq = brp.pnp_plan(bendresseq, grasp, armjntsseq, transmat4)
    brp.show_motion_withrbt(bendresseq, transmat4, pathseq)

    base.run()
