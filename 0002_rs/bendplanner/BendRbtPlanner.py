import copy
import math
import pickle

import numpy as np

import basis.robot_math as rm
import bendplanner.BendSim as b_sim
import bendplanner.InvalidPermutationTree as ip_tree
import bendplanner.PremutationTree as p_tree
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import motionplanner.motion_planner as m_planner
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import utils.panda3d_utils as p3u
import modeling.geometric_model as gm


class BendRbtPlanner(object):
    def __init__(self, bendsim, init_pseq, init_rotseq, motionplanner=None):
        self._bs = bendsim
        self.init_pseq = init_pseq
        self.init_rotseq = init_rotseq
        self.reset_bs(init_pseq, init_rotseq)

        self._mp = motionplanner
        self.rbt = motionplanner.rbt
        self.env = motionplanner.env
        self.gripper = motionplanner.gripper
        self.obslist = self.env.getstationaryobslist() + self.env.getchangableobslist()

        self._ptree = p_tree.PTree(0)
        self._iptree = ip_tree.IPTree(0)

        self.bendset = []
        self.grasp_list = []
        self.transmat4 = np.eye(4)

    def reset_bs(self, pseq, rotseq, extend=True):
        self._bs.reset(pseq, rotseq, extend)

    def set_bs_stick_sec(self, stick_sec):
        self._bs.set_stick_sec(stick_sec)

    def set_up(self, bendset, grasp_list, transmat4):
        # self.ptree = p_tree.PTree(len(bendset))
        self._iptree = ip_tree.IPTree(len(bendset))
        self.bendset = bendset
        self.grasp_list = grasp_list
        self.transmat4 = transmat4
        for obs in self._bs.staticobs_list():
            obs_rp = obs.copy()
            obs_rp.set_homomat(self.transmat4)
            self.obslist.append(obs_rp)
            # obs.show_cdprimit()
            obs_rp.attach_to(base)
            self._mp.add_obs(obs_rp)

    def transseq(self, pseq, rotseq, transmat4):
        return rm.homomat_transform_points(transmat4, pseq).tolist(), \
               np.asarray([transmat4[:3, :3].dot(r) for r in rotseq])

    def load_bendresseq(self, f_name='./penta_bendresseq.pkl'):
        return pickle.load(open(f_name, 'rb'))

    def cal_rbtik(self, bendresseq, grasp, grasp_l=0.0, max_fail=np.inf):
        armjntsseq = []
        fail_cnt = 0
        msc = None
        for bendres in bendresseq:
            _, _, _, pseq_init, rotseq_init, _, _ = bendres
            pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
            self._bs.reset(pseq_init, rotseq_init, extend=False)
            _, _, objpos, objrot = self._bs.get_posrot_by_l(grasp_l, pseq_init, rotseq_init)
            armjnts = self._mp.get_numik_hold(grasp, self.obslist, rm.homomat_from_posrot(objpos, objrot),
                                              obj=self._bs.objcm, msc=msc)
            armjntsseq.append(armjnts)
            msc = armjntsseq[-1] if len(armjntsseq) != 0 else None
            if armjnts is None:
                fail_cnt += 1
                if fail_cnt > max_fail:
                    break
        return armjntsseq

    def check_ik(self, bendresseq, grasp_l=0.0):
        min_fail = np.inf
        armjntsseq = None
        all_result = []
        for i, g_tmp in enumerate(self.grasp_list):
            print(f'----------grasp_id: {i}  of {len(self.grasp_list)}----------')
            armjntsseq_tmp = self.cal_rbtik(bendresseq, g_tmp, grasp_l=grasp_l, max_fail=min_fail)
            fail_cnt = [str(v) for v in armjntsseq_tmp].count('None')
            print(min_fail, fail_cnt)
            if fail_cnt < min_fail:
                min_fail = fail_cnt
                armjntsseq = armjntsseq_tmp
            # self._mp.ah.show_armjnts(rgba=(i * .01, 1 - i * .01, 0, .5))
            if fail_cnt == 0:
                all_result.append([g_tmp, armjntsseq_tmp])
        if len(all_result) == 0:
            self.show_bendresseq_withrbt(bendresseq, armjntsseq)
            base.run()
            return [str(v) for v in armjntsseq].index('None'), all_result

        return -1, all_result

    def pre_grasp_reasoning(self):
        print(f'----------pre-grasp reasoning----------')
        _, bendresseq, _ = self._bs.gen_by_bendseq(self.bendset, cc=True, toggledebug=False)
        objmat4_list = []
        for bendres in bendresseq:
            init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres
            pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
            objmat4_list.append(rm.homomat_from_posrot(pseq_init[0], rotseq_init[0]))
        # print(len(objmat4_list))

        failed_id_list = []
        for i, grasp in enumerate(self.grasp_list):
            for objmat4 in objmat4_list:
                hndpos, hndrot = self._mp.get_hnd_by_objmat4(grasp, objmat4)
                self.gripper.fix_to(hndpos, hndrot)
                if self.gripper.is_mesh_collided(self.obslist):
                    failed_id_list.append(i)
                    self.gripper.gen_meshmodel(rgba=(.7, 0, 0, .2)).attach_to(base)
                    break
                self.gripper.gen_meshmodel(rgba=(0, .7, 0, .2)).attach_to(base)

        self.grasp_list = [g for i, g in enumerate(self.grasp_list) if i not in failed_id_list]
        print('Remain grasp:', len(self.grasp_list))
        base.run()

    def plan_pnp(self, bendresseq, grasp, armjntsseq):
        print(f'----------plan pick & place----------')
        pathseq = [[armjntsseq[0]]]
        for i in range(len(armjntsseq) - 1):
            if armjntsseq[i] is None or armjntsseq[i + 1] is None:
                pathseq.append(None)
                continue
            _, _, _, _, _, pseq_init, rotseq_init = bendresseq[i]
            _, _, _, pseq_end, rotseq_end, _, _ = bendresseq[i + 1]

            pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
            pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, self.transmat4)
            objmat4_end = np.linalg.inv(rm.homomat_from_posrot(pseq_init[0], rotseq_init[0])) \
                .dot(rm.homomat_from_posrot(pseq_end[0], rotseq_end[0]))
            self.reset_bs(pseq_init, rotseq_init, extend=False)
            objcm = copy.deepcopy(self._bs.objcm)
            path = self._mp.plan_picknplace(grasp, [np.eye(4), objmat4_end], objcm,
                                            use_msc=True, start=armjntsseq[i], goal=armjntsseq[i + 1],
                                            use_pickupprim=True, use_placedownprim=True,
                                            pickupprim_len=.05, placedownprim_len=.05)
            pathseq.append(path)
        return pathseq

    def plan_pull(self, bendresseq, grasp, armjntsseq):
        print(f'----------plan linear motion----------')
        pathseq = [[armjntsseq[0]]]
        for i, armjnts in enumerate(armjntsseq[:-1]):
            _, _, _, _, _, pseq_init, rotseq_init = bendresseq[i]
            _, _, _, pseq_end, rotseq_end, _, _ = bendresseq[i + 1]

            pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
            pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, self.transmat4)

            objmat4_init = rm.homomat_from_posrot(pseq_init[0], rotseq_init[0])
            objmat4_end = rm.homomat_from_posrot(pseq_end[0], rotseq_end[0])
            self.reset_bs(pseq_init, rotseq_init, extend=False)
            objcm = copy.deepcopy(self._bs.objcm)
            objmat4_list = self._mp.objmat4_list_inp([objmat4_init, objmat4_end])
            path = self._mp.get_continuouspath_hold_ik(None, grasp, objmat4_list, objcm)
            # gm.gen_frame(pseq_init[0], rotseq_init[0], length=.01, thickness=.001).attach_to(base)
            # gm.gen_frame(pseq_end[0], rotseq_end[0], length=.01, thickness=.001).attach_to(base)
            pathseq.append(path)
        return pathseq

    def plan_motion(self, seqs, bendresseq, grasp, armjntsseq):
        print(f'----------plan linear motion----------')
        pathseq = [[armjntsseq[0]]]
        for i, armjnts in enumerate(armjntsseq[:-1]):
            _, _, _, _, _, pseq_init, rotseq_init = bendresseq[i]
            _, _, _, pseq_end, rotseq_end, _, _ = bendresseq[i + 1]

            pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
            pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, self.transmat4)

            objmat4_init = rm.homomat_from_posrot(pseq_init[0], rotseq_init[0])
            objmat4_end = rm.homomat_from_posrot(pseq_end[0], rotseq_end[0])
            self.reset_bs(pseq_init, rotseq_init, extend=False)
            objcm = copy.deepcopy(self._bs.objcm)
            if all([seqs[i + 1] > v for v in seqs[:i + 1]]):
                objmat4_list = self._mp.objmat4_list_inp([objmat4_init, objmat4_end])
                path = self._mp.get_continuouspath_hold_ik(None, grasp, objmat4_list, objcm)
            else:
                path = self._mp.plan_picknplace(grasp, [np.eye(4), objmat4_end], objcm,
                                                use_msc=True, start=armjntsseq[i], goal=armjntsseq[i + 1],
                                                use_pickupprim=True, use_placedownprim=True,
                                                pickupprim_len=.05, placedownprim_len=.05)
            # path = self.mp.plan_picknplace(grasp, [np.eye(4), objmat4_end], objcm,
            #                                use_msc=True, start=armjntsseq[i], goal=armjntsseq[i + 1],
            #                                use_pickupprim=True, use_placedownprim=True,
            #                                pickupprim_len=.05, placedownprim_len=.05)
            # gm.gen_frame(pseq_init[0], rotseq_init[0], length=.01, thickness=.001).attach_to(base)
            # gm.gen_frame(pseq_end[0], rotseq_end[0], length=.01, thickness=.001).attach_to(base)
            pathseq.append(path)
        return pathseq

    def check_motion(self, seqs, bendresseq, armjntsseq_list):
        min_fail = np.inf
        pathseq = None
        all_result = []
        for i, v in enumerate(armjntsseq_list):
            print(f'----------grasp_id: {i} of {len(armjntsseq_list)}----------')
            grasp_tmp, armjntsseq = v
            pathseq_tmp = self.plan_motion(seqs, bendresseq, grasp_tmp, armjntsseq)
            fail_cnt = [str(v) for v in pathseq_tmp].count('None')
            print(min_fail, fail_cnt)
            if fail_cnt < min_fail:
                min_fail = fail_cnt
                pathseq = pathseq_tmp
            if fail_cnt == 0:
                all_result.append([grasp_tmp, pathseq_tmp])
        # self.show_motion_withrbt(bendresseq, transmat4, pathseq)
        # base.run()
        if len(all_result) == 0:
            return [str(v) for v in pathseq].index('None'), all_result
        return -1, all_result

    def check_pull_motion(self, bendresseq, armjntsseq_list):
        min_fail = np.inf
        pathseq = None
        all_result = []
        for i, v in enumerate(armjntsseq_list):
            print(f'----------grasp_id: {i} of {len(armjntsseq_list)}----------')
            grasp_tmp, armjntsseq = v
            pathseq_tmp = self.plan_pull(bendresseq, grasp_tmp, armjntsseq)
            fail_cnt = [str(v) for v in pathseq_tmp].count('None')
            print(min_fail, fail_cnt)
            if fail_cnt < min_fail:
                min_fail = fail_cnt
                pathseq = pathseq_tmp
            if fail_cnt == 0:
                all_result.append([grasp_tmp, pathseq_tmp])
                # self.show_motion_withrbt(bendresseq, self.transmat4, pathseq_tmp)
                # base.run()

        if len(all_result) == 0:
            return [str(v) for v in pathseq].index('None'), all_result
        return -1, all_result

    def check_combine_motion(self, bendseq, bendresseq, armjntsseq_list):
        min_fail = np.inf
        pathseq = None
        all_result = []
        for i, v in enumerate(armjntsseq_list):
            print(f'----------grasp_id: {i} of {len(armjntsseq_list)}----------')
            grasp_tmp, armjntsseq = v
            pathseq_tmp = self.plan_pull(bendseq, bendresseq, grasp_tmp, armjntsseq)
            fail_cnt = [str(v) for v in pathseq_tmp].count('None')
            print(min_fail, fail_cnt)
            if fail_cnt < min_fail:
                min_fail = fail_cnt
                pathseq = pathseq_tmp
            if fail_cnt == 0:
                all_result.append([grasp_tmp, pathseq_tmp])
                self.show_motion_withrbt(bendresseq, pathseq_tmp)
                # base.run()

        if len(all_result) == 0:
            return [str(v) for v in pathseq].index('None'), all_result
        return -1, pathseq

    def run(self, f_name='tmp', grasp_l=0.0, folder_name='stick'):
        seqs, _ = self._iptree.get_potential_valid()
        while len(seqs) != 0:
            bendseq = [self.bendset[i] for i in seqs]
            self._iptree.show()
            print(seqs)
            self.reset_bs(self.init_pseq, self.init_rotseq)
            is_success, bendresseq, _ = self._bs.gen_by_bendseq(bendseq, cc=True, prune=True, toggledebug=False)
            # self.show_bendresseq(bendresseq, self.transmat4)
            # base.run()
            if not all(is_success):
                self._iptree.add_invalid_seq(seqs[:is_success.index(False) + 1])
                seqs, _ = self._iptree.get_potential_valid()
                continue
            pickle.dump([seqs, is_success, bendresseq],
                        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendresseq.pkl', 'wb'))
            seqs, _, bendresseq = pickle.load(
                open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendresseq.pkl', 'rb'))

            fail_index, armjntsseq_list = self.check_ik(bendresseq, grasp_l=grasp_l)
            if fail_index != -1:
                self._iptree.add_invalid_seq(seqs[:fail_index + 1])
                seqs, _ = self._iptree.get_potential_valid()
                continue
            pickle.dump(armjntsseq_list,
                        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_armjntsseq.pkl', 'wb'))
            # self.show_bendresseq_withrbt(bendresseq, armjntsseq_list[0][1])
            # base.run()
            seqs, _, bendresseq = pickle.load(
                open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendresseq.pkl', 'rb'))
            armjntsseq_list = pickle.load(
                open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_armjntsseq.pkl', 'rb'))
            fail_index, pathseq_list = self.check_motion(seqs, bendresseq, armjntsseq_list)
            # fail_index, pathseq_list = self.check_pull_motion(bendresseq, armjntsseq_list)
            if fail_index != -1:
                self._iptree.add_invalid_seq(seqs[:fail_index + 1])
                seqs = self._iptree.get_potential_valid()
                continue
            pickle.dump(pathseq_list,
                        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_pathseq.pkl', 'wb'))
            print(f'success {seqs}')
            break

    def check_force(self, bendresseq, pathseq):
        min_f_list = []
        pseq_init, rotseq_init = self.transseq(bendresseq[0][3], bendresseq[0][4], self.transmat4)
        for i in range(len(pathseq)):
            min_f = np.inf
            print(f'----------{i}----------')
            g, path_list = pathseq[i]
            for j in range(len(bendresseq)):
                init_a, end_a, plate_a, _, _, _, _ = bendresseq[j]
                f_dir = rm.rotmat_from_axangle((0, 0, 1), init_a / 2).dot(rotseq_init[0][:, 1])
                armjnts = path_list[j][0]
                eepos, eerot = self._mp.get_ee(armjnts)
                gm.gen_arrow(eepos, eepos + f_dir * .1).attach_to(base)
                # self._mp.ah.show_armjnts(armjnts=armjnts, rgba=None)
                f = self._mp.get_max_force(list(f_dir) + [0, 0, 0])
                if f < min_f:
                    min_f = f
            min_f_list.append(min_f)
        g, path_list = pathseq[min_f_list.index(min(min_f_list))]
        for path in path_list:
            self._mp.ah.show_armjnts(armjnts=path[0], rgba=(1, 1, 0, .5))
        g, path_list = pathseq[min_f_list.index(max(min_f_list))]
        for path in path_list:
            self._mp.ah.show_armjnts(armjnts=path[0], rgba=(0, 1, 1, .5))
        print(min_f_list.index(min(min_f_list)), min_f_list[min_f_list.index(min(min_f_list))])
        print(min_f_list.index(max(min_f_list)), min_f_list[min_f_list.index(max(min_f_list))])
        print(min_f_list)

    def show_bendresseq(self, bendresseq):
        motioncounter = [0]
        taskMgr.doMethodLater(.05, self._update, "update_bendresseq",
                              extraArgs=[motioncounter, bendresseq, self.transmat4], appendTask=True)

    def show_bendresseq_withrbt(self, bendresseq, armjntsseq):
        motioncounter = [0]
        rbtmnp = [None, None]
        taskMgr.doMethodLater(.05, self._update_rbt, "update_rbt_bendresseq",
                              extraArgs=[rbtmnp, motioncounter, bendresseq, self.transmat4, armjntsseq],
                              appendTask=True)

    def show_motion_withrbt(self, bendresseq, pathseq):
        motioncounter = [0]
        obj_hold = None
        taskMgr.doMethodLater(.05, self._update_rbt_motion, "update_rbt_motion",
                              extraArgs=[motioncounter, bendresseq, self.transmat4, pathseq, obj_hold, self.rbt],
                              appendTask=True)

    def _update(self, motioncounter, bendresseq, transmat4, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            self._bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, transmat4)
                pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, transmat4)
                # for i in range(len(pseq_init)):
                #     gm.gen_frame(pseq_init[i], rotseq_init[i], length=.01, thickness=.0004).attach_to(base)

                self._bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self._bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, .7))
                objcm_init.attach_to(base)

                self._bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self._bs.objcm)
                objcm_end.set_rgba((0, .7, 0, .7))
                objcm_end.attach_to(base)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(init_a), self._bs.c2c_dist * math.sin(init_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self._bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(end_a), self._bs.c2c_dist * math.sin(end_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self._bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def _update_rbt(self, rbtmnp, motioncounter, bendresseq, transmat4, armjntsseq, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj'])
            self._bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                if rbtmnp[0] is not None:
                    rbtmnp[0].detach()
                armjnts = armjntsseq[motioncounter[0]]
                if armjnts is not None:
                    self.rbt.fk(self._mp.armname, armjnts)
                    # objcm = cm.gen_sphere(pos=np.asarray([-.1, 0, 0]), radius=.001)
                    # self.rbt.hold(objcm, jawwidth=.01, hnd_name=self.mp.hnd_name)
                    rbtmnp[0] = self.rbt.gen_meshmodel()
                    rbtmnp[0].attach_to(base)

                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, transmat4)
                pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, transmat4)
                # gm.gen_frame(pseq_init[0], rotseq_init[0]).attach_to(base)
                self._bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self._bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, .7))
                objcm_init.attach_to(base)

                self._bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self._bs.objcm)
                objcm_end.set_rgba((0, .7, 0, .7))
                objcm_end.attach_to(base)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(init_a), self._bs.c2c_dist * math.sin(init_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self._bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(end_a), self._bs.c2c_dist * math.sin(end_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self._bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again

    def _update_rbt_motion(self, motioncounter, bendresseq, transmat4, pathseq, obj_hold, rbt, task):
        if base.inputmgr.keymap['space']:
            p3u.clearobj_by_name(['obj', 'auto'])
            self._bs.move_posrot(transmat4)
            if motioncounter[0] < len(bendresseq):
                print('-------------')
                taskMgr.remove('update')
                rbt.release_all(hnd_name=self._mp.hnd_name)
                path = pathseq[motioncounter[0]]
                if path is not None:
                    self._mp.ah.show_ani(path)

                if bendresseq[motioncounter[0]][0] is None:
                    print('Failed')
                    motioncounter[0] += 1
                    return task.again
                init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[motioncounter[0]]
                print(np.degrees(init_a), np.degrees(end_a), np.degrees(plate_a))

                pseq_init, rotseq_init = self.transseq(pseq_init, rotseq_init, self.transmat4)
                pseq_end, rotseq_end = self.transseq(pseq_end, rotseq_end, self.transmat4)

                self._bs.reset(pseq_init, rotseq_init, extend=False)
                objcm_init = copy.deepcopy(self._bs.objcm)
                objcm_init.set_rgba((.7, .7, 0, 1))
                objcm_init.attach_to(base)

                self._bs.reset(pseq_end, rotseq_end, extend=False)
                objcm_end = copy.deepcopy(self._bs.objcm)
                objcm_end.set_rgba((0, .7, 0, 1))
                objcm_end.attach_to(base)

                rbt.fk(self._mp.armname, path[-1])
                _, _ = rbt.hold(objcm=objcm_init.copy(), hnd_name=self._mp.hnd_name)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(init_a), self._bs.c2c_dist * math.sin(init_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch.set_rgba(rgba=[.7, 0, 0, .7])
                self._bs.pillar_punch.attach_to(base)

                tmp_p = np.asarray([self._bs.c2c_dist * math.cos(end_a), self._bs.c2c_dist * math.sin(end_a), 0])
                tmp_p = np.dot(transmat4[:3, :3], tmp_p)
                self._bs.pillar_punch_end.set_homomat(np.dot(rm.homomat_from_posrot(tmp_p, np.eye(3)), transmat4))
                self._bs.pillar_punch_end.set_rgba(rgba=[0, .7, 0, .7])
                self._bs.pillar_punch_end.attach_to(base)
                motioncounter[0] += 1
            else:
                motioncounter[0] = 0
            base.inputmgr.keymap['space'] = False
        return task.again


if __name__ == '__main__':
    import localenv.envloader as el

    gripper = rtqhe.RobotiqHE()
    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e(showrbt=True)

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=True)

    bs = b_sim.BendSim(show=True)
    mp = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")

    transmat4 = rm.homomat_from_posrot((.9, -.35, .78 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    f_name = 'penta'
    # goal_pseq = bu.gen_polygen(5, .05)
    goal_pseq = bu.gen_ramdom_curve(kp_num=5, length=.12, step=.0005, z_max=.005, toggledebug=False)
    # goal_pseq = bu.gen_screw_thread(r=.02, lift_a=np.radians(5), rot_num=2)
    # goal_pseq = bu.gen_circle(.05)
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) * .4
    # goal_pseq = np.asarray([[0, 0, .1], [0, 0, 0], [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1]]) * .4
    # goal_pseq = np.asarray([[.1, 0, .1], [0, 0, .1], [0, 0, 0]]) * .4
    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    brp = BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
    grasp_list = mp.load_all_grasp('stick')
    grasp_list = grasp_list

    fit_pseq, _ = bu.decimate_pseq(goal_pseq, tor=.001, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)[::-1]
    init_rot = bu.get_init_rot(fit_pseq)
    # pickle.dump(bendset, open(f'planres/{f_name}_bendseq.pkl', 'wb'))
    # bendset = pickle.load(open(f'planres/{f_name}_bendseq.pkl', 'rb'))

    bs.reset([(0, 0, 0), (0, max(np.asarray(bendset)[:, 3]), 0)], [np.eye(3), np.eye(3)])
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    bs.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)
    base.run()

    brp.set_up(bendset, grasp_list, transmat4)
    brp.run(f_name=f_name, grasp_l=0)
