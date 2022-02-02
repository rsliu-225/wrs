import copy
import pickle
import random
import time
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import config
import graspplanner.grasp_planner as gp
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import motionplanner.animation_helper as ani_helper
import motionplanner.ik_solver as iks
import motionplanner.robot_helper as rbt_helper
import utils.graph_utils as gh
import basis.robot_math as rm
import modeling.geometric_model as gm
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp
import math
import motion.optimization_based.incremental_nik as inik


class MotionPlanner(object):
    def __init__(self, env, rbt, armname="lft_arm"):
        self.rbt = rbt
        self.env = env
        self.armname = armname
        if self.armname == 'lft_arm':
            self.hnd_name = 'lft_hnd'
            self.arm = self.rbt.lft_arm
        else:
            self.hnd_name = 'rgt_hnd'
            self.arm = self.rbt.rgt_arm

        self.obscmlist = []
        # for obscm in self.obscmlist:
        #     obscm.showcn()
        self.gripper = rtqhe.RobotiqHE()
        self.iksolver = iks.IkSolver(self.env, self.rbt, self.armname)
        # self.adp_s = adp.ADPlanner(self.rbt)
        self.inik_slvr = inik.IncrementalNIK(self.rbt)
        self.initjnts = self.rbt.get_jnt_values(self.armname)
        # print(self.initjnts)

        self.graspplanner = gp.GraspPlanner(self.gripper)
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)
        self.ah = ani_helper.AnimationHelper(self.env, self.rbt, self.armname)
        self.init_obs()

    def add_obs(self, obs):
        self.obscmlist.append(obs)

    def init_obs(self):
        self.obscmlist = self.env.getstationaryobslist() + self.env.getchangableobslist()

    def get_ee(self, armjnts=None, relmat4=np.eye(4)):
        return self.rbth.get_ee(armjnts, relmat4)

    def get_tcp(self, armjnts=None):
        return self.rbth.get_tcp(armjnts)

    def get_numik(self, eepos, eerot, msc=None):
        if msc is None:
            armjnts = self.rbt.ik(self.armname, eepos, eerot)
        else:
            armjnts = self.rbt.ik(self.armname, eepos, eerot, seed_jnt_values=msc)
        if armjnts is None:
            return None
        # self.ah.show_armjnts(armjnts=armjnts)
        if self.rbth.is_selfcollided(armjnts):
            print('Collided!')
            return None
        return armjnts

    def get_numik_nlopt(self, tgtpos, tgtrot=None, seedjntagls="default", releemat4=np.eye(4), col_ps=None,
                        roll_limit=1e-2, pos_limit=1e-2, movedir=None, toggledebug=False):
        return self.iksolver.solve_numik_nlopt(tgtpos, tgtrot, seedjntagls=seedjntagls, releemat4=releemat4,
                                               col_ps=col_ps, roll_limit=roll_limit, pos_limit=pos_limit,
                                               movedir=movedir, toggledebug=toggledebug)

    def load_grasp(self, model_name, grasp_id):
        return pickle.load(open(config.PREGRASP_REL_PATH + model_name + "_pregrasps.pkl", "rb"))[grasp_id]

    def load_all_grasp(self, model_name):
        return pickle.load(open(config.PREGRASP_REL_PATH + model_name + "_pregrasps.pkl", "rb"))

    def load_objmat4(self, model_name, objmat4_id):
        return pickle.load(open(config.GRASPMAP_REL_PATH + model_name + "_objmat4_list.pkl", "rb"))[objmat4_id]

    def load_all_objmat4(self, model_name, grasp_id):
        result = []
        graspmap = pickle.load(open(config.GRASPMAP_REL_PATH + model_name + "_graspmap.pkl", "rb"))
        objmat4_dict = graspmap[grasp_id]
        for k, v in objmat4_dict.items():
            if v:
                result.append(self.load_objmat4(model_name, k))
        return result

    def load_all_objmat4_failed(self, model_name, grasp_id):
        result = []
        graspmap = pickle.load(open(config.GRASPMAP_REL_PATH + model_name + "_graspmap.pkl", "rb"))
        objmat4_dict = graspmap[grasp_id]
        for k, v in objmat4_dict.items():
            if not v:
                result.append(self.load_objmat4(model_name, k))
        return result

    def get_armjnts_by_objmat4ngrasp(self, grasp, obslist, objmat4, msc=None):
        eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
        armjnts = self.get_numik(eepos, eerot, msc=msc)
        if armjnts is None:
            print("No ik solution")
            return None
        if (not self.rbth.is_selfcollided(armjnts=armjnts)) \
                and (not self.rbth.is_objcollided(obslist, armjnts=armjnts)):
            return armjnts
        else:
            # self.rbth.show_armjnts(armjnts=armjnts, rgba=(.7, 0, 0, .7))
            print("Collided")
        return None

    def get_available_graspid_by_objmat4(self, grasp_list, obj, objmat4):
        remaingrasp_id_list = []
        for i in range(len(grasp_list)):
            grasp = grasp_list[i]
            if self.is_grasp_available(grasp, obj, objmat4):
                remaingrasp_id_list.append(i)
        print("Num of remain grasps:", len(remaingrasp_id_list))
        return remaingrasp_id_list

    def filter_gid_by_objmat4_list(self, grasp_list, obj, objmat4_list, candidate_list=None, toggledebug=False):
        time_start = time.time()
        remaingraspid_list = []
        time_cost_dict = {}
        if candidate_list is None:
            candidate_list = range(len(grasp_list))
        for i in candidate_list:
            time_start_tmp = time.time()
            grasp = grasp_list[i]
            _, _, hndmat4 = grasp
            success = True
            for objmat4 in objmat4_list:
                if self.is_grasp_available(grasp, obj, objmat4):
                    if toggledebug:
                        self.ah.show_hnd_sgl(np.dot(objmat4, hndmat4), rgba=(0, 1, 0, .2))
                    continue
                else:
                    if toggledebug:
                        self.ah.show_hnd_sgl(np.dot(objmat4, hndmat4), rgba=(1, 0, 0, .2))
                    success = False
            if success:
                remaingraspid_list.append(i)
                print("time cost(first start/goal pose):", i, time.time() - time_start)
            time_cost_dict[i] = {'flag': success, 'time_cost': time.time() - time_start_tmp}

        print("Num of remain grasps:", len(remaingraspid_list))
        return remaingraspid_list, time_cost_dict

    def get_available_graspid_by_objmat4_list_msc(self, grasp_list, obj, objmat4_list, available_graspid_list=None,
                                                  threshold=0.9):
        time_start = time.time()
        remaingraspid_list = []
        time_cost_dict = {}
        if available_graspid_list is None:
            available_graspid_list = range(len(grasp_list))
        if objmat4_list is None:
            print("get_available_graspid_by_objmat4_list_msc, objmat4_list is None!")
            return available_graspid_list
        for i in available_graspid_list:
            time_start_tmp = time.time()
            grasp = grasp_list[i]
            success_cnt = 0
            failed_cnt = 0
            msc = None
            for objmat4 in objmat4_list:
                if failed_cnt > len(objmat4_list) * (1 - threshold):
                    break
                eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
                armjnts = self.get_numik(eepos, eerot, msc=msc)

                if armjnts is not None:
                    if (not self.rbth.is_selfcollided(armjnts=armjnts)) \
                            and (not self.rbth.is_objcollided([obj], armjnts=armjnts)):
                        msc = armjnts
                        success_cnt += 1
                        continue
                failed_cnt += 1

            if success_cnt >= len(objmat4_list) * threshold:
                remaingraspid_list.append(i)
                time_cost_dict[i] = {'flag': True, 'time_cost': time.time() - time_start_tmp}
                print("time cost(first common grasp):", i, time.time() - time_start)
            else:
                time_cost_dict[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            print("Success count:", success_cnt, "of", len(objmat4_list))

        print("Num of remain grasps:", len(remaingraspid_list))
        return remaingraspid_list, time_cost_dict

    def get_ee_by_objmat4(self, grasp, objmat4):
        _, gl_jaw_center_pos, gl_jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp
        prehndmat4 = rm.homomat_from_posrot(gl_jaw_center_pos, gl_jaw_center_rotmat)
        hndmat4 = np.dot(objmat4, prehndmat4)
        return [hndmat4[:3, 3], hndmat4[:3, :3]]

    def get_rel_posrot(self, grasp, objpos, objrot):
        eepos, eerot = self.get_ee_by_objmat4(grasp, rm.homomat_from_posrot(objpos, objrot))
        armjnts = self.get_numik(eepos, eerot)
        if armjnts is None:
            return None, None
        self.rbth.goto_armjnts(armjnts)
        return self.arm.cvt_gl_to_loc_tcp(objpos, objrot)

    def get_tool_primitive_armjnts(self, armjnts, objrelrot, tool_direction=np.array([-1, 0, 0]), length=50):
        eepos, eerot = self.get_ee(armjnts)
        direction = np.dot(np.dot(eerot, objrelrot), -tool_direction)
        eepos = eepos + np.dot(direction, length)
        return self.get_numik(eepos, eerot, msc=armjnts)

    def plan_start2end(self, end, start=None, additional_obscmlist=[]):
        if start is None:
            start = self.initjnts

        print("--------------start2end(rrt)---------------")
        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=start, goal_conf=end,
                            obstacle_list=self.obscmlist + additional_obscmlist, ext_dist=.02, max_time=300)

        if path is None:
            print("rrt failed!")

        return path

    def plan_start2end_hold(self, grasp, objmat4_pair, obj, start=None, use_msc=True, additional_obscmlist=[]):
        start_grasp = self.get_ee_by_objmat4(grasp, objmat4_pair[0])
        end_grasp = self.get_ee_by_objmat4(grasp, objmat4_pair[1])
        if start is None:
            start = self.get_numik(start_grasp[0], start_grasp[1])
        if start is None:
            print("Cannot reach init position!")
            return None
        objcm_hold = obj.copy()
        objcm_hold.set_homomat(objmat4_pair[0])
        _, _ = self.rbt.hold(objcm_hold, hnd_name=self.hnd_name, jaw_width=.02)

        if use_msc:
            goal = self.get_numik(end_grasp[0], end_grasp[1], msc=start)
        else:
            goal = self.get_numik(end_grasp[0], end_grasp[1])
        if goal is None:
            print("Cannot reach final position!")
            return None

        print("--------------rrt---------------")
        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=start, goal_conf=goal,
                            obstacle_list=self.obscmlist + additional_obscmlist, ext_dist=.02, max_time=300)

        if path is None:
            print("rrt failed!")
        return None

    def plan_start2end_hold_armj(self, armj_pair, obj, objrelpos, objrelrot):
        start = armj_pair[0]
        goal = armj_pair[1]
        self.rbt.fk(component_name=self.armname, jnt_values=start)
        objcm_hold = obj.copy()
        objcm_hold.set_homomat(self.get_world_objmat4(objrelpos, objrelrot, start))
        _, _ = self.rbt.hold(objcm_hold, hnd_name=self.hnd_name, jaw_width=.02)

        print("--------------rrt---------------")
        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=start, goal_conf=goal,
                            obstacle_list=self.obscmlist, ext_dist=.02, max_time=300)
        self.rbt.release(self.hnd_name, objcm_hold)
        if path is None:
            print("rrt failed!")
        return None

    def plan_gotopick(self, grasp, objmat4_pick, obj, start=None):
        eepos_initial, eerot_initial = self.get_ee_by_objmat4(grasp, objmat4_pick)
        pick_start = self.get_numik(eepos_initial, eerot_initial)
        # planning
        if pick_start is None:
            return None
        pickupprim = self.get_linear_path_to(pick_start, direction=np.asarray([0, 0, 1]), length=.1)
        if pickupprim == []:
            print("Cannot reach pick up primitive position!")
            return None

        if start is None:
            start = self.initjnts
        else:
            self.rbth.goto_armjnts(start)

        print("--------------rrt---------------")
        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=start, goal_conf=pickupprim[0],
                            obstacle_list=[self.obscmlist, obj], ext_dist=.02, max_time=300)

        if path is not None:
            path = path + pickupprim
        else:
            print("rrt failed!")

        return path

    def plan_picknplace(self, grasp, objmat4_pair, obj, use_msc=True, use_pickupprim=True,
                        use_placedownprim=True, start=None, goal=None, pickupprim_len=.1, placedownprim_len=.1):
        eepos_initial, eerot_initial = self.get_ee_by_objmat4(grasp, objmat4_pair[0])
        if start is None:
            start = self.get_numik(eepos_initial, eerot_initial)
        if start is None:
            print("Cannot reach init position!")
            return None
        self.rbt.fk(component_name=self.armname, jnt_values=start)
        obj_hold = obj.copy()
        obj_hold.set_homomat(objmat4_pair[0])
        _, _ = self.rbt.hold(obj_hold, hnd_name=self.hnd_name, jaw_width=.02)
        eepos_final, eerot_final = self.get_ee_by_objmat4(grasp, objmat4_pair[1])
        if goal is None:
            if use_msc:
                goal = self.get_numik(eepos_final, eerot_final, msc=start)
                if goal is None:
                    print("get_picknplace_goal msc failed!")
                    goal = self.get_numik(eepos_final, eerot_final)
            else:
                goal = self.get_numik(eepos_final, eerot_final, msc=start)

        if goal is None:
            print("Cannot reach final position!")
            self.rbt.release(hnd_name=self.hnd_name, objcm=obj_hold)
            return None

        # planning
        if use_pickupprim:
            pickupprim = self.get_linear_path_from(start=start, length=pickupprim_len)
            if pickupprim == []:
                print("Cannot reach init primitive position!")
                self.rbt.release(hnd_name=self.hnd_name, objcm=obj_hold)
                return None
        else:
            pickupprim = [start]

        if use_placedownprim:
            placedownprim = self.get_linear_path_from(start=goal, length=placedownprim_len)[::-1]
            if placedownprim == []:
                print("Cannot reach final primitive position!")
                self.rbt.release(hnd_name=self.hnd_name, objcm=obj_hold)
                return None
        else:
            placedownprim = [goal]

        # self.ah.show_armjnts(armjnts=pickupprim[-1], rgba=[1, 0, 1, .5])
        # self.ah.show_armjnts(armjnts=pickupprim[0], rgba=[1, 0, 0, .5])
        # self.ah.show_armjnts(armjnts=placedownprim[0], rgba=[0, 1, 1, .5])
        # self.ah.show_armjnts(armjnts=placedownprim[-1], rgba=[0, 0, 1, .5])
        # base.run()
        print("--------------rrt---------------")
        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=pickupprim[-1], goal_conf=placedownprim[0],
                            obstacle_list=self.obscmlist, ext_dist=.1, max_time=500)
        if path is not None:
            path = pickupprim + path + placedownprim
            self.rbt.release(hnd_name=self.hnd_name, objcm=obj_hold)
            return path
        else:
            print("rrt failed!")

        self.rbt.release(hnd_name=self.hnd_name, objcm=obj_hold)
        return None

    def objmat4_list_inp(self, objmat4_list, max_inp=30):
        inp_mat4_list = []
        for i, objmat4 in enumerate(objmat4_list):
            if i > 0:
                inp_mat4_list.append(objmat4_list[i - 1])
                _, angle = rm.axangle_between_rotmat(objmat4_list[i - 1][:3, :3], objmat4[:3, :3])
                if angle < 1.0:
                    continue
                cnt = int(angle) if int(angle) < max_inp else max_inp
                times = [1 / cnt * n for n in range(1, cnt)]
                # print(angle, cnt, times)

                p1 = objmat4_list[i - 1][:3, 3]
                p2 = objmat4[:3, 3]

                rots = Rotation.from_matrix([objmat4_list[i - 1][:3, :3], objmat4[:3, :3]])
                slerp = Slerp([0, 1], rots)

                interp_r_list = slerp(times)
                interp_p_list = [p1 + (p2 - p1) * t for t in times]
                interp_rot_list = [interp_r.as_matrix() for interp_r in interp_r_list]

                inp_mat4_list.extend([rm.homomat_from_posrot(p, rot) for p, rot in zip(interp_p_list, interp_rot_list)])
        print("length of interpolation result:", len(inp_mat4_list))

        return inp_mat4_list

    def objmat4_list_inp_ms(self, objmat4_list_ms, max_inp=30):
        inp_mat4_list_ms = []
        for objmat4_list in objmat4_list_ms:
            inp_mat4_list_ms.append(self.objmat4_list_inp(objmat4_list, max_inp=max_inp))
        return inp_mat4_list_ms

    def get_continuouspath_ik(self, msc, grasp, objmat4_list, grasp_id=0, threshold=1, toggledebug=False,
                              dump_f_name=None):
        """
        plan init armjnts to first drawpath armjnts, append all the others armjnts in the drawpath.

        :param graspseq:
        :param obj:
        :param objrelpos:
        :param objrelrot:
        :param threshold: 0-1
        :return: armjnts list
        """
        print(f"--------------get continuous path(ik) {grasp_id}---------------")
        success_cnt = 0
        path = []
        # init_msc = msc
        time_start_1 = time.time()
        relpos, relrot = None, None
        for i, objmat4 in enumerate(objmat4_list):
            eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
            tgtpos = objmat4[:3, 3]
            tgtrot = objmat4[:3, :3]
            if relpos is None:
                relpos, relrot = rm.rel_pose(eepos, eerot, tgtpos, tgtrot)
            armjnts = self.get_numik(eepos, eerot, msc=msc)

            if armjnts is not None:
                path.append(armjnts)
                msc = copy.deepcopy(armjnts)
                success_cnt += 1
            # else:
            #     print('get draw path failed!', i)
            #     return None

        print("Success point:", success_cnt, "of", len(objmat4_list))
        if toggledebug:
            if success_cnt > 0:
                cost = self.get_path_cost(path)
                fig = plt.figure(1, figsize=(6.4, 4.8))
                plt.ion()
                self.rbth.plot_armjnts(path)
                plt.show()
                if dump_f_name is not None:
                    f_name = f"./log/path/ik/{dump_f_name}_" \
                             f"{grasp_id}_{round(cost, 3)}_{round(time.time() - time_start_1, 3)}_{len(path)}"
                    plt.savefig(f"{f_name}.png")
                    plt.close(fig)
                    pickle.dump([relpos, relrot, path], open(f"{f_name}.pkl", "wb"))
                return path
        if success_cnt >= len(objmat4_list) * threshold:
            return path

        return None

    def get_draw_sconfig(self, objmat4, grasp):
        # sample_range = self.get_rpy_list((-180, 180, 10), (0, 0, 0), (0, 0, 0))
        sample_range = self.get_rpy_list((-10, 11, 1), (0, 0, 0), (0, 0, 0))
        msc, start_id = self.__find_start_config(objmat4, grasp, sample_range)
        print(start_id, msc)
        # self.rbth.show_armjnts(armjnts=msc)
        return msc

    def get_continuouspath_nlopt(self, msc, grasp, objmat4_list, grasp_id=0, threshold=1, col_ps=None, roll_limit=1e-2,
                                 pos_limit=1e-2, add_mvcon=False, dump_f_name=None, toggledebug=False):
        """
        plan init armjnts to first drawpath armjnts, append all the others armjnts in the drawpath.

        :param graspseq:
        :param obj:
        :param objrelpos:
        :param objrelrot:
        :param threshold: 0-1
        :return: armjnts list
        """
        print(f"--------------get continuous path(nlopt) {grasp_id}---------------")
        success_cnt = 0
        path = []
        time_start_1 = time.time()
        relpos, relrot = None, None

        if msc is None:
            return None

        for inx, objmat4 in enumerate(objmat4_list):
            if add_mvcon:
                if inx == 0:
                    movedir = None
                else:
                    movedir = np.asarray(objmat4_list[inx - 1][:3, 3] - objmat4[:3, 3])
            else:
                movedir = None
            eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
            if relpos is None:
                relpos, relrot = rm.rel_pose(eepos, eerot, objmat4[:3, 3], objmat4[:3, :3])
            armjnts = self.get_numik_nlopt(objmat4[:3, 3], objmat4[:3, :3], seedjntagls=msc, toggledebug=toggledebug,
                                           releemat4=rm.homomat_from_posrot(relpos, relrot), col_ps=col_ps,
                                           roll_limit=roll_limit, pos_limit=pos_limit, movedir=movedir)

            if armjnts is not None:
                path.append(armjnts)
                msc = copy.deepcopy(armjnts)
                success_cnt += 1
                if toggledebug:
                    eepos, eerot = self.rbth.get_ee(armjnts, releemat4=rm.homomat_from_posrot(relpos, relrot))
                    self.rbth.draw_axis(eepos, eerot, length=100)
                    self.ah.show_armjnts(armjnts=armjnts, rgba=(0, 1, 0, .5))
                    axmat = self.rbt.manipulability_axmat(component_name=self.armname)
                    manipulability = self.rbt.manipulability(component_name=self.armname)
                    print("%e" % manipulability)
                    self.rbth.draw_axis_uneven(objmat4[:3, 3], axmat, scale=.5)
                    base.run()
            else:
                return None

        print("Success point:", success_cnt, "of", len(objmat4_list))
        if success_cnt >= len(objmat4_list) * threshold:
            cost = self.get_path_cost(path)
            if dump_f_name is not None:
                f_name = f"./log/path/nlopt/{dump_f_name}_" \
                         f"{grasp_id}_{round(cost, 3)}_{round(time.time() - time_start_1, 3)}_{len(path)}"
                pickle.dump([relpos, relrot, path], open(f"{f_name}.pkl", "wb"))
            return path

        return None

    def get_rpy_list(self, r_range, p_range, y_range):
        rpy_list = []
        r_range = range(r_range[0], r_range[1], r_range[2]) if r_range[2] != 0 else [0]
        p_range = range(p_range[0], p_range[1], p_range[2]) if p_range[2] != 0 else [0]
        y_range = range(y_range[0], y_range[1], y_range[2]) if y_range[2] != 0 else [0]
        for r in r_range:
            for p in p_range:
                for y in y_range:
                    rpy_list.append((r, p, y))
        return rpy_list

    def __find_start_config(self, objmat4, grasp, sample_range, toggledebug=False):
        time_start = time.time()
        best_config = None
        best_score = 0
        best_inx = 0
        if toggledebug:
            best_ellipsoid = np.eye(3)
            best_objmat4 = objmat4
        print(objmat4)
        for inx, v in enumerate(sample_range):
            r, p, y = v
            rot = np.dot(rm.rotmat_from_axangle(objmat4[:3, 0], r),
                         np.dot(rm.rotmat_from_axangle(objmat4[:3, 2], p),
                                rm.rotmat_from_axangle(objmat4[:3, 1], y)))
            objmat4_new = np.eye(4)
            objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
            objmat4_new[:3, 3] = objmat4[:3, 3]
            eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4_new)
            armjnts = self.get_numik(eepos, eerot)
            # relpos, relrot = rm.rel_pose(eepos, eerot, objmat4_new[:3, 3], objmat4_new[:3, :3])
            if armjnts is not None:
                score = self.rbt.manipulability(component_name=self.armname)
                if toggledebug:
                    axmat = self.rbt.manipulability_axmat(component_name=self.armname)
                    self.ah.show_armjnts(armjnts=armjnts, rgba=(1, 1, 0, .2))
                    # self.rbth.draw_axis(objmat4_new[:3, 3], objmat4_new[:3, :3], rgba=(1, 1, 0, .5))
                    # self.rbth.draw_axis_uneven(objmat4_new[:3, 3], axmat,scale=.5)
                    # print(score)

                if score > best_score:
                    best_config = armjnts
                    best_score = score
                    best_inx = inx
                    if toggledebug:
                        best_ellipsoid = axmat
                        best_objmat4 = objmat4_new
        if toggledebug:
            self.ah.show_armjnts(armjnts=best_config, rgba=(0, 1, 0, .5))
            self.rbth.draw_axis_uneven(best_objmat4[:3, 3], best_ellipsoid, scale=.2)
            print(("%e" % best_score))
            base.run()

        print("find start configuration time cost:", time.time() - time_start)

        return best_config, best_inx

    def get_continuouspath_opt1(self, grasp, grasp_id, objmat4_list, sample_range, msc=None):
        msc, start_id = self.__find_start_config(objmat4_list[0], grasp, sample_range)
        print(msc, start_id)
        init = copy.deepcopy(msc)
        if msc is None:
            return None
        time_start_1 = time.time()
        gtsp_dict = {}
        end = len(objmat4_list) - 1
        relpos = None
        relrot = None

        for key, objmat4 in enumerate(objmat4_list):
            gtsp_dict[key] = {"main": objmat4, "objmat4_list": []}
            gm.gen_arrow(spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                         rgba=(1, 0, 0, 0.2), thickness=1)
            cnt = 0
            for r, p, y in sample_range:
                cnt += 1
                rot = np.dot(rm.rotmat_from_axangle(objmat4[:3, 0], r),
                             np.dot(rm.rotmat_from_axangle(objmat4[:3, 2], p),
                                    rm.rotmat_from_axangle(objmat4[:3, 1], y)))
                objmat4_new = np.eye(4)
                objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
                objmat4_new[:3, 3] = objmat4[:3, 3]
                gtsp_dict[key]["objmat4_list"].append(objmat4_new)
                # gm.gen_arrow(spos=objmat4_new[:3, 3],
                #                      epos=objmat4_new[:3, 3] + 10 * objmat4_new[:3, 0],
                #                      rgba=(1, 1, 0, 0.2), thickness=1)

        for k, v in gtsp_dict.items():
            armjnts_list = []
            node_success = False
            for i, objmat4 in enumerate(v["objmat4_list"]):
                eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
                if relpos is None and relrot is None:
                    relpos, relrot = rm.rel_pose(eepos, eerot, objmat4[:3, 3], objmat4[:3, :3])
                armjnts = self.get_numik(eepos, eerot, msc=msc)
                if armjnts is not None:
                    armjnts_list.append(armjnts)
                    if np.linalg.norm(armjnts - msc) < 50 and np.linalg.norm(armjnts - init) < 300:
                        msc = copy.deepcopy(armjnts)
                    node_success = True
                else:
                    # print(k, i, objmat4)
                    armjnts_list.append(armjnts)
                    armjnts_list.append(armjnts)
            if not node_success:
                # print(k, "node failed!", armjnts_list)
                return None
            print(k, f"{len([v for v in armjnts_list if v is not None])} of {len(sample_range)}")
            gtsp_dict[k]["armjnts_list"] = armjnts_list

        G = nx.DiGraph()
        N = len(sample_range)
        for k, v in list(gtsp_dict.items())[:-1]:
            for i in range(N):
                for j in range(N):
                    armjnts_1 = gtsp_dict[k]["armjnts_list"][i]
                    armjnts_2 = gtsp_dict[k + 1]["armjnts_list"][j]
                    if armjnts_1 is None or armjnts_2 is None:
                        continue
                    diff = np.linalg.norm(armjnts_2 - armjnts_1, ord=1)
                    G.add_weighted_edges_from([(f"{str(k)}_{str(i)}", f"{str(k + 1)}_{str(j)}", diff)])
        print("number of nodes:", len(G))
        print("build graph time cost:", time.time() - time_start_1)

        time_start_2 = time.time()
        best_dist = np.inf
        best_path = None
        # for i in range(len(gtsp_dict[0]["objmat4_list"])):
        #     dist_dict, path_dict = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=f"0_{i}")
        if G.has_node(f"0_{start_id}"):
            dist_dict, path_dict = \
                nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=f"0_{start_id}")
        else:
            source_list = [n for n in list(G.nodes) if str(n)[:2] == "0_"]
            if source_list is []:
                return None
            dist_dict, path_dict = \
                nx.algorithms.shortest_paths.weighted.single_source_dijkstra(G, source=source_list[0])
        if path_dict is None:
            return None

        target_list = [n for n in list(G.nodes) if str(n).split("_")[0] == str(end)]
        for target in target_list:
            dist = dist_dict[target]
            path = path_dict[target]
            if dist < best_dist:
                best_dist = dist
                best_path = path

        print("search time cost:", time.time() - time_start_2)
        print("min cost:", best_dist)
        print("best path:", best_path)
        armjnts_path = []
        for node in best_path:
            k, objmat4_id = node.split("_")
            objmat4 = gtsp_dict[int(k)]["objmat4_list"][int(objmat4_id)]
            # self.show_armjnts(rgba=(0, 1, 0, 0.5), armjnts=gtsp_dict[int(k)]["armjnts_list"][int(objmat4_id)])
            armjnts_path.append(gtsp_dict[int(k)]["armjnts_list"][int(objmat4_id)])
            gm.gen_arrow(spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                         rgba=(0, 1, 0, 1), thickness=1)

        fig = plt.figure(1, figsize=(12.8, 4.8))
        plt.ion()
        plt.subplot(121)
        self.rbth.plot_nodepath(best_path, title="node path")
        plt.subplot(122)
        self.rbth.plot_armjnts(armjnts_path)
        plt.show()
        f_name = f"./log/path/discrete/" \
                 f"m1w5_ik3_{grasp_id}_{round(best_dist, 3)}_{round(time.time() - time_start_1, 3)}_{len(G)}"
        plt.savefig(f"{f_name}.png")
        plt.close(fig)
        pickle.dump([relpos, relrot, armjnts_path], open(f"{f_name}.pkl", "wb"))
        # self.show_ani(armjnts_path)
        # base.run()
        return armjnts_path

    def goto_posrot(self, eepos, eerot):
        # armjnts = self.rbt.numik(eepos, eerot, self.armname)
        armjnts = self.get_numik(eepos, eerot)
        if armjnts is not None:
            self.rbth.goto_armjnts(armjnts)
            return armjnts
        else:
            print("No IK solution for the given pos, rot.")
            return None

    def goto_posrot_msc(self, eepos, eerot, msc):
        armjnts = self.get_numik(eepos, eerot, msc=msc)
        if armjnts is not None:
            self.rbth.goto_armjnts(armjnts)
            return armjnts
        else:
            print("No IK solution for the given pos, rot.")
            return None

    def get_world_objmat4(self, objrelpos, objrelrot, armjnts=None):
        if armjnts is not None:
            self.rbth.goto_armjnts(armjnts)
        # objpos, objrot = self.rbt.getee(armname=self.armname)
        objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        objmat4 = rm.homomat_from_posrot(objpos, objrot)
        return objmat4

    def get_linear_path_from(self, start, direction=np.asarray([0, 0, 1]), length=.05):
        path = self.inik_slvr.gen_rel_linear_motion_with_given_conf(self.armname, start,
                                                                    direction=direction,
                                                                    distance=length,
                                                                    obstacle_list=[],
                                                                    granularity=0.01,
                                                                    seed_jnt_values=None,
                                                                    type='source',
                                                                    toggle_debug=False)
        if path is not None:
            return path
        else:
            return []

    def get_linear_path_to(self, goal, direction=np.asarray([0, 0, 1]), length=.05):
        path = self.inik_slvr.gen_rel_linear_motion_with_given_conf(self.armname, goal,
                                                                    direction=-direction,
                                                                    distance=length,
                                                                    obstacle_list=[],
                                                                    granularity=0.01,
                                                                    seed_jnt_values=None,
                                                                    type='source',
                                                                    toggle_debug=False)
        if path is not None:
            return path[::-1]
        else:
            return []

    def homomat2vec(self, objrelpos, objrelrot):
        rotation = Rotation.from_dcm(objrelrot)
        return objrelpos.tolist() + rotation.as_rotvec().tolist()

    def refine_relpose_by_transmat(self, objrelpos, objrelrot, transmat):
        objmat4 = rm.homomat_from_posrot(objrelpos, objrelrot)
        objmat4_new = np.dot(objmat4, transmat)
        # objrelrot = np.dot(objrelrot, transmat[:3, :3])
        # objrelpos = np.dot(objrelrot, transmat[:3, 3]) + objrelpos
        return objmat4_new[:3, 3], objmat4_new[:3, :3]

    def get_path_cost(self, path):
        cost = 0
        for i in range(0, len(path) - 1):
            cost += np.linalg.norm(np.asarray(path[i + 1]) - np.asarray(path[i]), ord=1)
        return cost

    def is_grasp_available(self, grasp, obj, objmat4):
        eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
        armjnts = self.get_numik(eepos, eerot)
        obj.set_homomat(objmat4)
        if armjnts is not None:
            return True
        return False


if __name__ == '__main__':
    import numpy as np

    import utils.phoxi as phoxi
    import utils.phoxi_locator as pl
    import basis.robot_math as rm
    from utils.run_script_utils import *

    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()
    # rbtx = el.loadUr3ex(rbt)
    # rbtx.lft_arm_hnd.open_gripper()
    # rbtx.rgt_arm_hnd.open_gripper()

    pen = el.loadObj(config.PEN_STL_F_NAME)

    '''
    init class
    '''
    mp_rgt = MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = MotionPlanner(env, rbt, armname="lft_arm")

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    # mp_lft.ah.show_armjnts(toggleendcoord=True)

    # mp_x_lft.goto_init_x()
    # mp_x_rgt.goto_init_x()
    import math

    glist = mp_lft.load_all_grasp(config.PEN_STL_F_NAME.split('.stl')[0])
    objmat4_init = rm.homomat_from_posrot(np.asarray([.9, .4, .9]), rm.rotmat_from_axangle((0, 1, 0), -math.pi / 4))
    objmat4_goal = rm.homomat_from_posrot(np.asarray([.9, .3, .9]), rm.rotmat_from_axangle((0, 1, 0), -math.pi / 4))

    mp_lft.ah.show_objmat4(pen, objmat4_init, rgba=(1, 0, 1, .5), showlocalframe=True)
    mp_lft.ah.show_objmat4(pen, objmat4_goal, rgba=(0, 1, 1, .5), showlocalframe=True)

    gripper = rtqhe.RobotiqHE()
    for i, grasp in enumerate(glist):
        print(f'-----------{i}-------------')
        path = mp_lft.plan_picknplace(grasp, [objmat4_init, objmat4_goal], pen)
        if path is not None:
            mp_lft.rbt.fk(component_name=mp_lft.armname, jnt_values=path[0])
            obj_hold = pen.copy()
            obj_hold.set_homomat(objmat4_init)
            _, _ = mp_lft.rbt.hold(obj_hold, hnd_name=mp_lft.hnd_name, jaw_width=.02)
            mp_lft.ah.show_ani(path)
            base.run()
        # eepos_initial, eerot_initial = mp_lft.get_ee_by_objmat4(grasp, objmat4_init)
        # start = mp_lft.get_numik(eepos_initial, eerot_initial)
        # eepos_final, eerot_final = mp_lft.get_ee_by_objmat4(grasp, objmat4_goal)
        # goal = mp_lft.get_numik(eepos_final, eerot_final, msc=start)
        # if start is not None and goal is not None:
        #     path = mp_lft.plan_start2end(end=goal, start=start)
        #     if path is not None:
        #         mp_lft.ah.show_ani(path)
        #         base.run()
