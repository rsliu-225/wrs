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
import manipulation.grip.robotiqhe.robotiqhe as rtqhe
import motionplanner.animation_helper as ani_helper
import motionplanner.ik_solver as iks
import motionplanner.robot_helper as rbt_helper
import utils.graph_utils as gh
import basis.robot_math as rm
import motion.probabilistic.rrt_connect as rrtc


class MotionPlanner(object):
    def __init__(self, env, rbt, armname="lft_arm"):
        self.rbt = rbt
        self.env = env
        self.armname = armname

        self.obscmlist = env.getstationaryobslist() + env.getchangableobslist()
        # for obscm in self.obscmlist:
        #     obscm.showcn()
        self.hndfa = rtqhe.HandFactory()
        self.iksolver = iks.IkSolver(self.env, self.rbt, self.armname)

        if self.armname == "lft_arm":
            self.initjnts = self.rbt.initlftjnts
            self.arm = self.rbt.lft_arm
        else:
            self.initjnts = self.rbt.initrgtjnts
            self.arm = self.rbt.rgt_arm

        self.graspplanner = gp.GraspPlanner(self.hndfa)
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)
        self.ah = ani_helper.AnimationHelper(self.env, self.rbt,  self.armname)

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
            # armjnts = self.rbt.numik(eepos, eerot, armname=self.armname)
            armjnts = self.iksolver.solve_numik3(eepos, eerot)

        else:
            # armjnts = self.rbt.numik(eepos, eerot, seedjntagls=msc, armname=self.armname)
            armjnts = self.iksolver.solve_numik3(eepos, eerot, seedjntagls=msc)
        if armjnts is None:
            return None
        if not self.rbth.is_selfcollided(armjnts):
            return armjnts
        return None

    def get_numik_nlopt(self, tgtpos, tgtrot=None, seedjntagls="default", releemat4=np.eye(4), col_ps=None,
                        roll_limit=1e-2, pos_limit=1e-2, movedir=None, toggledebug=False):
        return self.iksolver.solve_numik4(tgtpos, tgtrot, seedjntagls=seedjntagls, releemat4=releemat4,
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

    def refine_grasp(self, grasp, transmat):
        prejawwidth, prehndfc, prehndmat4 = grasp
        prehndmat4 = np.dot(transmat, prehndmat4)
        prehndfc = rm.homotransformpoint(transmat, prehndfc)
        return [prejawwidth, prehndfc, prehndmat4]

    def get_armjnts_by_objmat4ngrasp(self, grasp, obj, objmat4, msc=None):
        prejawwidth, prehndfc, prehndmat4 = grasp
        hndmat4 = np.dot(objmat4, prehndmat4)
        eepos = rm.homotransformpoint(objmat4, prehndfc)[:3]
        eerot = hndmat4[:3, :3]
        armjnts = self.get_numik(eepos, eerot, msc=msc)

        if armjnts is None:
            print("No ik solution")
            return None
        if (not self.rbth.is_selfcollided(armjnts=armjnts)) and (not self.rbth.is_objcollided(obj, armjnts=armjnts)):
            return armjnts
        else:
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
                prejawwidth, prehndfc, prehndmat4 = grasp
                hndmat4 = np.dot(objmat4, prehndmat4)
                eepos = rm.homotransformpoint(objmat4, prehndfc)[:3]
                eerot = hndmat4[:3, :3]
                armjnts = self.get_numik(eepos, eerot, msc=msc)

                if armjnts is not None:
                    if (not self.rbth.is_selfcollided(armjnts=armjnts)) \
                            and (not self.rbth.is_objcollided(obj, armjnts=armjnts)):
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
        _, prehndfc, prehndmat4 = grasp
        hndmat4 = np.dot(objmat4, prehndmat4)
        eepos = rm.homotransformpoint(objmat4, prehndfc)[:3]
        eerot = hndmat4[:3, :3]
        return [eepos, eerot]

    def get_eeseq_by_objmat4seq(self, grasp, objmat4_list):
        graspseq = []
        for objmat4 in objmat4_list:
            graspseq.append(self.get_ee_by_objmat4(grasp, objmat4))
        return graspseq

    def get_rel_posrot(self, grasp, objpos, objrot):
        eepos, eerot = self.get_ee_by_objmat4(grasp, rm.homobuild(objpos, objrot))
        armjnts = self.get_numik(eepos, eerot)
        if armjnts is None:
            return None, None
        self.rbth.goto_armjnts(armjnts)
        return self.rbt.getinhandpose(objpos, objrot, self.armname)

    def get_tool_primitive_armjnts(self, armjnts, objrelrot, tool_direction=np.array([-1, 0, 0]), length=50):
        eepos, eerot = self.get_ee(armjnts)
        direction = np.dot(np.dot(eerot, objrelrot), -tool_direction)
        eepos = eepos + np.dot(direction, length)
        return self.get_numik(eepos, eerot, msc=armjnts)

    def plan_start2end(self, end, start=None, additional_obscmlist=[]):
        if start is None:
            if self.armname == "lft_arm":
                start = self.rbt.initlftjnts
            else:
                start = self.rbt.initrgtjnts

        print("--------------start2end(rrt)---------------")
        # planner = rrtc.RRTConnect(start=start, goal=end, checker=self.ctcallback,
        #                           starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=200,
        #                           maxtime=100.0)
        # path, _ = planner.planning(self.obscmlist + additional_obscmlist)

        planner = rrtc.RRTConnect(self.rbt)
        path = planner.plan(component_name=self.armname, start_conf=start, goal_conf=end,
                            obstacle_list=self.obscmlist + additional_obscmlist, ext_dist=.2, max_time=300)

        if path is None:
            print("rrt failed!")

        return path

    def plan_start2end_hold(self, grasp, objmat4_pair, obj, objrelpos, objrelrot, start=None, use_msc=True):
        start_grasp = self.get_ee_by_objmat4(grasp, objmat4_pair[0])
        end_grasp = self.get_ee_by_objmat4(grasp, objmat4_pair[1])
        if start is None:
            start = self.get_numik(start_grasp[0], start_grasp[1])
        if start is None:
            print("Cannot reach init position!")
            return None

        if use_msc:
            goal = self.get_numik(end_grasp[0], end_grasp[1], msc=start)
        else:
            goal = self.get_numik(end_grasp[0], end_grasp[1])
        if goal is None:
            print("Cannot reach final position!")
            return None

        print("--------------rrt---------------")
        planner = \
            rrtc.RRTConnect(start=start, goal=goal, checker=self.ctcallback,
                            starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=200, maxtime=100.0)
        path, samples = planner.planninghold([obj], [[objrelpos, objrelrot]], self.obscmlist)

        if path is None:
            print("rrt failed!")
        return None

    def plan_start2end_hold_armj(self, armj_pair, obj, objrelpos, objrelrot):
        start = armj_pair[0]
        goal = armj_pair[1]
        print("--------------rrt---------------")
        planner = \
            rrtc.RRTConnect(start=start, goal=goal, checker=self.ctcallback,
                            starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=200, maxtime=100.0)
        path, samples = planner.planninghold([obj], [[objrelpos, objrelrot]], self.obscmlist)

        if path is None:
            print("rrt failed!")
        return None

    def plan_gotopick(self, grasp, objmat4_pick, obj, objrelpos, objrelrot, start=None):
        eepos_initial, eerot_initial = self.get_ee_by_objmat4(grasp, objmat4_pick)
        pick_start = self.get_numik(eepos_initial, eerot_initial)
        # planning
        if pick_start is None:
            return None
        pickupprim = self.ctcallback.getLinearPrimitive(pick_start, [0, 0, 1], 100, [obj], [[objrelpos, objrelrot]], [],
                                                        type="sink")
        if pickupprim == []:
            print("Cannot reach pick up primitive position!")
            return None

        if start is None:
            if self.armname == "lft_arm":
                start = self.rbt.initlftjnts
            else:
                start = self.rbt.initrgtjnts
        else:
            self.rbth.goto_armjnts(start)

        print("--------------rrt---------------")
        planner = rrtc.RRTConnect(start=start, goal=pickupprim[0], checker=self.ctcallback,
                                  starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=200,
                                  maxtime=100.0)
        path, _ = planner.planning(self.obscmlist)

        if path is not None:
            path = path + pickupprim
        else:
            print("rrt failed!")

        return path

    def plan_picknplace(self, grasp, objmat4_pair, obj, objrelpos, objrelrot, use_msc=True, use_pickupprim=True,
                        use_placedownprim=True, start=None, pickupprim_len=100, placedownprim_len=70):
        eepos_initial, eerot_initial = self.get_ee_by_objmat4(grasp, objmat4_pair[0])
        if start is None:
            start = self.get_numik(eepos_initial, eerot_initial)

        if start is None:
            print("Cannot reach init position!")
            return None

        eepos_final, eerot_final = self.get_ee_by_objmat4(grasp, objmat4_pair[1])
        if use_msc:
            goal = self.get_numik(eepos_final, eerot_final, msc=start)

            if goal is None:
                print("get_picknplace_goal msc failed!")
                goal = self.get_numik(eepos_final, eerot_final)

        else:
            goal = self.get_numik(eepos_final, eerot_final, msc=start)

        if goal is None:
            print("Cannot reach final position!")
            return None

        # planning
        if use_pickupprim:
            pickupprim = self.ctcallback.getLinearPrimitive(start, [0, 0, 1], pickupprim_len, [obj],
                                                            [[objrelpos, objrelrot]], [], type="source")
            if pickupprim == []:
                print("Cannot reach init primitive position!")
                return None
        else:
            pickupprim = [start]

        if use_placedownprim:
            placedownprim = self.ctcallback.getLinearPrimitive(goal, [0, 0, 1], placedownprim_len, [obj],
                                                               [[objrelpos, objrelrot]], [], type="sink")
            if placedownprim == []:
                print("Cannot reach final primitive position!")
                return None
        else:
            placedownprim = [goal]

        # self.ah.show_armjnts(armjnts=pickupprim[-1], rgba=[1.0, 0.0, 1.0, .5])
        # self.ah.show_armjnts(armjnts=pickupprim[0], rgba=[1.0, 0.0, 0.0, .5])
        # objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        # objmat4 = rm.homobuild(objpos, objrot)
        # obj = copy.deepcopy(obj)
        # obj.setColor(1.0, 0.0, 1.0, .5)
        # obj.setMat(base.pg.np4ToMat4(objmat4))
        #
        # self.ah.show_armjnts(armjnts=placedownprim[0], rgba=[0.0, 1.0, 1.0, .5])
        # self.ah.show_armjnts(armjnts=placedownprim[-1], rgba=[0.0, 0.0, 1.0, .5])
        # objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        # objmat4 = rm.homobuild(objpos, objrot)
        # obj = copy.deepcopy(obj)
        # obj.setColor(0.0, 1.0, 1.0, .5)
        # obj.setMat(base.pg.np4ToMat4(objmat4))
        # base.run()

        print("--------------rrt---------------")
        planner = \
            rrtc.RRTConnect(start=pickupprim[-1], goal=placedownprim[0], checker=self.ctcallback,
                            starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=200, maxtime=100.0)
        path, samples = planner.planninghold([obj], [[objrelpos, objrelrot]], self.obscmlist)

        if path is not None:
            path = pickupprim + path + placedownprim
            return path
        else:
            print("rrt failed!")

        return None

    def objmat4_list_inp(self, objmat4_list, max_inp=30):
        inp_mat4_list = []
        for i, objmat4 in enumerate(objmat4_list):
            if i > 0:
                inp_mat4_list.append(objmat4_list[i - 1])
                angle, _ = rm.degree_betweenrotmat(objmat4_list[i - 1][:3, :3], objmat4[:3, :3])
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

                inp_mat4_list.extend([rm.homobuild(p, rot) for p, rot in zip(interp_p_list, interp_rot_list)])
        print("length of interpolation result:", len(inp_mat4_list))

        return inp_mat4_list

    def objmat4_list_inp_ms(self, objmat4_list_ms):
        inp_mat4_list_ms = []
        for objmat4_list in objmat4_list_ms:
            inp_mat4_list_ms.append(self.objmat4_list_inp(objmat4_list))
        return inp_mat4_list_ms

    def get_continuouspath(self, msc, grasp, objmat4_list, grasp_id=0, threshold=1, toggledebug=False,
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
        print(f"--------------get continuous path {grasp_id}---------------")
        success_cnt = 0
        path = []
        graspseq = self.get_eeseq_by_objmat4seq(grasp, objmat4_list)
        # init_msc = msc
        time_start_1 = time.time()
        for i in range(len(graspseq)):
            eepos = graspseq[i][0]
            eerot = graspseq[i][1]
            armjnts = self.get_numik(eepos, eerot, msc)
            if armjnts is not None:
                # stepdiff_norm = np.linalg.norm(armjnts - msc, ord=1)
                # if stepdiff_norm > 150:
                #     print(stepdiff_norm, armjnts)
                #     # armjnts = self.get_armjnts_by_eeposrot(eepos, eerot, init_msc)
                #     # print(armjnts)
                #     print("***************")
                #     continue
                path.append(armjnts)
                msc = copy.deepcopy(armjnts)
                success_cnt += 1
            else:
                return None
        print("Success point:", success_cnt, "of", len(graspseq))
        if success_cnt >= len(graspseq) * threshold - 1:
            if toggledebug:
                cost = self.get_path_cost(path)
                fig = plt.figure(1, figsize=(6.4, 4.8))
                plt.ion()
                self.rbth.plot_armjnts(path)
                plt.show()
                if dump_f_name is not None:
                    plt.savefig(
                        f"./log/path/{dump_f_name}_{grasp_id}_{round(cost, 3)}_{round(time.time() - time_start_1, 3)}_{len(objmat4_list)}.png")
                    plt.close(fig)
            return path

        return None

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
        _, _, prehndmat4 = grasp
        relpos, relrot = None, None
        for i, objmat4 in enumerate(objmat4_list):
            eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
            tgtpos = objmat4[:3, 3]
            tgtrot = objmat4[:3, :3]
            if relpos is None:
                relpos, relrot = rm.relpose(eepos, eerot, tgtpos, tgtrot)
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
        _, _, prehndmat4 = grasp
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
                relpos, relrot = rm.relpose(eepos, eerot, objmat4[:3, 3], objmat4[:3, :3])
            armjnts = self.get_numik_nlopt(objmat4[:3, 3], objmat4[:3, :3], seedjntagls=msc, toggledebug=toggledebug,
                                           releemat4=rm.homobuild(relpos, relrot), col_ps=col_ps,
                                           roll_limit=roll_limit, pos_limit=pos_limit, movedir=movedir)

            if armjnts is not None:
                path.append(armjnts)
                msc = copy.deepcopy(armjnts)
                success_cnt += 1
                if toggledebug:
                    eepos, eerot = self.rbth.get_ee(armjnts, releemat4=rm.homobuild(relpos, relrot))
                    self.rbth.draw_axis(eepos, eerot, length=100)
                    self.ah.show_armjnts(armjnts=armjnts, rgba=(0, 1, 0, .5))
                    axmat = self.rbth.manipulability_axmat(armjnts=armjnts, releemat4=rm.homobuild(relpos, relrot))
                    manipulability = self.rbth.manipulability(armjnts=armjnts, releemat4=rm.homobuild(relpos, relrot))
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

        for inx, v in enumerate(sample_range):
            r, p, y = v
            rot = np.dot(rm.rodrigues(objmat4[:3, 0], r),
                         np.dot(rm.rodrigues(objmat4[:3, 2], p), rm.rodrigues(objmat4[:3, 1], y)))
            objmat4_new = np.eye(4)
            objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
            objmat4_new[:3, 3] = objmat4[:3, 3]
            eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4_new)
            armjnts = self.get_numik(eepos, eerot)
            relpos, relrot = rm.relpose(eepos, eerot, objmat4_new[:3, 3], objmat4_new[:3, :3])
            if armjnts is not None:
                score = self.rbth.manipulability(armjnts=armjnts, releemat4=rm.homobuild(relpos, relrot))
                if toggledebug:
                    axmat = self.rbth.manipulability_axmat(armjnts=armjnts, releemat4=rm.homobuild(relpos, relrot))
                    self.ah.show_armjnts(armjnts=armjnts, rgba=(1, 1, 0, .2))
                    # self.rbth.draw_axis(objmat4_new[:3, 3], objmat4_new[:3, :3], rgba=(1, 1, 0, .5))
                    # self.rbth.draw_axis_uneven(objmat4_new[:3, 3], axmat,scale=.5)
                    print(score)

                if score > best_score:
                    best_config = armjnts
                    best_score = score
                    best_inx = inx
                    if toggledebug:
                        best_ellipsoid = axmat
                        best_objmat4 = objmat4_new
        if toggledebug:
            self.ah.show_armjnts(armjnts=best_config, rgba=(0, 1, 0, .5))
            # self.rbth.draw_axis(best_objmat4[:3, 3], best_objmat4[:3, :3], rgba=(1, 1, 0, .5))
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
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                 rgba=(1, 0, 0, 0.2), thickness=1)
            cnt = 0
            for r, p, y in sample_range:
                cnt += 1
                rot = np.dot(rm.rodrigues(objmat4[:3, 0], r),
                             np.dot(rm.rodrigues(objmat4[:3, 2], p), rm.rodrigues(objmat4[:3, 1], y)))
                objmat4_new = np.eye(4)
                objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
                objmat4_new[:3, 3] = objmat4[:3, 3]
                gtsp_dict[key]["objmat4_list"].append(objmat4_new)
                # base.pggen.plotArrow(base.render, spos=objmat4_new[:3, 3],
                #                      epos=objmat4_new[:3, 3] + 10 * objmat4_new[:3, 0],
                #                      rgba=(1, 1, 0, 0.2), thickness=1)

        for k, v in gtsp_dict.items():
            armjnts_list = []
            node_success = False
            for i, objmat4 in enumerate(v["objmat4_list"]):
                eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
                if relpos is None and relrot is None:
                    relpos, relrot = rm.relpose(eepos, eerot, objmat4[:3, 3], objmat4[:3, :3])
                # armjnts = self.get_numik(eepos, eerot, msc)
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
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
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
        # self.show_animation(armjnts_path)
        # base.run()
        return armjnts_path

    def get_continuouspath_opt2(self, grasp, grasp_id, objmat4_list, sample_range, msc=None):
        def __update(G, gtsp_dict, node_path, grasp, msc):
            print("msc", msc)
            for node in node_path:
                k, inx = node.split("_")
                objmat4 = gtsp_dict[int(k)]["objmat4_list"][int(inx)]
                armjnts = gtsp_dict[int(k)]["armjnts_list"][int(inx)]
                if armjnts is None:
                    eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
                    armjnts = self.get_numik(eepos, eerot, msc)
                if armjnts is None:
                    print("remove", node)
                    G.remove_node(node)
                    break
                else:
                    gtsp_dict[int(k)]["armjnts_list"][int(inx)] = armjnts
                    # msc = copy.deepcopy(armjnts)

        msc, start_id = self.__find_start_config(objmat4_list[0], grasp, sample_range)
        print(msc, start_id)
        # start_id = sample_range.index((0, 0, 0))
        if msc is None:
            return None

        time_start_1 = time.time()
        gtsp_dict = {}
        end = len(objmat4_list) - 1

        for key, objmat4 in enumerate(objmat4_list):
            gtsp_dict[key] = {"main": objmat4, "eemat4_list": [], "objmat4_list": [], "armjnts_list": []}
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                 rgba=(1, 0, 0, 0.2), thickness=1)
            cnt = 0
            for r, p, y in sample_range:
                cnt += 1
                rot = np.dot(rm.rodrigues(objmat4[:3, 0], r),
                             np.dot(rm.rodrigues(objmat4[:3, 2], p), rm.rodrigues(objmat4[:3, 1], y)))
                objmat4_new = np.eye(4)
                objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
                objmat4_new[:3, 3] = objmat4[:3, 3]
                eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4_new)
                gtsp_dict[key]["eemat4_list"].append(rm.homobuild(eepos, eerot))
                gtsp_dict[key]["objmat4_list"].append(objmat4_new)
                gtsp_dict[key]["armjnts_list"].append(None)

        G = nx.DiGraph()
        N = len(sample_range)

        for k, v in list(gtsp_dict.items())[:-1]:
            for i in range(N):
                for j in range(N):
                    eemat4_1 = gtsp_dict[k]["eemat4_list"][i]
                    eemat4_2 = gtsp_dict[k + 1]["eemat4_list"][j]
                    diff, _ = rm.degree_betweenrotmat(eemat4_1[:3, :3], eemat4_2[:3, :3])
                    G.add_weighted_edges_from([(f"{str(k)}_{str(i)}", f"{str(k + 1)}_{str(j)}", diff)])

        print("number of nodes:", len(G))
        print("build graph time cost:", time.time() - time_start_1)

        time_start_2 = time.time()

        armjnts_path = [None]
        iteration = 0

        while any(v is None for v in armjnts_path):
            if iteration > 50:
                return None
            source_list = [n for n in list(G.nodes) if str(n)[:2] == "0_"]
            target_list = [n for n in list(G.nodes) if str(n).split("_")[0] == str(end)]
            if len(source_list) == 0:
                return None
            print(f"---------------iteration {iteration}---------------")
            if G.has_node(f"0_{start_id}"):
                best_dist, best_path = gh.multi_target_shortest_path(G, f"0_{start_id}", target_list)
            else:
                best_dist, best_path = gh.multi_target_shortest_path(G, source_list[0], target_list)

            print("min cost:", best_dist)
            print("best path:", best_path)

            if best_path is None:
                return None

            __update(G, gtsp_dict, best_path, grasp, msc)
            armjnts_path = gh.get_nodes_value(gtsp_dict, best_path, "armjnts_list")
            objmat4_list = gh.get_nodes_value(gtsp_dict, best_path, "objmat4_list")
            iteration += 1

        print("search time cost:", time.time() - time_start_2)
        for objmat4 in objmat4_list:
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                 rgba=(0, 1, 0, 1), thickness=1)

        cost = self.get_path_cost(armjnts_path)
        fig = plt.figure(1, figsize=(12.8, 4.8))
        plt.ion()
        plt.subplot(121)
        self.rbth.plot_nodepath(best_path, title="node path")
        plt.subplot(122)
        self.rbth.plot_armjnts(armjnts_path)
        f_name = f"./log/path/discrete/" \
                 f"m2_{grasp_id}_{round(cost, 3)}_{round(time.time() - time_start_1, 3)}_{len(G)}"
        plt.savefig(f"{f_name}.png")
        plt.close(fig)
        pickle.dump(armjnts_path, open(f"{f_name}.pkl", "wb"))
        # self.show_animation(armjnts_path)
        # base.run()
        return armjnts_path

    def get_continuouspath_opt3(self, grasp, grasp_id, objmat4_list, sample_range, msc=None):
        def __get_candidate_nodes(k_len, inx_len, step=60, limit=None, seed=None):
            if seed is None:
                inx_list = [i for i in range(inx_len) if i % step == 0]
                candidate_nodes = [inx_list] * k_len
            else:
                candidate_nodes = []
                for s in seed:
                    k, inx = [int(v) for v in s.split("_")]
                    inx_list = [i for i in range(inx - limit, inx + limit) if i % step == 0 and i >= 0 and i != inx]
                    candidate_nodes.append(inx_list)
            return candidate_nodes

        def __get_candidate_nodes_random(gtsp_dict, num=5, seed=None):
            candidate_nodes = []
            inx_len = len(gtsp_dict[0]["armjnts_list"])
            k_len = len(gtsp_dict)
            pool_all = [range(inx_len)] * k_len
            pool_partial = copy.deepcopy(pool_all)
            if seed is not None:
                for s in seed:
                    pool_partial.append(range(int(s.split("_")[1]) - 5, int(s.split("_")[1]) + 5))

            for k, v in gtsp_dict.items():
                if all(v is None for v in v["armjnts_list"]):
                    candidate_nodes.append(random.choices(pool_all[k], k=int(np.ceil(num))))
                else:
                    candidate_nodes.append(random.choices(pool_all[k], k=int(np.ceil(num / 2))) +
                                           random.choices(pool_partial[k], k=int(np.ceil(num / 2))))

            return candidate_nodes

        def __update_dict(gtsp_dict, candidate_nodes, grasp, msc):
            for k, inx_list in enumerate(candidate_nodes):
                for inx in inx_list:
                    objmat4 = gtsp_dict[int(k)]["objmat4_list"][int(inx)]
                    armjnts = gtsp_dict[int(k)]["armjnts_list"][int(inx)]
                    if armjnts is None:
                        eepos, eerot = self.get_ee_by_objmat4(grasp, objmat4)
                        armjnts = self.get_numik(eepos, eerot, msc)
                        if armjnts is not None:
                            gtsp_dict[int(k)]["armjnts_list"][int(inx)] = armjnts
                            # msc = copy.deepcopy(armjnts)

        def __update_graph(G, gtsp_dict, candidate_nodes):
            for k in range(len(gtsp_dict) - 1):
                for i in candidate_nodes[k]:
                    for j in candidate_nodes[k + 1]:
                        armjnts_1 = gtsp_dict[k]["armjnts_list"][i]
                        armjnts_2 = gtsp_dict[k + 1]["armjnts_list"][j]
                        if armjnts_1 is None or armjnts_2 is None:
                            continue
                        diff = np.linalg.norm(armjnts_2 - armjnts_1, ord=1)
                        G.add_weighted_edges_from([(f"{str(k)}_{str(i)}", f"{str(k + 1)}_{str(j)}", diff)])
            print("number of nodes:", len(G))

        # start_id = sample_range.index((0, 0, 0))
        msc, start_id = self.__find_start_config(objmat4_list[0], grasp, sample_range)
        print(msc, start_id)
        if msc is None:
            return None

        time_start_1 = time.time()
        gtsp_dict = {}
        G = nx.DiGraph()
        end = len(objmat4_list) - 1

        for key, objmat4 in enumerate(objmat4_list):
            gtsp_dict[key] = {"main": objmat4, "objmat4_list": [], "armjnts_list": []}
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                 rgba=(1, 0, 0, 0.2), thickness=1)
            cnt = 0
            for r, p, y in sample_range:
                cnt += 1
                rot = np.dot(rm.rodrigues(objmat4[:3, 0], r),
                             np.dot(rm.rodrigues(objmat4[:3, 2], p), rm.rodrigues(objmat4[:3, 1], y)))
                objmat4_new = np.eye(4)
                objmat4_new[:3, :3] = np.dot(rot, objmat4[:3, :3])
                objmat4_new[:3, 3] = objmat4[:3, 3]
                gtsp_dict[key]["objmat4_list"].append(objmat4_new)
                gtsp_dict[key]["armjnts_list"].append(None)
                # base.pggen.plotArrow(base.render, spos=objmat4_new[:3, 3],
                #                      epos=objmat4_new[:3, 3] + 10 * objmat4_new[:3, 0],
                #                      rgba=(1, 1, 0, 0.2), thickness=1)

        iteration = 0
        fig = plt.figure(1, figsize=(12.8, 4.8))
        best_path = None
        while iteration < 5:
            print(f"---------------iteration {iteration}---------------")
            iteration += 1
            # candidate_nodes = __get_candidate_nodes(len(objmat4_list), len(sample_range), step=5, limit=None, seed=None)
            candidate_nodes = __get_candidate_nodes_random(gtsp_dict, num=5, seed=None)
            print(Counter([len(l) for l in candidate_nodes]))
            __update_dict(gtsp_dict, candidate_nodes, grasp, msc)
            __update_graph(G, gtsp_dict, candidate_nodes)

            source_list = [n for n in list(G.nodes) if str(n)[:2] == "0_"]
            target_list = [n for n in list(G.nodes) if str(n).split("_")[0] == str(end)]

            if len(source_list) == 0:
                continue

            if G.has_node(f"0_{start_id}"):
                best_dist, best_path = gh.multi_target_shortest_path(G, f"0_{start_id}", target_list)
            else:
                best_dist, best_path = gh.multi_target_shortest_path(G, source_list[0], target_list)
            if best_path is None:
                continue
            plt.subplot(121)
            self.rbth.plot_nodepath(best_path, label=str(iteration), title="node path")
            print("min cost:", best_dist)
            print("best path:", best_path)

        if best_path is None:
            return None

        plt.ion()
        armjnts_path = []
        for node in best_path:
            k, objmat4_id = node.split("_")
            objmat4 = gtsp_dict[int(k)]["objmat4_list"][int(objmat4_id)]
            # self.show_armjnts(rgba=(0, 1, 0, 0.5), armjnts=gtsp_dict[int(k)]["armjnts_list"][int(objmat4_id)])
            armjnts_path.append(gtsp_dict[int(k)]["armjnts_list"][int(objmat4_id)])
            base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                 rgba=(0, 1, 0, 1), thickness=1)
        print("time cost:", time.time() - time_start_1)

        plt.subplot(122)
        self.rbth.plot_armjnts(armjnts_path)
        plt.show()
        f_name = f"./log/path/discrete/" \
                 f"m3_{grasp_id}_{round(best_dist, 3)}_{round(time.time() - time_start_1, 3)}_{len(G)}"
        plt.savefig(f"{f_name}.png")
        plt.close(fig)
        pickle.dump(armjnts_path, open(f"{f_name}.pkl", "wb"))
        # self.show_animation(armjnts_path)
        # base.run()
        return armjnts_path

    def refine_continuouspath_by_transmat(self, objrelpos, objrelrot, path, grasp, objcm, transmat):
        time_start = time.time()
        print("transmat:", transmat)
        success_cnt = 0
        path_new = []
        path_mask = []
        grasp_refined = self.refine_grasp(grasp, transmat)
        # objrelpos_refined, objrelrot_refined = copy.deepcopy(objrelpos), copy.deepcopy(objrelrot)
        objrelpos_refined, objrelrot_refined = None, None
        for armjnts in path:
            self.rbth.goto_armjnts(armjnts)
            objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
            if objrelpos_refined is None:
                objrelpos_refined, objrelrot_refined = self.get_rel_posrot(grasp_refined, objpos, objrot)

            objmat4 = rm.homobuild(objpos, objrot)
            # print("pos diff:", objmat4[:3, 3] - objmat4_new[:3, 3])
            armjnts = self.get_armjnts_by_objmat4ngrasp(grasp_refined, objcm, objmat4, msc=armjnts)

            if armjnts is not None:
                stepdiff_norm = np.linalg.norm(armjnts - path[-1], ord=1)
                if stepdiff_norm > 300:
                    print(stepdiff_norm, armjnts)
                    print("***************")
                    path_mask.append(False)
                    continue
                path_new.append(armjnts)
                path_mask.append(True)
                success_cnt += 1
            else:
                path_mask.append(False)

        print("Success point:", success_cnt, "of", len(path))
        print("time cost(refine by transmat):", time.time() - time_start)
        return grasp_refined, objrelpos_refined, objrelrot_refined, path_new, path_mask

    def refine_continuouspath_by_posdiff(self, objrelpos, objrelrot, path, grasp, objcm, posdiff, path_mask=None):
        if path_mask is None:
            path_mask = [True] * len(path)
        success_cnt = 0
        path_new = []
        time_start = time.time()

        for i, armjnts in enumerate(path):
            if path_mask[i]:
                self.rbth.goto_armjnts(armjnts)
                objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
                objmat4 = rm.homobuild(objpos + posdiff, objrot)
                armjnts_new = self.get_armjnts_by_objmat4ngrasp(grasp, objcm, objmat4, armjnts)
                if armjnts_new is not None:
                    stepdiff_norm = np.linalg.norm(armjnts_new - armjnts, ord=1)
                    if stepdiff_norm > 300:
                        print("******nlopt******")
                        armjnts_new = self.get_numik_nlopt(objmat4[:3, 3], objmat4[:3, :3], seedjntagls=armjnts,
                                                           releemat4=rm.homobuild(objrelpos, objrelrot))
                    stepdiff_norm = np.linalg.norm(armjnts_new - armjnts, ord=1)
                    if stepdiff_norm > 300:
                        print(stepdiff_norm, armjnts_new)
                        print("******fail******")
                        path_mask[i] = False
                        continue
                    path_new.append(armjnts_new)
                    success_cnt += 1
                else:
                    path_mask[i] = False

        print("Success point:", success_cnt, "of", len(path))
        print("time cost(refine by pos diff):", time.time() - time_start)

        return path_new, path_mask

    def refine_continuouspath_lft(self, path, stop_id, objrelpos, objrelrot, n0, grasp, objcm):
        path_res = path[stop_id:]
        pos1 = self.get_world_objmat4(objrelpos, objrelrot, path[stop_id - 5])[:3, 3]
        self.ah.show_armjnts(armjnts=path_res[0], rgba=(1, 0, 0, .5))
        for i, armjnts in enumerate(path_res):
            objmat4 = self.get_world_objmat4(objrelpos, objrelrot, armjnts)
            n = objmat4[:3, 0]
            if np.degrees(rm.angle_between_vectors(n, n0)) > 15:
                pos2 = objmat4[:3, 3]
                posdiff = pos1 - pos2
                # posdiff[1] = posdiff[1] - 6
                print('posdiff:', posdiff)
                path_res, _ = \
                    self.refine_continuouspath_by_posdiff(objrelpos, objrelrot, path_res[i:], grasp, objcm, posdiff)
                # self.ah.show_armjnts(armjnts=armjnts, rgba=(0, 1, 0, .5))
                self.ah.show_armjnts_with_obj(armjnts, objcm, objrelpos, objrelrot, rgba=(0, 1, 0, .5))
                break
        return path_res

    def refine_continuouspath_rgt(self, path, stop_id, objrelpos, objrelrot, f, grasp, objcm):
        path_pre = path[:stop_id][::-1]
        path_res = path[stop_id:]
        path_mid = []
        posdiff = [0, 0, 0]
        self.ah.show_armjnts_with_obj(path[stop_id], objcm, objrelpos, objrelrot, rgba=(0, 1, 0, .5))
        spos = self.get_world_objmat4(objrelpos, objrelrot, path[stop_id])[:3, 3]
        base.pggen.plotArrow(base.render, spos=spos, epos=spos + 10 * f)

        for i, armjnts in enumerate(path_pre):
            objmat4 = self.get_world_objmat4(objrelpos, objrelrot, armjnts)
            n = objmat4[:3, 0]
            print(np.degrees(rm.angle_between_vectors(n, f)))
            if rm.angle_between_vectors(n, np.asarray(f)) < np.radians(1):
                self.ah.show_armjnts(armjnts=armjnts, rgba=(1, 0, 0, .5))
                pos1 = objmat4[:3, 3]
                pos2 = self.get_world_objmat4(objrelpos, objrelrot, path_pre[i + 2])[:3, 3]
                posdiff = pos1 - pos2

                print(posdiff)
                objmat4_new = np.eye(4)
                objmat4_new[:3, :3] = objmat4[:3, :3]
                path_mid = [armjnts]
                for cnt in range(1, 16):
                    posdiff[1] = posdiff[1] + 1
                    objmat4_new[:3, 3] = pos1 + posdiff
                    path_mid.append(self.get_armjnts_by_objmat4ngrasp(grasp, objcm, objmat4_new, msc=armjnts))
                path_res = path[(stop_id - i):]
                self.ah.show_armjnts(armjnts=path_mid[-1], rgba=(1, 1, 0, .5))
                break
        path_res, _ = self.refine_continuouspath_by_posdiff(objrelpos, objrelrot, path_res, grasp, objcm, posdiff)

        return path_mid + path_res

    def goto_posrot(self, eepos, eerot):
        # armjnts = self.rbt.numik(eepos, eerot, self.armname)
        armjnts = self.get_numik(eepos, eerot)
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, self.armname)
            return armjnts
        else:
            print("No IK solution for the given pos, rot.")
            return None

    def goto_posrot_msc(self, eepos, eerot, msc):
        armjnts = self.get_numik(eepos, eerot, msc=msc)
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, self.armname)
            return armjnts
        else:
            print("No IK solution for the given pos, rot.")
            return None

    def get_world_objmat4(self, objrelpos, objrelrot, armjnts=None):
        if armjnts is not None:
            self.rbth.goto_armjnts(armjnts)
        # objpos, objrot = self.rbt.getee(armname=self.armname)
        objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        objmat4 = rm.homobuild(objpos, objrot)
        return objmat4

    def get_moveup_path(self, start, obj, objrelpos, objrelrot, direction=[0, 0, 1], length=20):
        return self.ctcallback.getLinearPrimitive(start, direction, length, [obj], [[objrelpos, objrelrot]], [],
                                                  type="source")

    def get_movedown_path(self, start, obj, objrelpos, objrelrot, direction=[0, 0, 1], length=20):
        return self.ctcallback.getLinearPrimitive(start, direction, length, [obj], [[objrelpos, objrelrot]], [],
                                                  type="sink")

    def homomat2vec(self, objrelpos, objrelrot):
        rotation = Rotation.from_dcm(objrelrot)
        return objrelpos.tolist() + rotation.as_rotvec().tolist()

    def refine_relpose_by_transmat(self, objrelpos, objrelrot, transmat):
        objmat4 = rm.homobuild(objrelpos, objrelrot)
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
        prejawwidth, prehndfc, prehndmat4 = grasp
        hndmat4 = np.dot(objmat4, prehndmat4)
        eepos = rm.homotransformpoint(objmat4, prehndfc)[:3]
        eerot = hndmat4[:3, :3]
        armjnts = self.get_numik(eepos, eerot)
        obj.sethomomat(objmat4)

        if armjnts is not None:
            if not self.rbth.is_selfcollided(armjnts):
                return True
        return False
