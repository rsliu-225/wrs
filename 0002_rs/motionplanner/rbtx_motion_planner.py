import numpy as np
import time
import pickle
import utiltools.robotmath as rm
from forcecontrol.force_controller import ForceController
from motion import collisioncheckerball as cdck
from motion import smoother as sm
from motion.rrt import rrtconnect as rrtc
from motionplanner.motion_planner import MotionPlanner
import utils.run_utils as ru


class MotionPlannerRbtX(MotionPlanner):
    def __init__(self, env, rbt, rbtmg, rbtball, rbtx, armname):
        MotionPlanner.__init__(self, env, rbt, rbtmg, rbtball, armname)
        self.rbtx = rbtx
        self.pcdchecker = cdck.CollisionCheckerBall(self.rbtball)
        self.force_controller = ForceController(self.rbt, self.rbtx, armname)
        self.arm = self.rbtx.rgtarm if armname == "rgt" else self.rbtx.lftarm

    def movepath(self, path):
        print("--------------move path---------------")

        self.rbtx.movejntssgl_cont(path, self.armname, wait=True)
        time.sleep(.5)
        while self.arm.is_program_running():
            time.sleep(1)
        time.sleep(1)

    def get_armjnts(self):
        return self.rbtx.getjnts(armname=self.armname)

    def goto_init_x(self):
        start = self.rbtx.getjnts(armname=self.armname)
        if self.armname == "lft":
            goal = self.rbt.initlftjnts
        else:
            goal = self.rbt.initrgtjnts

        print("--------------go to init(rrt)---------------")
        planner = rrtc.RRTConnect(start=start, goal=goal, checker=self.ctcallback,
                                  starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=500,
                                  maxtime=100.0)
        path_gotoinit, samples = planner.planning(self.obscmlist)
        smoother = sm.Smoother()

        if path_gotoinit is not None:
            print("--------------smoother---------------")
            path_gotoinit = smoother.pathsmoothing(path_gotoinit, planner)
        self.rbtx.movejntssgl_cont(path_gotoinit, self.armname, wait=False)
        time.sleep(.5)
        while self.arm.is_program_running():
            pass
        time.sleep(.5)

    def goto_init_hold_x(self, grasp, objcm, objrelpos, objrelrot):
        start = self.rbtx.getjnts(armname=self.armname)
        if self.armname == "lft":
            goal = self.rbt.initlftjnts
        else:
            goal = self.rbt.initrgtjnts
        # objmat4_start = self.get_world_objmat4(objrelpos, objrelrot, start)
        # objmat4_goal = self.get_world_objmat4(objrelpos, objrelrot, goal)

        path_gotoinit = self.plan_start2end_hold_armj([start, goal], objcm, objrelpos, objrelrot)
        if path_gotoinit is not None:
            self.rbtx.movejntssgl_cont(path_gotoinit, self.armname, wait=True)
            time.sleep(.5)
            while self.arm.is_program_running():
                time.sleep(1)
            time.sleep(1)
            return True
        return False

    def move_up_x(self, obj, objrelpos, objrelrot, direction=[0, 0, 1], length=20):
        print(f"--------------move up {length}---------------")
        path_up = self.get_moveup_path(self.get_armjnts(), obj, objrelpos, objrelrot, length=length,
                                       direction=direction)
        self.rbtx.movejntssgl_cont(path_up, self.armname, wait=True)
        while self.arm.is_program_running():
            time.sleep(1)
        return path_up

    def get_obj_direction(self, objrelpos, objrelrot, org_direction=np.array([-1, 0, 0])):
        objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        print(objrot)
        obj_direction = np.dot(org_direction, objrot.T)
        print("obj direction:", obj_direction)
        return obj_direction

    def zerotcpforce(self):
        self.rbtx.zerotcpforce(armname=self.armname)

    def attachfirm(self, direction=np.array([0, 0, -1]), forcethreshold=2.0):
        self.force_controller.attachfirm(direction=direction, forcethreshold=forcethreshold)
        time.sleep(5)

    def refine_path_by_attatchfirm(self, objrelpos, objrelrot, path, objcm, grasp, path_mask=None, forcethreshold=2.5):
        if path_mask is not None:
            path_mask = [True] * len(path)
        time.sleep(1)
        eepos, eerot = self.get_ee(path[0])
        direction = np.dot(eerot, objrelrot)
        direction = np.dot(direction, np.array([-1, 0, 0]))
        print("--------------attach firm---------------")
        self.zerotcpforce()
        time.sleep(1)
        self.attachfirm(direction=direction, forcethreshold=forcethreshold)
        while self.arm.is_program_running():
            time.sleep(1)
        time.sleep(1)

        eepos_real, eerot_real = self.get_ee()
        print("--------------refine draw path---------------")
        posdiff = eepos_real - eepos
        print("pen tip deviation:", posdiff)
        path_new, path_mask_new = \
            self.refine_continuouspath_by_posdiff(objrelpos, objrelrot, path, grasp, objcm, posdiff,
                                                  path_mask=path_mask)
        return path_new, path_mask_new

    def get_objmat4_inhand(self, phxilocator, phoxi_f_name, stl_f_name, objmat4_sim, armjnts=None, load=True,
                           toggledubug=False, showicp=False, showcluster=False):
        tcppos, tcprot = self.get_ee(armjnts)
        obj_inhand_item = \
            ru.get_obj_inhand_from_phoxiinfo_withmodel(phxilocator, stl_f_name, tcppos, inithomomat=objmat4_sim,
                                                       load=load, phoxi_f_name=phoxi_f_name,
                                                       showicp=showicp, showcluster=showcluster)
        objmat4_real = obj_inhand_item.objmat4
        rot_x = rm.euler_from_matrix(objmat4_sim[:3, :3])[0] - rm.euler_from_matrix(objmat4_real[:3, :3])[0]
        objmat4_real[:3, :3] = np.dot(objmat4_real[:3, :3], rm.rodrigues((1, 0, 0), rot_x))
        if toggledubug:
            self.ah.show_armjnts(armjnts=armjnts, jawwidth=15, rgba=(1, 1, 0, .5))
            pcdu.show_pcd(obj_inhand_item.pcd, rgba=(0, 1, 0, 1))
            self.ah.show_objmat4(obj_inhand_item.objcm, obj_inhand_item.objmat4, rgba=(1, 0, 0, 1),
                                 showlocalframe=True)
            self.ah.show_objmat4(obj_inhand_item.objcm, objmat4=objmat4_sim, rgba=(1, 1, 0, .5), showlocalframe=True)
            base.run()
        return objmat4_real

    def get_transmat_by_vision(self, phxilocator, phoxi_f_name, stl_f_name, objmat4_sim, load=True, armjnts=None,
                               toggledubug=False, showicp=False, showcluster=False):
        time_start = time.time()
        objmat4_real = self.get_objmat4_inhand(phxilocator, phoxi_f_name, stl_f_name, objmat4_sim, armjnts=armjnts,
                                               load=load, toggledubug=toggledubug, showicp=showicp,
                                               showcluster=showcluster)
        # rot_x = rm.euler_from_matrix(objmat4_sim[:3, :3])[0] - rm.euler_from_matrix(objmat4_real[:3, :3])[0]
        # objmat4_real[:3, :3] = np.dot(objmat4_real[:3, :3], rm.rodrigues((1, 0, 0), rot_x))

        transmat = np.dot(np.linalg.inv(objmat4_real), objmat4_sim)

        pickle.dump(transmat, open("transmat_temp.pkl", "wb"))
        print("time cost(get transmat):", time.time() - time_start)
        print("pen tip deviation:", objmat4_real[:3, 3] - objmat4_sim[:3, 3])

        return transmat

    # def passive_move(self, path, toolrelpose):
    #     armjnts_list_flatten = []
    #     path_inp = []
    #     traj = trajectory.Trajectory(type="quintic")
    #
    #     for i in range(len(path) - 1):
    #         diff = np.linalg.norm(path[i] - path[i + 1])
    #         if diff < 2:
    #             path_inp.append(path[i])
    #         else:
    #             path_temp = traj.piecewiseinterpolation([path[i], path[i + 1]], sampling=np.floor(diff/2))
    #             path_inp.extend(path_temp[:-1])
    #     # self.plot_armjnts(traj.piecewiseinterpolation(path, sampling=100))
    #     for armjnts in path:
    #         armjnts_rad = np.deg2rad(armjnts)
    #         armjnts_list_flatten += armjnts_rad.tolist()
    #     self.force_controller.passive_move(armjnts_list_flatten, toolrelpose)

    def get_lower_path(self, objrelpos, objrelrot, path, objcm, grasp, distance=2):
        success_cnt = 0
        path_new = []

        tcppos, tcprot = self.get_ee(armjnts=path[0])
        penrot = np.dot(tcprot, objrelrot)
        pentip_direction = np.dot(penrot, np.array([-1, 0, 0]))

        for armjnts in path:
            self.rbth.goto_armjnts(armjnts)
            objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
            objpos_new = objpos + distance * pentip_direction
            objmat4_new = rm.homobuild(objpos_new, objrot)
            # print("pos diff:", np.linalg.norm(objpos - objpos_new))
            armjnts = self.get_armjnts_by_objmat4ngrasp(grasp, objcm, objmat4_new, armjnts)

            if armjnts is not None:
                path_new.append(armjnts)
                success_cnt += 1
        print("Success point:", success_cnt, "of", len(path))
        return path_new

    def goto_armjnts_x(self, armjnts):
        start = self.get_armjnts()
        print("--------------goto_armjnts_x(rrt)---------------")
        planner = rrtc.RRTConnect(start=start, goal=armjnts, checker=self.ctcallback,
                                  starttreesamplerate=30, goaltreesamplerate=30, expanddis=10, maxiter=500,
                                  maxtime=100.0)
        path, _ = planner.planning(self.obscmlist)
        smoother = sm.Smoother()

        if path is not None:
            print("--------------smoother---------------")
            path = smoother.pathsmoothing(path, planner)
            self.rbtx.movejntssgl_cont(path, self.armname, wait=True)
            time.sleep(.5)
            while self.arm.is_program_running():
                pass
            time.sleep(.5)
            return True
        else:
            return False

    def goto_armjnts_hold_x(self, grasp, objcm, objrelpos, objrelrot, armjnts):
        start = self.get_armjnts()
        objmat4_start = self.get_world_objmat4(objrelpos, objrelrot, start)
        objmat4_goal = self.get_world_objmat4(objrelpos, objrelrot, armjnts)
        print("--------------goto_armjnts_hold_x(rrt)---------------")
        path = self.plan_start2end_hold(grasp, [objmat4_start, objmat4_goal], objcm, objrelpos, objrelrot, start=start,
                                        use_msc=False)
        if path is not None:
            self.rbtx.movejntssgl_cont(path, self.armname, wait=True)
            time.sleep(.5)
            while self.arm.is_program_running():
                pass
            time.sleep(.5)
            return True
        return False

    def goto_objmat4_goal_x(self, grasp, objrelpos, objrelrot, objmat4_goal, objcm):
        objmat4_start = self.get_world_objmat4(objrelpos, objrelrot, armjnts=self.get_armjnts())
        path = self.plan_start2end_hold(grasp, [objmat4_start, objmat4_goal], objcm, objrelpos, objrelrot,
                                        start=self.get_armjnts())
        if path is not None:
            self.rbtx.movejntssgl_cont(path, self.armname, wait=True)
            while self.arm.is_program_running():
                pass
            time.sleep(.5)
            return True
        else:
            return False


if __name__ == '__main__':
    from utils.run_script_utils import *

    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")
    mp_x_lft = MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    armjnts = mp_x_lft.get_armjnts()
    mp_x_lft.ah.show_armjnts(armjnts=armjnts, toggleendcoord=True)
    base.run()
