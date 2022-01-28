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
import motionplanner.motion_planner as m_planner


class BendRbtPlanner(object):
    def __init__(self, bendsim, motionplanner):
        self.bs = bendsim
        self.rbt = motionplanner.rbt
        self.mp = motionplanner

    def gen_rbtpose(self, bendresseq, grasp, max_fail=np.inf):
        armjntsseq = []
        fail_cnt = 0
        for bendres in bendresseq:
            init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres
            pseq_init = rm.homomat_transform_points(transmat4, pseq_init).tolist()
            rotseq_init = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq_init])
            self.bs.reset(pseq_init, rotseq_init, extend=False)
            objcm_init = copy.deepcopy(self.bs.objcm)
            objpos = pseq_init[0]
            objrot = rotseq_init[0]
            armjnts = self.mp.get_armjnts_by_objmat4ngrasp(grasp, objcm_init, rm.homomat_from_posrot(objpos, objrot))
            if armjnts is None:
                fail_cnt += 1
                if fail_cnt > max_fail:
                    break
            #     self.mp.ah.show_armjnts(armjnts=armjnts)
            #     objcm_init.set_rgba((1, 1, 0, 1))
            #     objcm_init.attach_to(base)
            armjntsseq.append(armjnts)
        return armjntsseq

    def load_bendresseq(self, f_name='./tmp_bendresseq.pkl'):
        return pickle.load(open(f_name, 'rb'))

    def show_bendresseq(self, bendresseq, transmat4):
        motioncounter = [0]
        taskMgr.doMethodLater(.05, self.__update, "update",
                              extraArgs=[motioncounter, bendresseq, transmat4], appendTask=True)

    def show_bendresseq_withrbt(self, bendresseq, transmat4, armjntsseq):
        motioncounter = [0]
        rbtmnp = [None, None]
        taskMgr.doMethodLater(.05, self.__update_rbt, "update",
                              extraArgs=[rbtmnp, motioncounter, bendresseq, transmat4, armjntsseq], appendTask=True)

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
                gm.gen_frame(pseq_init[0], rotseq_init[0], length=.02, thickness=.001).attach_to(base)

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
    bendseq = pickle.load(open('./tmp_bendseq.pkl', 'rb'))

    transmat4 = rm.homomat_from_posrot((.8, .3, .78 + 0.15175), np.eye(3))
    bendresseq = pickle.load(open('./tmp_bendresseq.pkl', 'rb'))

    grasp_list = mp_lft.load_all_grasp('stick')
    print(len(grasp_list))
    # min_fail = np.inf
    # for i, grasp in enumerate(grasp_list):
    #     print('-' * 10)
    #     print('grasp_id:', i)
    #     armjntsseq_tmp = brp.gen_rbtpose(bendresseq, grasp, max_fail=min_fail)
    #     fail_cnt = [str(v) for v in armjntsseq_tmp].count('None')
    #     print(min_fail, fail_cnt)
    #     if fail_cnt < min_fail:
    #         min_fail = fail_cnt
    #         armjntsseq = armjntsseq_tmp
    #     if fail_cnt == 0:
    #         break
    # pickle.dump(armjntsseq, open('./tmp_armjntsseq.pkl', 'wb'))
    armjntsseq = pickle.load(open('./tmp_armjntsseq.pkl', 'rb'))
    brp.show_bendresseq_withrbt(bendresseq, transmat4, armjntsseq)

    base.run()
