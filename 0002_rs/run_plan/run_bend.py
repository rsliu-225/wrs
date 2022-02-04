import numpy as np
import visualization.panda.world as wd

import motionplanner.motion_planner as m_planner
# import motionplanner.rbtx_motion_planner as m_plannerx
import bendplanner.BendSim as b_sim
import bendplanner.PremutationTree as p_tree
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import basis.robot_math as rm
from utils.run_script_utils import *
import pickle

if __name__ == '__main__':
    # '''
    # set up env and param
    # '''
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()
    # rbtx = el.loadUr3ex(rbt)
    # rbt.lft_arm_hnd.open_gripper()
    # rbt.rgt_arm_hnd.open_gripper()
    #
    # '''
    # init class
    # '''
    # mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    # mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")
    # phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bendset = [
        [np.radians(225), np.radians(0), np.radians(0), .04],
        [np.radians(-90), np.radians(0), np.radians(0), .08],
        [np.radians(90), np.radians(0), np.radians(0), .1],
        [np.radians(-45), np.radians(0), np.radians(0), .12],
        # [np.radians(40), np.radians(0), np.radians(0), .06],
        # [np.radians(-15), np.radians(0), np.radians(0), .08],
        # [np.radians(20), np.radians(0), np.radians(0), .1]
    ]
    # bendseq = pickle.load(open('./tmp_bendseq.pkl', 'rb'))

    bs = b_sim.BendSim(show=True)
    bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
    # bs.show(rgba=(.7, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)

    ptree = p_tree.PTree(len(bendset))
    dummy_ptree = copy.deepcopy(ptree)
    seqs = dummy_ptree.output()
    while len(seqs) != 0:
        bendseq = [bendset[i] for i in seqs[0]]
        is_success, bendresseq = bs.gen_by_bendseq(bendseq, cc=True, toggledebug=False)
        print(is_success)
        if all(is_success[:3]):
            bs.show_bendresseq(bendresseq, is_success)
            base.run()
        if all(is_success):
            pickle.dump(bendresseq, open('./tmp_bendresseq.pkl', 'wb'))
            bs.show_bendresseq(bendresseq, is_success)
            base.run()
        bs.reset([(0, 0, 0), (0, bendseq[-1][3], 0)], [np.eye(3), np.eye(3)])
        dummy_ptree.prune(seqs[0][:is_success.index(False) + 1])
        ptree.prune(seqs[0][:is_success.index(False) + 1])
        seqs = dummy_ptree.output()

    # is_success, bendresseq = bs.gen_by_bendseq(bendseq, cc=True, toggledebug=False)
    # # pickle.dump(bendresseq, open('./tmp_bendresseq.pkl', 'wb'))
    # # bendresseq = pickle.load(open('./tmp_bendresseq.pkl', 'rb'))
    #
    # bs.show_bendresseq(bendresseq)
    # print('Success:', is_success)
    # # bs.show(rgba=(0, 0, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1)), show_pseq=True, show_frame=True)
    # bs.show(rgba=(0, 0, .7, .7), show_pseq=True, show_frame=True)
    #
    base.run()
