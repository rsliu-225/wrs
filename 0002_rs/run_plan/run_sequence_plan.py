import time

import numpy as np
import visualization.panda.world as wd

# import motionplanner.motion_planner as m_planner
# import motionplanner.rbtx_motion_planner as m_plannerx
import bendplanner.BendSim as b_sim
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.PremutationTree as p_tree
import bendplanner.InvalidPermutationTree as ip_tree
import bendplanner.bend_utils as bu
# import utils.phoxi as phoxi
# import utils.phoxi_locator as pl
import basis.robot_math as rm
# from utils.run_script_utils import *
import pickle
import copy
import localenv.envloader as el
import motionplanner.motion_planner as m_planner
import modeling.geometric_model as gm
import matplotlib.pyplot as plt
import config


def plan_pt(bendset):
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


def plan_ipt(bs, bendset, mode='all', f_name=''):
    ts = time.time()
    iptree = ip_tree.IPTree(len(bendset))
    valid_tree = ip_tree.IPTree(len(bendset))
    seqs = iptree.get_potential_valid()
    result = []
    tc_list = []
    cnt = 0
    while len(seqs) != 0:
        bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
        bendseq = [bendset[i] for i in seqs]
        is_success, bendresseq = bs.gen_by_bendseq(bendseq, cc=True, prune=True, toggledebug=False)
        print(is_success)
        cnt += 1
        # bs.show_bendresseq(bendresseq, is_success)
        # base.run()
        if all(is_success):
            result.append([bendresseq, seqs])
            tc_list.append(time.time() - ts)
            if mode == 'all':
                valid_tree.add_invalid_seq(seqs)
                iptree.add_invalid_seq(seqs)
                valid_tree.show()
                seqs = iptree.get_potential_valid()
                continue
            else:
                bs.show_bendresseq(bendresseq, is_success)
                return result, tc_list, cnt, time.time() - ts
        iptree.add_invalid_seq(seqs[:is_success.index(False) + 1])
        # iptree.show()
        seqs = iptree.get_potential_valid()
        print(seqs)
    valid_tree.show()
    pickle.dump([result, tc_list, cnt, time.time() - ts, bendset],
                open(f'{config.ROOT}/bendplanner/bendresseq/{f_name}.pkl', 'wb'))
    return result, tc_list, cnt, time.time() - ts


if __name__ == '__main__':
    '''
    set up env and param
    '''
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()

    '''
    init class
    '''
    # mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bs = b_sim.BendSim(show=True)

    # bendset = [
    #     [np.radians(225), np.radians(0), np.radians(0), .04],
    #     [np.radians(90), np.radians(0), np.radians(180), .08],
    #     [np.radians(90), np.radians(0), np.radians(0), .1],
    #     [np.radians(45), np.radians(0), np.radians(180), .12],
    #     # [np.radians(40), np.radians(0), np.radians(0), .06],
    #     # [np.radians(-15), np.radians(0), np.radians(0), .08],
    #     # [np.radians(20), np.radians(0), np.radians(0), .1]
    # ]
    # bendset = pickle.load(open('./tmp_bendseq.pkl', 'rb'))
    random_cnt = 8

    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) / 2
    # init_pseq = [(0, 0, 0), (0, .1 + bu.cal_length(goal_pseq), 0)]
    # init_rotseq = [np.eye(3), np.eye(3)]
    # brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp_lft)
    # fit_pseq = bu.iter_fit(goal_pseq, tor=.002, toggledebug=False)
    # bendset = brp.pseq2bendset(fit_pseq, pos=.1, toggledebug=False)

    # bs.show(rgba=(.7, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)
    for i in range(3, 10):
        bendset = bs.gen_random_bendset(random_cnt)
        bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
        # is_success, bendresseq = bs.gen_by_bendseq(bendset, cc=False, prune=False, toggledebug=False)
        # ax = plt.axes(projection='3d')
        # bu.plot_pseq(ax, bs.pseq, c='k')
        # bu.scatter_pseq(ax, bs.pseq, c='r')
        # plt.show()
        flag, tc, cnt, total_tc = plan_ipt(bs, bendset, mode='all', f_name=f'{str(random_cnt)}_{str(i)}')
        print(tc, cnt)
        print(total_tc)

    # base.run()
