import copy
import pickle
import time
import itertools

import numpy as np
import motionplanner.motion_planner as m_planner
import bendplanner.bend_utils as bu
import bendplanner.BendSim as b_sim
import bendplanner.InvalidPermutationTree as ip_tree
import bendplanner.PremutationTree as p_tree
import config
import matplotlib.pyplot as plt
import visualization.panda.world as wd


def plan_pt(bendset):
    ptree = p_tree.PTree(len(bendset))
    dummy_ptree = copy.deepcopy(ptree)
    seqs = dummy_ptree.output()
    while len(seqs) != 0:
        bendseq = [bendset[i] for i in seqs[0]]
        is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=True, toggledebug=False)
        print(is_success)
        if all(is_success[:3]):
            bs.show_bendresseq(bendresseq, is_success)
            base.run()
        if all(is_success):
            pickle.dump(bendresseq, open('./penta_bendresseq.pkl', 'wb'))
            bs.show_bendresseq(bendresseq, is_success)
            base.run()
        bs.reset([(0, 0, 0), (0, max([b[-1] for b in bendset]) + .05, 0)], [np.eye(3), np.eye(3)])
        dummy_ptree.prune(seqs[0][:is_success.index(False) + 1])
        ptree.prune(seqs[0][:is_success.index(False) + 1])
        seqs = dummy_ptree.output()


def plan_premutation(bs, bendset, snum=None, fo='180', f_name=''):
    ts = time.time()
    combs = list(itertools.permutations(range(len(bendset)), len(bendset)))

    result = []
    tc_list = []
    attemp_cnt_list = []
    cnt = 0
    success_cnt = 0
    combs = bu.rank_combs(combs)
    while combs:
        seqs = combs[0]
        print('Candidate seqs:', seqs, 'out of', len(combs))
        bendseq = [bendset[i] for i in seqs]
        bs.reset([(0, 0, 0), (0, max([b[-1] for b in bendset]) + .05, 0)], [np.eye(3), np.eye(3)])
        is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=True, prune=True, toggledebug=False)
        print(is_success)
        cnt += 1
        # bs.show_bendresseq(bendresseq, is_success)
        # base.run()
        if all(is_success):
            result.append(seqs)
            tc_list.append(time.time() - ts)
            attemp_cnt_list.append(cnt)
            success_cnt += 1
            if snum is None or success_cnt < snum:
                # combs = remove_combs(seqs, combs)
                combs = list(combs)
                combs.remove(seqs)
                continue
            else:
                # pickle.dump([result, tc_list, attemp_cnt_list, time.time() - ts, bendset],
                #             open(f'{config.ROOT}/bendplanner/bendresseq/{fo}/{f_name}.pkl', 'wb'))
                return result, tc_list, attemp_cnt_list, time.time() - ts

        print(seqs[:is_success.index(False) + 1])
        combs = bu.remove_combs(seqs[:is_success.index(False) + 1], combs)
        print(seqs)
    attemp_cnt_list.append(cnt)
    # pickle.dump([result, tc_list, attemp_cnt_list, time.time() - ts, bendset],
    #             open(f'{config.ROOT}/bendplanner/bendresseq/{fo}/{f_name}.pkl', 'wb'))
    return result, tc_list, attemp_cnt_list, time.time() - ts


def plan_ipt(bs, bendset, snum=None, fo='180', f_name=''):
    ts = time.time()
    iptree = ip_tree.IPTree(len(bendset))
    valid_tree = ip_tree.IPTree(len(bendset))
    seqs, catch = iptree.get_potential_valid()
    result = []
    tc_list = []
    attemp_cnt_list = []
    cnt = 0
    success_cnt = 0
    while len(seqs) != 0:
        print('Candidate seqs:', seqs)
        bendseq = [bendset[i] for i in seqs]
        cnt_inx, prot_seq = catch
        if cnt_inx != -1:
            print(f'resume from {cnt_inx + 1}')
            bs.reset(prot_seq[0], prot_seq[1])
            is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq[cnt_inx + 1:], cc=True, prune=True, toggledebug=False)
            node_id = ''
            catch_tmp = []
            for inx in range(cnt_inx + 1):
                node_id = node_id + '-' + str(seqs[inx])
                catch_tmp.append(iptree.tree.get_node(node_id).data)
            bendresseq = catch_tmp[::-1] + bendresseq
            is_success = [True] * (cnt_inx + 1) + is_success
        else:
            bs.reset([(0, 0, 0), (0, max([b[-1] for b in bendset]) + .05, 0)], [np.eye(3), np.eye(3)])
            is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=True, prune=True, toggledebug=False)
        print(is_success)
        cnt += 1
        # bs.show_bendresseq(bendresseq, is_success)
        # base.run()
        if all(is_success):
            result.append(seqs)
            tc_list.append(time.time() - ts)
            attemp_cnt_list.append(cnt)
            success_cnt += 1
            if snum is None or success_cnt < snum:
                valid_tree.add_invalid_seq(seqs)
                iptree.add_invalid_seq(seqs, cache_idx=len(is_success),
                                       cache_data=[bendresseq[a][5:] for a in range(len(bendresseq))])
                valid_tree.show()
                seqs, catch = iptree.get_potential_valid()
                continue
            else:
                # pickle.dump([result, tc_list, attemp_cnt_list, time.time() - ts, bendset],
                #             open(f'{config.ROOT}/bendplanner/bendresseq/{fo}/{f_name}.pkl', 'wb'))
                return result, tc_list, attemp_cnt_list, time.time() - ts

        cache_data = [bendresseq[a][5:] for a in range(bendresseq.index([None]))]
        iptree.add_invalid_seq(seqs[:is_success.index(False) + 1],
                               cache_idx=is_success.index(False),
                               # cache_data=bendresseq[bendresseq.index([None]) - 1][5:],
                               cache_data=cache_data)
        # iptree.show()
        seqs, catch = iptree.get_potential_valid()
        print(seqs)
    attemp_cnt_list.append(cnt)
    valid_tree.show()
    # pickle.dump([result, tc_list, attemp_cnt_list, time.time() - ts, bendset],
    #             open(f'{config.ROOT}/bendplanner/bendresseq/{fo}/{f_name}.pkl', 'wb'))
    return result, tc_list, attemp_cnt_list, time.time() - ts


def inf_bend_check(bs, bendset):
    print('-----------influence bend check-----------')
    flag = True

    for i in range(len(bendset) - 1):
        bs.reset([(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)], [np.eye(3), np.eye(3)])
        is_success, bendresseq, fail_reason_list = bs.gen_by_bendseq([bendset[i], bendset[i + 1]], cc=True, prune=True,
                                                                     toggledebug=False)
        print(is_success)
        if 'unbendable' in fail_reason_list:
            flag = False
            break
        bs.reset([(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)], [np.eye(3), np.eye(3)])
        is_success, bendresseq, fail_reason_list = bs.gen_by_bendseq([bendset[i + 1], bendset[i]], cc=True, prune=True,
                                                                     toggledebug=False)
        if 'unbendable' in fail_reason_list:
            flag = False
            break
    print(f'-----------{flag}-----------')

    return flag


if __name__ == '__main__':
    import os
    import localenv.envloader as el
    import BendRbtPlanner as br_planner

    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()

    '''
    init class
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    # base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bs = b_sim.BendSim(show=True)
    f_name = 'randomc'
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, 'bendplanner/goal/pseq', f'{f_name}.pkl'), 'rb'))
    # goal_pseq = np.asarray([[.1, 0, .2], [.1, 0, .1], [0, 0, .1], [0, 0, 0],
    #                         [.1, 0, 0], [.1, .1, 0], [0, .1, 0], [0, .1, .1],
    #                         [.1, .1, .1], [.1, .1, .2]]) / 2

    fit_pseq, _, _ = bu.decimate_pseq(goal_pseq, tor=.002, toggledebug=False)
    bendset = bu.pseq2bendset(fit_pseq, toggledebug=False)[::-1]
    for b in bendset:
        print(b)

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]) + .05, 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp_lft)
    # bs.show(rgba=(.7, .7, .7, .7), objmat4=rm.homomat_from_posrot((0, 0, .1), np.eye(3)))
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)
    bs.reset(init_pseq, init_rotseq)

    # res, tc, attemp_cnt_list, total_tc = plan_ipt(bs, bendset, snum=1)
    res, tc, attemp_cnt_list, total_tc = plan_premutation(bs, bendset, snum=1)
    print(tc, attemp_cnt_list)
    print(total_tc)

    bs.reset(init_pseq, init_rotseq)
    is_success, bendresseq, _ = bs.gen_by_bendseq([bendset[i] for i in res[0]], cc=False, prune=False,
                                                  toggledebug=False)
    # ax = plt.axes(projection='3d')
    # bu.plot_pseq(ax, bs.pseq, c='k')
    # bu.scatter_pseq(ax, bs.pseq, c='r')
    # plt.show()
    bs.show_bendresseq(bendresseq, is_success)

    base.run()
