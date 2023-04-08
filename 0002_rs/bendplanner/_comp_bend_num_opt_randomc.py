import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

import BendOpt as b_opt
import BendOpt_cmaes as b_opt_cmaes
import bend_utils as bu
import bendplanner.BendSim as b_sim
import modeling.geometric_model as gm
import visualization.panda.world as wd

from multiprocessing import Process


def run_parallel(fn, args):
    proc = []
    for arg in args:
        p = Process(target=fn, args=arg)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def opt_process(i, bs, opt, tor=None, obj_type='avg', method='SLSQP'):
    res_bendseq, cost, time_cost = opt.solve(tor=tor, cnt=i, method=method)
    bs.reset(opt.init_pseq, opt.init_rotseq, extend=False)
    bs.gen_by_bendseq(opt.init_bendset, cc=False)
    goal_pseq_aligned, goal_rotseq_aligned = bu.align_with_init(bs, opt.goal_pseq, opt.init_rot, opt.goal_rotseq)
    # bs.show(rgba=(1, 0, 0, 1))
    init_res_pseq = bs.pseq[1:]
    init_err, init_res_kpts = bu.mindist_err(init_res_pseq, goal_pseq_aligned, type=obj_type, toggledebug=False)
    print('org error:', init_err)

    if cost is not None:
        bs.reset(opt.init_pseq, opt.init_rotseq, extend=False)
        bs.gen_by_bendseq(res_bendseq, cc=False)
        _, _ = bu.align_with_init(bs, opt.goal_pseq, opt.init_rot, opt.goal_rotseq)
        # bs.show(rgba=(0, 1, 0, 1))
        opt_res_pseq = bs.pseq[1:]
        opt_err, opt_res_kpts = bu.mindist_err(opt_res_pseq, goal_pseq_aligned, type=obj_type, toggledebug=False)
        print('opt err', opt_err)
    else:
        opt_res_kpts = None
        opt_res_pseq = None
        opt_err = None
    res_dict = {
        'goal_pseq': goal_pseq_aligned,
        'init_bendset': opt.init_bendset,
        'init_err': init_err,
        'init_res_pseq': init_res_pseq,
        'init_res_kpts': init_res_kpts,
        'opt_bendset': res_bendseq,
        'opt_err': opt_err,
        'opt_res_pseq': opt_res_pseq,
        'opt_res_kpts': opt_res_kpts,
        'opt_time_cost': time_cost
    }
    return res_dict


def find_best_n(err_list, threshold=1.):
    min_err = np.inf
    inx = 0
    for i, v in enumerate(err_list):
        if v < min_err:
            min_err = v
            inx = i
        else:
            break
        if v < threshold:
            break
    return inx, min_err


def find_best_n_ploy(err_list, threshold=1.):
    min_err = np.inf
    inx = 0
    for i, v in enumerate(err_list):
        if v < min_err:
            min_err = v
            inx = i
        else:
            break
        if v < threshold:
            break
    return inx, min_err


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')

    '''
    fit init param
    '''
    tor = None
    obj_type = 'avg'
    method = 'SLSQP'

    sigma0 = None

    '''
    opt
    '''
    org_err_list = []
    opt_err_list = []

    f = 'bspl_10'
    res_list = pickle.load(open(f'./bendnum/{f}.pkl', 'rb'))
    try:
        opt_res_list = pickle.load(open(f'./bendnum/{f}_opt_2.pkl', 'rb'))
    except:
        opt_res_list = []

    best_n_list = []
    min_err_list = []
    for i, res in enumerate(res_list):
        if len(opt_res_list) > i:
            print(opt_res_list[i].keys())
            continue
        fit_max_err_list, bend_max_err_list, fit_avg_err_list, bend_avg_err_list, \
        m_list, fit_pseq_list, bend_pseq_list, goal_pseq_list = res
        goal_pseq = goal_pseq_list[i]

        init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
        init_rotseq = [np.eye(3), np.eye(3)]
        # best_n, min_err = find_best_n(bend_avg_err_list, threshold=.5)
        pseq_coarse, _, _ = bu.decimate_pseq_avg(goal_pseq, tor=.0005, toggledebug=False)
        best_n = len(pseq_coarse) - 6
        min_err = bend_avg_err_list[best_n]
        best_n_list.append(best_n + 6)
        min_err_list.append(min_err)

        opt = b_opt.BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq, bend_times=1, obj_type=obj_type)

        opt_res = opt_process(best_n + 6, bs, opt, tor=tor, obj_type=obj_type, method=method)
        opt_res_list.append(opt_res)
        pickle.dump(opt_res_list, open(f'./bendnum/{f}_opt_2.pkl', 'wb'))
