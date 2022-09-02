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


def opt_process(i, bs, opt, tor=None, obj_type='avg', method='SLSQP', n_trials=2000, n_startup_trials=10, sigma0=None):
    if method == 'cmaes':
        res_bendseq, cost, time_cost = opt.solve(tor=tor, cnt=i,
                                                 n_trials=n_trials, sigma0=sigma0,
                                                 n_startup_trials=n_startup_trials)
    else:
        res_bendseq, cost, time_cost = opt.solve(tor=tor, cnt=i,
                                                 method=method)
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
    dump_dict[i] = {
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
    pickle.dump(dump_dict, open(f'bendopt/{f_name}', 'wb'))
    return init_err, opt_err


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(thickness=.0005, alpha=.1, length=.01).attach_to(base)
    bs = b_sim.BendSim(show=True, granularity=np.pi / 90, cm_type='stick')
    goal_f_name = 'randomc'
    goal_pseq = pickle.load(open(f'goal/pseq/{goal_f_name}.pkl', 'rb'))
    goal_rotseq = None
    # goal_pseq, goal_rotseq = pickle.load(open('../data/goal/rotpseq/skull2.pkl', 'rb'))

    init_pseq = [(0, 0, 0), (0, .05 + bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    '''
    fit init param
    '''
    tor = None
    obj_type = 'avg'
    method = 'SLSQP'
    # method = 'cmaes'

    n_trials = 2000
    n_startup_trials = 10
    sigma0 = None

    f_name = f'{goal_f_name}_{method}_{obj_type}.pkl'

    '''
    opt
    '''
    org_err_list = []
    opt_err_list = []
    if os.path.isfile(f'bendopt/{f_name}'):
        dump_dict = pickle.load(open(f'bendopt/{f_name}', 'rb'))
    else:
        dump_dict = {}
    print(dump_dict.keys())

    if method == 'cmaes':
        opt = b_opt_cmaes.BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq,
                                        goal_rotseq=goal_rotseq,
                                        bend_times=1,
                                        obj_type=obj_type)
    else:
        opt = b_opt.BendOptimizer(bs, init_pseq, init_rotseq, goal_pseq,
                                  goal_rotseq=goal_rotseq,
                                  bend_times=1,
                                  obj_type=obj_type)

    for i in range(20, 30):
        init_err, opt_err = opt_process(i, bs, opt, tor=tor, obj_type=obj_type, method=method,
                                        n_trials=n_trials, n_startup_trials=n_startup_trials, sigma0=sigma0)

    # run_parallel(opt_process, [[19, bs, opt, tor, obj_type, method, n_trials, n_startup_trials, sigma0]])
    # run_parallel(opt_process, [[20, bs, opt, tor, obj_type, method, n_trials, n_startup_trials, sigma0]])

