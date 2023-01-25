import os.path
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import visualization.panda.world as wd

import nbv_utils as nbv_utl
import config

if __name__ == '__main__':
    cam_pos = [0, 0, .5]
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'D:/nbv_mesh'
    cat = 'bspl_4'
    fo = 'res_75_rlen'

    load = False

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    if load:
        cd_rnd, hd_rnd, cov_rnd = pickle.load(open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_rnd.pkl')))
        cd_org, hd_org, cov_org = pickle.load(open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_org.pkl')))
        cd_pcn, hd_pcn, cov_pcn = pickle.load(open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_pcn.pkl')))
        cd_opt, hd_opt, cov_opt = pickle.load(open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_opt.pkl')))
    else:
        cd_rnd, hd_rnd, cov_rnd = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, prefix='random', toggledebug=False)
        cd_org, hd_org, cov_org = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, prefix='org')
        cd_pcn, hd_pcn, cov_pcn = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, prefix='pcn')
        cd_opt, hd_opt, cov_opt = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, prefix='pcn_opt')

        # pickle.dump([cd_rnd, hd_rnd, cov_rnd], open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_rnd.pkl')))
        # pickle.dump([cd_org, hd_org, cov_org], open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_org.pkl')))
        # pickle.dump([cd_pcn, hd_pcn, cov_pcn], open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_pcn.pkl')))
        # pickle.dump([cd_opt, hd_opt, cov_opt], open(os.path.join(config.ROOT, 'nbv/res_data', cat, fo, f'cd_opt.pkl')))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 22))
    # ax1.set_title('Num. of Attempts')
    ax1.scatter(cov_rnd, cd_rnd, c='tab:blue', marker='1')
    ax1.scatter(cov_org, cd_org, c='tab:orange', marker='2')
    ax1.scatter(cov_pcn, cd_pcn, c='tab:green', marker='3')
    ax1.scatter(cov_opt, cd_opt, c='tab:red', marker='4')

    # ax1.set_xticks([4 * v + 1 for v in x])
    # ax1.set_xticklabels(x)
    # ax1.set_yticks(np.linspace(.3, 1, 7))
    ax1.minorticks_on()
    # ax1.yaxis.set_major_locator(mticker.MultipleLocator(base=2))
    # ax1.yaxis.set_minor_locator(mticker.MultipleLocator(base=.5))
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    ax2.scatter(cov_rnd, hd_rnd, c='tab:blue', marker='1')
    ax2.scatter(cov_org, hd_org, c='tab:orange', marker='2')
    ax2.scatter(cov_pcn, hd_pcn, c='tab:green', marker='3')
    ax2.scatter(cov_opt, hd_opt, c='tab:red', marker='4')

    # ax2.set_xticks([4 * v + 1 for v in x])
    # ax2.set_xticklabels(x)
    # ax2.set_yticks(np.linspace(.3, 1, 7))
    ax2.minorticks_on()
    # ax2.yaxis.set_major_locator(mticker.MultipleLocator(base=2))
    # ax2.yaxis.set_minor_locator(mticker.MultipleLocator(base=.5))
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    plt.show()
