import os.path
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import visualization.panda.world as wd

import nbv_utils as nbv_utl
import config


def filter(cd_list, hd_list, cov_list, cd_lim=10, hd_lim=20):
    inx = np.where(np.asarray(cd_list) < cd_lim, 1, 0) & np.where(np.asarray(hd_list) < hd_lim, 1, 0)

    return [v for i, v in enumerate(cd_list) if inx[i]], \
           [v for i, v in enumerate(hd_list) if inx[i]], \
           [v for i, v in enumerate(cov_list) if inx[i]]


if __name__ == '__main__':
    cam_pos = [0, 0, .5]
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    # path = 'D:/nbv_mesh'
    path = 'E:/liu/nbv_mesh/'
    cat_list = ['bspl_3', 'bspl_4', 'bspl_5']
    # cat = 'tmpl'
    kpts_num = 16
    fo = 'res_75_rlen'
    res_fo = 'res_data'
    load = False

    for cat in cat_list:
        res_path = os.path.join(config.ROOT, f'nbv/{res_fo}', cat, fo)
        if not os.path.exists(res_path):
            load = False
            # os.mkdir(os.path.join(config.ROOT, f'nbv/{res_fo}'))
            os.mkdir(os.path.join(config.ROOT, f'nbv/{res_fo}', cat))
            os.mkdir(os.path.join(config.ROOT, f'nbv/{res_fo}', cat, fo))

        width = .008
        thickness = .0015
        cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
        if load:
            cd_rnd, hd_rnd, cov_rnd = pickle.load(open(os.path.join(res_path, f'cd_rnd.pkl'), 'rb'))
            cd_org, hd_org, cov_org = pickle.load(open(os.path.join(res_path, f'cd_org.pkl'), 'rb'))
            cd_pcn, hd_pcn, cov_pcn = pickle.load(open(os.path.join(res_path, f'cd_pcn.pkl'), 'rb'))
            cd_opt, hd_opt, cov_opt = pickle.load(open(os.path.join(res_path, f'cd_opt.pkl'), 'rb'))
        else:
            cd_rnd, hd_rnd, cov_rnd = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, kpts_num=kpts_num, prefix='random',
                                                           toggledebug=True)
            cd_org, hd_org, cov_org = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, kpts_num=kpts_num, prefix='org')
            cd_pcn, hd_pcn, cov_pcn = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, kpts_num=kpts_num, prefix='pcn')
            cd_opt, hd_opt, cov_opt = nbv_utl.fit_dist_cov(path, cat, fo, cross_sec, kpts_num=kpts_num, prefix='pcn_opt')

            pickle.dump([cd_rnd, hd_rnd, cov_rnd], open(os.path.join(res_path, f'cd_rnd.pkl'), 'wb'))
            pickle.dump([cd_org, hd_org, cov_org], open(os.path.join(res_path, f'cd_org.pkl'), 'wb'))
            pickle.dump([cd_pcn, hd_pcn, cov_pcn], open(os.path.join(res_path, f'cd_pcn.pkl'), 'wb'))
            pickle.dump([cd_opt, hd_opt, cov_opt], open(os.path.join(res_path, f'cd_opt.pkl'), 'wb'))

        cd_rnd, hd_rnd, cov_rnd = filter(cd_rnd, hd_rnd, cov_rnd)
        cd_org, hd_org, cov_org = filter(cd_org, hd_org, cov_org)
        cd_pcn, hd_pcn, cov_pcn = filter(cd_pcn, hd_pcn, cov_pcn)
        cd_opt, hd_opt, cov_opt = filter(cd_opt, hd_opt, cov_opt)

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
