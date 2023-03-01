import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import visualization.panda.world as wd

import nbv_utils as nbv_utl

if __name__ == '__main__':
    cam_pos = [0, 0, .5]
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'D:/nbv_mesh/'
    if not os.path.exists(path):
        path = 'E:/liu/nbv_mesh/'
    cat = 'bspl_5'
    fo = 'res_75_rlen'

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    cd_rnd, hd_rnd = nbv_utl.fit(path, cat, fo, cross_sec, prefix='random', toggledebug=False)
    cd_org, hd_org = nbv_utl.fit(path, cat, fo, cross_sec, prefix='org')
    cd_pcn, hd_pcn = nbv_utl.fit(path, cat, fo, cross_sec, prefix='pcn')
    cd_opt, hd_opt = nbv_utl.fit(path, cat, fo, cross_sec, prefix='pcn_opt')

    x = [1, 2, 3, 4, 5, 6]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 22))
    # ax1.set_title('Num. of Attempts')
    nbv_utl.plot_box(ax1, cd_rnd, 'tab:blue', positions=[4 * v + .3 for v in x])
    nbv_utl.plot_box(ax1, cd_org, 'tab:orange', positions=[4 * v + 1 + .05 for v in x])
    nbv_utl.plot_box(ax1, cd_pcn, 'tab:green', positions=[4 * v + 2 - .05 for v in x])
    nbv_utl.plot_box(ax1, cd_opt, 'tab:red', positions=[4 * v + 3 - .3 for v in x])

    ax1.set_xticks([4 * v + 1 for v in x])
    ax1.set_xticklabels(x)
    # ax1.set_yticks(np.linspace(.3, 1, 7))
    ax1.minorticks_on()
    # ax1.yaxis.set_major_locator(mticker.MultipleLocator(base=2))
    # ax1.yaxis.set_minor_locator(mticker.MultipleLocator(base=.5))
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    nbv_utl.plot_box(ax2, hd_rnd, 'tab:blue', positions=[4 * v + .3 for v in x], showfliers=True)
    nbv_utl.plot_box(ax2, hd_org, 'tab:orange', positions=[4 * v + 1 + .05 for v in x], showfliers=True)
    nbv_utl.plot_box(ax2, hd_pcn, 'tab:green', positions=[4 * v + 2 - .05 for v in x], showfliers=True)
    nbv_utl.plot_box(ax2, hd_opt, 'tab:red', positions=[4 * v + 3 - .3 for v in x], showfliers=True)

    ax2.set_xticks([4 * v + 1 for v in x])
    ax2.set_xticklabels(x)
    # ax2.set_yticks(np.linspace(.3, 1, 7))
    ax2.minorticks_on()
    # ax2.yaxis.set_major_locator(mticker.MultipleLocator(base=10))
    # ax2.yaxis.set_minor_locator(mticker.MultipleLocator(base=2))
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)
    plt.show()
