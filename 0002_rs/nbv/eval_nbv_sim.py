import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np

import nbv_utils as nbv_utl

if __name__ == '__main__':
    # path = 'E:/liu/nbv_mesh/'
    path = 'D:/nbv_mesh/'
    cat = 'plat'
    fo = 'res_60_rlen'
    coverage_rnd, max_rnd, cnt_rnd = nbv_utl.load_cov(path, cat, fo, prefix='random')
    coverage_org, max_org, cnt_org = nbv_utl.load_cov(path, cat, fo, prefix='org')
    coverage_pcn, max_pcn, cnt_pcn = nbv_utl.load_cov(path, cat, fo, prefix='pcn')
    coverage_opt, max_opt, cnt_opt = nbv_utl.load_cov(path, cat, fo, prefix='pcn_opt')

    x = np.asarray([1, 2, 3, 4, 5, 6])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 22))
    # ax1.set_title('Coverage')
    ax1.axhline(y=.95, color='r', linewidth='0.5', linestyle=':')
    # ax2.set_title('Num. of Captures')

    nbv_utl.plot_box(ax1, max_rnd, 'tab:blue', positions=[4 * v + .3 for v in x], showfliers=True)
    nbv_utl.plot_box(ax1, max_org, 'tab:orange', positions=[4 * v + 1 + .05 for v in x], showfliers=True)
    nbv_utl.plot_box(ax1, max_pcn, 'tab:green', positions=[4 * v + 2 - .05 for v in x], showfliers=True)
    nbv_utl.plot_box(ax1, max_opt, 'tab:red', positions=[4 * v + 3 - .3 for v in x], showfliers=True)

    # nbv_utl.plot_box(ax1, coverage_org, 'tab:blue', positions=[4 * v + .3 for v in x], showfliers=True)
    # nbv_utl.plot_box(ax1, coverage_org, 'tab:orange', positions=[4 * v + 1 + .05 for v in x], showfliers=True)
    # nbv_utl.plot_box(ax1, coverage_pcn, 'tab:green', positions=[4 * v + 2 - .05 for v in x], showfliers=True)
    # nbv_utl.plot_box(ax1, coverage_opt, 'tab:red', positions=[4 * v + 3 - .3 for v in x], showfliers=True)

    ax1.set_xticks([4 * v + 1.5 for v in x])
    ax1.set_xticklabels(x)
    ax1.set_yticks(np.linspace(.3, 1, 7))

    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(base=1 / 5))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(base=1 / 20))
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    ax2.bar(x[1:] - .3, cnt_rnd[1:], color='tab:blue', width=0.2)
    ax2.bar(x[1:] - .1, cnt_org[1:], color='tab:orange', width=0.2)
    ax2.bar(x[1:] + .1, cnt_pcn[1:], color='tab:green', width=0.2)
    ax2.bar(x[1:] + .3, cnt_opt[1:], color='tab:red', width=0.2)

    ax2.set_xticks(x[1:])
    # ax2.set_yticks(np.linspace(0, 100, 10))
    ax2.minorticks_on()
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(base=100 / 10))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(base=100 / 50))
    ax2.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)

    print(nbv_utl.cal_avg(cnt_rnd[1:]))
    print(nbv_utl.cal_avg(cnt_org[1:]))
    print(nbv_utl.cal_avg(cnt_pcn[1:]))
    print(nbv_utl.cal_avg(cnt_opt[1:]))

    plt.show()
    # base.run()
