import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import bendplanner.bend_utils as bu
import datagenerator.data_utils as du
import utils.pcd_utils as pcdu
import visualization.panda.world as wd
import matplotlib.ticker as mticker

import nbv_utils as nbv_utl

if __name__ == '__main__':
    cam_pos = [0, 0, .5]
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'D:/nbv_mesh'
    cat = 'bspl'
    fo = 'res_75'

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    cd_rnd, hd_rnd = nbv_utl.load_pts(path, cat, fo, cross_sec, prefix='random')
    cd_org, hd_org = nbv_utl.load_pts(path, cat, fo, cross_sec, prefix='org')
    cd_pcn, hd_pcn = nbv_utl.load_pts(path, cat, fo, cross_sec, prefix='pcn')

    cd_rnd = nbv_utl.transpose(cd_rnd)
    cd_org = nbv_utl.transpose(cd_org)
    cd_pcn = nbv_utl.transpose(cd_pcn)

    x = [0, 1, 2, 3, 4]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 22))
    ax1.set_title('Num. of Attempts')
    nbv_utl.plot_box(ax1, cd_rnd, 'tab:blue', positions=[3 * v + .25 for v in x])
    nbv_utl.plot_box(ax1, cd_org, 'tab:orange', positions=[3 * v + 1 for v in x])
    nbv_utl.plot_box(ax1, cd_pcn, 'tab:green', positions=[3 * v + 2 - .25 for v in x])
    ax1.set_xticks([3 * v + 1 for v in x])
    ax1.set_xticklabels(x)
    ax1.set_yticks(np.linspace(.3, 1, 7))

    ax1.minorticks_on()
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(base=1 / 5))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(base=1 / 20))

    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=.8)
    ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=.5)
    plt.show()
