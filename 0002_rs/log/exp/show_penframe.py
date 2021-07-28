import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
# from scipy.ndimage.filters import uniform_filter1d
import config
import motionplanner.motion_planner as m_planner
import run_plan.run_config as rconfig
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utiltools.robotmath as rm
from localenv import envloader as el

colors = ['#d62728', '#2ca02c', '#1f77b4']


def path_padding(path, mask):
    path_new = []
    path_index = 0
    for i, v in enumerate(mask):
        if v:
            try:
                path_new.append(path[path_index])
                path_index += 1
            except:
                path_new.append(None)
        else:
            path_new.append(None)
    return path_new


def path_filter(path, mask):
    return [a for i, a in enumerate(path) if mask[i]]


def plot_all(info, objrelrot=None):
    if objrelrot is None:
        f_name = f'./{exp_name}_{exp_id}.png'
    else:
        f_name = f'./{exp_name}_{exp_id}_penframe.png'

    force = np.asarray(info[0])
    path = np.asarray(info[1])
    speed = np.asarray(info[2])
    diff = np.asarray(info[3])
    tcppose = np.asarray(info[4])

    tcprot = [l[3:] for l in tcppose]
    diff = [d for d in diff]

    f_new = []
    t_new = []
    s_new = []
    as_new = []

    if objrelrot is None:
        f_new = force[:, :3]
        t_new = force[:, 3:]
        s_new = speed[:, 3:]
        as_new = speed[:, 3:]
    else:
        for i, armjnts in enumerate(path):
            eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
            baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
            f_new.append(np.dot(T, force[i][:3]))
            t_new.append(np.dot(T, force[i][3:]))
            s_new.append(np.dot(T, speed[i][:3]))
            as_new.append(np.dot(T, speed[i][3:]))

    x = [i for i in range(len(f_new))]
    f_new = np.asarray(f_new)
    t_new = np.asarray(t_new)

    fx = gaussian_filter1d(f_new[:, 0], 3)
    fy = gaussian_filter1d(f_new[:, 1], 3)
    fz = gaussian_filter1d(f_new[:, 2], 3)

    tx = gaussian_filter1d(t_new[:, 0], 3)
    ty = gaussian_filter1d(t_new[:, 1], 3)
    tz = gaussian_filter1d(t_new[:, 2], 3)

    fig = plt.figure(1, figsize=(16, 9))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    major_ticks = np.arange(0, len(f_new), 10)
    minor_ticks = np.arange(0, len(f_new), 1)

    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    # ax1.plot(x, fx, label=['Fx'])
    # ax1.plot(x, fy, label=['Fy'])
    # ax1.plot(x, fz, label=['Fz'])
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.set_title('Force')
    ax1.legend(('Fx', 'Fy', 'Fz'))

    ax2.plot(x, t_new, label=['Rx', 'Ry', 'Rz'])
    # ax2.plot(x, tx, label=['Rx'])
    # ax2.plot(x, ty, label=['Ry'])
    # ax2.plot(x, tz, label=['Rz'])
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax2.grid(which='major', linestyle='dotted', alpha=1)
    ax2.set_title('Torque')
    ax2.legend(('Rx', 'Ry', 'Rz'))

    ax3.plot(x, diff, label='diff')
    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)
    ax3.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax3.grid(which='major', linestyle='dotted', alpha=1)
    ax3.set_title('TCP deviation')
    ax3.legend()

    ax4.plot(x, s_new, label=['x', 'y', 'z'])
    ax4.set_xticks(major_ticks)
    ax4.set_xticks(minor_ticks, minor=True)
    ax4.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax4.grid(which='major', linestyle='dotted', alpha=1)
    ax4.set_title('Speed')
    ax4.legend(('x', 'y', 'z'))

    ax5.plot(x, as_new, label=['Rx', 'Ry', 'Rz'])
    ax5.set_xticks(major_ticks)
    ax5.set_xticks(minor_ticks, minor=True)
    ax5.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax5.grid(which='major', linestyle='dotted', alpha=1)
    ax5.set_title('Speed')
    ax5.legend(('Rx', 'Ry', 'Rz'))

    ax6.set_xticks(major_ticks)
    ax6.set_xticks(minor_ticks, minor=True)
    ax6.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax6.grid(which='major', linestyle='dotted', alpha=1)
    ax6.set_title('TCP rotation')
    ax6.plot(x, tcprot, label=['Rx', 'Ry', 'Rz'])

    plt.savefig(f_name)
    plt.close(fig)


def plot_part(info, objrelrot=None, angle_list=None):
    if objrelrot is None:
        f_name = f'./{exp_name}_{exp_id}_part.png'
    else:
        f_name = f'./{exp_name}_{exp_id}_penframe_part.png'

    force = np.asarray(info[0])
    path = np.asarray(info[1])
    diff = np.asarray(info[3])
    tcppose = np.asarray(info[4])

    tcprot = [np.degrees(l[3:]) for l in tcppose]
    diff = [d for d in diff]

    f_new = []
    if objrelrot is None:
        f_new = force[:, :3]
    else:
        for i, armjnts in enumerate(path):
            # tcprotmat = rm.rotmat_from_euler(np.degrees(tcprot[i][0]),
            #                                  np.degrees(tcprot[i][1]),
            #                                  np.degrees(tcprot[i][2]))
            # baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            # mp_lft.ah.show_armjnts(armjnts=armjnts, toggleendcoord=True)
            # base.pggen.plotAxis(base.render,
            #                     spos=np.asarray(baserot).dot(np.asarray(tcppose[i][:3]) * 1000) +
            #                          mp_lft.rbt.lftarm[1]['linkpos'],
            #                     srot=np.asarray(baserot.dot(tcprotmat)))
            # base.pggen.plotAxis(base.render, spos=mp_lft.rbt.lftarm[1]['linkpos'], srot=np.asarray(baserot))
            # base.run()
            # T = objrelrot.dot(np.linalg.inv(tcprotmat))

            eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
            baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
            f_new.append(np.dot(T, force[i][:3]))

    f_new = np.asarray(f_new)
    x = [i for i in range(len(f_new))]
    fig = plt.figure(1, figsize=(9, 24))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.set_prop_cycle(color=colors)
    ax3.set_prop_cycle(color=colors)
    major_ticks = np.arange(0, len(f_new), 10 if len(f_new) < 160 else 20)
    minor_ticks = np.arange(0, len(f_new), 1 if len(f_new) < 160 else 2)
    hline_c = 'gold'

    flag = True
    for i, f in enumerate(f_new):
        if f[0] < 0:
            ax1.axvline(x=i, color=hline_c, alpha=1)
            ax2.axvline(x=i, color=hline_c, alpha=1)
            ax3.axvline(x=i, color=hline_c, alpha=1)
            ax1.axhline(y=0, color=colors[0], alpha=.5, linestyle='dashed')
            print(i)
            flag = False
        if abs(f[1]) > 4.8:
            ax1.axvline(x=i, color=hline_c, alpha=1)
            ax2.axvline(x=i, color=hline_c, alpha=1)
            ax3.axvline(x=i, color=hline_c, alpha=1)
            ax1.axhline(y=4.8, color=colors[1], alpha=.5, linestyle='dashed')
            print(i)
            flag = False
        if abs(f[2]) > 6:
            ax1.axvline(x=i, color=hline_c, alpha=1)
            ax2.axvline(x=i, color=hline_c, alpha=1)
            ax3.axvline(x=i, color=hline_c, alpha=1)
            ax1.axhline(y=6, color=colors[2], alpha=.5, linestyle='dashed')
            print(i)
            flag = False

    if flag:
        ax1.axhline(y=np.mean(f_new[:, 0]), color=colors[0], alpha=.5, linestyle='dashed')
        ax1.axhspan(np.mean(f_new[:, 0]) - np.std(f_new[:, 0]), np.mean(f_new[:, 0]) + np.std(f_new[:, 0]),
                    facecolor=colors[0], alpha=.1)
        ax1.axhline(y=np.mean(f_new[:, 1]), color=colors[1], alpha=.5, linestyle='dashed')
        ax1.axhspan(np.mean(f_new[:, 1]) - np.std(f_new[:, 1]), np.mean(f_new[:, 1]) + np.std(f_new[:, 1]),
                    facecolor=colors[1], alpha=.1)
        ax1.axhline(y=np.mean(f_new[:, 2]), color=colors[2], alpha=.5, linestyle='dashed')
        ax1.axhspan(np.mean(f_new[:, 2]) - np.std(f_new[:, 2]), np.mean(f_new[:, 2]) + np.std(f_new[:, 2]),
                    facecolor=colors[2], alpha=.1)

    # ax1.set_title('Force')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=eval(labels[0]))

    if angle_list is not None:
        # ax2.set_title('Force deviation angle')
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', linestyle='dotted', alpha=0.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        # ax2.plot(x, uniform_filter1d(angle_list, size=4), alpha=.5, color='k', linestyle='dashed')
        ax2.plot(x, angle_list)
        ax2.axhline(y=np.mean(angle_list), alpha=.5, linestyle='dashed')
        ax2.axhspan(np.mean(angle_list) - np.std(angle_list), np.mean(angle_list) + np.std(angle_list), alpha=.1)
        # ax2.axhline(y=12.826, alpha=.5, linestyle='dashed')
        # ax2.axhspan(12.826 - 6.4362, 12.826 + 6.4362, alpha=.1)

        # 12.826148924585945
        # 6.436285715428524
        print(np.mean(angle_list), np.std(angle_list))
        ax2.legend()
    else:
        # ax2.set_title('TCP deviation')
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', linestyle='dotted', alpha=0.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        ax2.plot(x, diff)
        ax2.legend()

    # ax3.set_title('TCP rotation')
    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)
    ax3.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax3.grid(which='major', linestyle='dotted', alpha=1)
    ax3.plot(x, tcprot, label=['Rx', 'Ry', 'Rz'])
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles=handles, labels=eval(labels[0]))

    plt.savefig(f_name)
    plt.close(fig)


def plot_compare(info_1, info_2, objrelrot=None):
    f_name = f'./{exp_name}_{exp_id}_compare.png'
    force1 = np.asarray(info_1[0])
    path1 = np.asarray(info_1[1])

    force2 = np.asarray(info_2[0])
    path2 = np.asarray(info_2[1])

    f_new_1 = []
    f_new_2 = []
    if objrelrot is None:
        f_new_1 = force1[:, :3]
        f_new_2 = force2[:, :3]
    else:
        for i, armjnts in enumerate(path1):
            eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
            baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
            f_new_1.append(np.dot(T, force1[i][:3]))
        for i, armjnts in enumerate(path2):
            eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
            baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
            f_new_2.append(np.dot(T, force2[i][:3]))

    f_new_1 = np.asarray(f_new_1)
    f_new_2 = np.asarray(f_new_2)
    x = [i for i in range(len(f_new_1))]
    fig = plt.figure(1, figsize=(9, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    # ax1.set_prop_cycle(color=colors)
    major_ticks = np.arange(0, len(f_new_1), 10 if len(f_new_1) < 160 else 20)
    minor_ticks = np.arange(0, len(f_new_1), 1 if len(f_new_1) < 160 else 2)

    ax1.set_title('Force')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new_1[:, 0])
    ax1.plot(x, f_new_2[:, 0])
    ax1.axhline(y=np.mean(f_new_1[:, 0]), alpha=.5, linestyle='dashed')
    ax1.axhspan(np.mean(f_new_1[:, 0]) - np.std(f_new_1[:, 0]), np.mean(f_new_1[:, 0]) + np.std(f_new_1[:, 0]),
                alpha=.1)
    # handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(handles=handles, labels=eval(labels[0]))
    plt.show()

    # plt.savefig(f_name)
    # plt.close(fig)


def draw_motion(mp, info, obj_item, objrelpos, objrelrot):
    force = info[0]
    path = info[1]
    angle_list = []
    for i, armjnts in enumerate(path):
        baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
        f = np.dot(baserot, force[i][:3])
        objmat4 = mp.get_world_objmat4(objrelpos, objrelrot, armjnts)
        objpos = objmat4[:3, 3]
        objrot = objmat4[:3, :3]
        # f = np.dot(objrot, force[i][:3])
        angle = np.degrees(rm.angle_between_vectors(f, objrot[:3, 0]))
        angle_list.append(angle)
        eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
        T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
        f_tool = np.dot(T, force[i][:3])
        if abs(f_tool[1]) > 4.8 or abs(f_tool[2]) > 6 or f_tool[0] < 0:
            base.pggen.plotSphere(base.render, objpos, rgba=(1, 0, 0, 1))
            base.pggen.plotArrow(base.render, spos=objpos, epos=objpos + np.asarray(f) * 3,
                                 rgba=(1, 0, 0, .5), thickness=1)
        else:
            base.pggen.plotSphere(base.render, objpos, rgba=(0, 1, 0, 1))
            base.pggen.plotArrow(base.render, spos=objpos, epos=objpos + np.asarray(f) * 3,
                                 rgba=(0, 1, 0, .5), thickness=1)

        # base.pggen.plotArrow(base.render, spos=objpos, epos=objpos + np.asarray(objmat4[:3, 0]) * 10,
        #                      rgba=(0, 1, 0, .5), thickness=1)
        # obj_item.set_objmat4(objmat4)
        obj_item.show_objcm(show_localframe=True)
    # mp.ah.show_animation_hold(path, obj_item.objcm, objrelpos, objrelrot)

    return angle_list


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()

    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    item = el.loadObjitem('cube', pos=rconfig.OBJ_POS['cube'])
    item.show_objcm(rgba=(1, 1, 1, .5))

    exp_name = 'cube'
    exp_id = 'rgt_org'

    id_list = config.ID_DICT[exp_name]
    folder_name = '/exp_' + exp_name + '/'

    pen_item = el.loadObjitem(config.PEN_STL_F_NAME)
    # pen_cm.reparentTo(base.render)
    # base.run()

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, 'lft')

    motion_dict = pickle.load(open(f'{config.MOTIONSCRIPT_REL_PATH}/{folder_name}/draw_circle.pkl', 'rb'))
    objrelpos, objrelrot, path_draw = motion_dict[id_list[0]]
    grasp = mp_lft.load_grasp(config.PEN_STL_F_NAME.split('.stl')[0], id_list[0])
    _, _, hndmat4 = grasp
    # objrelpos, objrelrot, path_cam2place = pickle.load(
    #     open(config.MOTIONSCRIPT_REL_PATH + 'exp_force/draw_L.pkl', 'rb'))
    # objrelpos, objrelrot, path_cam2place = pickle.load(
    #     open(config.MOTIONSCRIPT_REL_PATH + 'exp_cube/draw_circle.pkl', 'rb'))

    '''
    show result
    '''
    # fig = plt.figure(1, figsize=(16, 9))
    # plt.clf()
    info = pickle.load(open(f'./{exp_name}_{exp_id}.pkl', 'rb'))

    angle_list = draw_motion(mp_lft, info, pen_item, objrelpos, np.eye(3))
    # base.run()
    # plot_all(info)
    plot_part(info, angle_list=angle_list)

    # plot_all(info, objrelrot)
    # plot_part(info, objrelrot, angle_list=angle_list)
    # base.run()

    # info2 = pickle.load(open(f'./{exp_name}_f.pkl', 'rb'))
    # plot_compare(info, info2, objrelrot)
