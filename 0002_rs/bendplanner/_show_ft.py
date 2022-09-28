import pickle

import matplotlib.pyplot as plt
import numpy as np
import config
import motionplanner.motion_planner as m_planner
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import basis.robot_math as rm
from localenv import envloader as el
import modeling.geometric_model as gm


def load_ft(f_name, cnt):
    mp_lft = m_planner.MotionPlanner(env, rbt, 'lft_arm')
    armjnts = pickle.load(open(f'bend_ft/{f_name}_armjnts{cnt}.pkl', 'rb'))
    ft_data = pickle.load(open(f'bend_ft/{f_name}_ft{cnt}.pkl', 'rb'))
    armjnts = np.radians(armjnts)
    print(armjnts)
    mp_lft.ah.show_armjnts(armjnts=armjnts, toggleendcoord=True, togglejntcoord=True)
    f_new = []
    t_new = []
    for i, ft in enumerate(ft_data):
        eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
        baserot = np.dot(rm.rotmat_from_axangle((1, 0, 0), -np.pi / 2), rm.rotmat_from_axangle((0, 1, 0), np.pi / 2))
        gm.gen_frame(pos=np.asarray([1, 1, 1]), rotmat=baserot).attach_to(base)
        T = np.linalg.inv(eerot).dot(baserot)
        f_new.append(np.dot(T, ft[:3]))
        t_new.append(np.dot(T, ft[3:]))
    return f_new, t_new


def plot(ax1, ax2, f, t):
    x = range(len(f))
    f = np.asarray(f)
    t = np.asarray(t)
    major_ticks = np.arange(0, len(f), 20)
    minor_ticks = np.arange(0, len(f), 5)
    ax1.plot(x, f[:, 0], label='Fx', c='tab:red')
    ax1.plot(x, f[:, 1], label='Fy', c='tab:green')
    ax1.plot(x, f[:, 2], label='Fz', c='tab:blue')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.set_title('Force')
    ax1.legend(('Fx', 'Fy', 'Fz'))

    ax2.plot(x, t[:, 0], label='Rx', c='tab:red')
    ax2.plot(x, t[:, 1], label='Ry', c='tab:green')
    ax2.plot(x, t[:, 2], label='Rz', c='tab:blue')
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(which='minor', linestyle='dotted', alpha=0.5)
    ax2.grid(which='major', linestyle='dotted', alpha=1)
    ax2.set_title('Torque')
    ax2.legend(('Rx', 'Ry', 'Rz'))


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()

    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    f_name = 'alu'
    '''
    init planner
    '''
    fig = plt.figure(1, figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    f, t = load_ft(f_name, '')
    plot(ax1, ax2, f, t)
    f, t = load_ft(f_name, '2')
    plot(ax1, ax2, f, t)
    f, t = load_ft(f_name, '3')
    plot(ax1, ax2, f, t)

    plt.show()

    base.run()
