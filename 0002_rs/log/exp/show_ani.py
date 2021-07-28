from show_penframe import *
import time


def plot_f(force, path, x, objrelrot=None):
    f_new = []
    if objrelrot is None:
        f_new = force[:, :3]
    else:
        for i, armjnts in enumerate(path):
            if armjnts is not None:
                eepos, eerot = mp_lft.get_ee(armjnts=armjnts)
                baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
                T = np.linalg.inv(np.dot(eerot, np.linalg.inv(objrelrot))).dot(baserot)
            else:
                T = np.eye(3)
            f_new.append(np.dot(T, force[i][:3]))

    f_new = np.asarray(f_new)
    ax1 = fig.add_subplot(111)
    plt.ylim(-6, 8)
    # ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_prop_cycle(color=colors)
    major_ticks = np.arange(x[0], x[-1], 10 if len(x) < 160 else 20)
    minor_ticks = np.arange(x[0], x[-1], 1 if len(x) < 160 else 2)
    hline_c = 'gold'

    for i, f in enumerate(f_new):
        if f[0] < 0:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=0, color=colors[0], alpha=.5, linestyle='dashed')
            print(i)
        if abs(f[1]) > 4.8:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=4.8 if f[1] > 0 else -4.8, color=colors[1], alpha=.5, linestyle='dashed')
            print(i)
        if abs(f[2]) > 6:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=6 if f[2] > 0 else -6, color=colors[2], alpha=.5, linestyle='dashed')
            print(i)
    ax1.axhline(y=4, color=colors[0], alpha=.5, linestyle='dashed')
    ax1.set_title('Force')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=eval(labels[0]), loc='upper left')
    plt.ion()
    plt.show()


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
    exp_id = 'rgt'
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
    info = pickle.load(open(f'./{exp_name}_{exp_id}.pkl', 'rb'))
    fig = plt.figure(1, figsize=(16, 6))

    force = info[0]
    path = info[1]
    force = [[0, 0, 0, 0, 0] for _ in range(100)] + force
    path = [None for _ in range(100)] + path
    for i in range(len(info[0])):
        plt.clf()
        force_tmp = np.asarray(force[i:i + 50])
        path_tmp = np.asarray(path[i:i + 50])
        plot_f(force_tmp, path_tmp, range(i, i + 50), objrelrot)
        plt.pause(0.005)
        time.sleep(.1)
