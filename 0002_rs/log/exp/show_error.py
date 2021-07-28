import pickle

import matplotlib.pyplot as plt
import numpy as np

import config
from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.prj_utils as pu
import utils.run_script_utils as rsu


def plot(exp_name, diff_1, diff_2,f_name="error"):
    fig = plt.figure(1)
    plt.ion()
    plt.show()
    plt.ylim((-10, 10))
    plt.clf()

    x = [i for i in range(len(diff_1))]
    plt.xlim((0, max(x)))
    plt.plot(x, diff_1, label="original")
    plt.plot(x, diff_2, label="refined")

    plt.legend(loc='upper left')
    plt.savefig(f"{f_name}_{exp_name}.png")
    plt.close(fig)


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


def get_diff_list(l1, l2,nn=True):
    diff = []
    l1 = list(l1)
    l2 = list(l2)
    if nn:
        kdt, _ = pu.get_kdt(l1)
        for i, p in enumerate(l2):
            nn = pu.get_knn(p, kdt, k=1)[0]
            diff.append(np.linalg.norm(p - nn))
    else:
        for i, p in enumerate(l1):
            diff.append(np.linalg.norm(p - l2[i]))
    return diff


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex

    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    exp_name = "cylinder_pcd"
    id_list = config.ID_DICT[exp_name]
    folder_name = "/real_exp_" + exp_name + "/"
    phoxi_f_name = "phoxi_tempdata_" + exp_name + ".pkl"
    phoxi_f_name_grasp = "phoxi_tempdata_grasp_" + exp_name + ".pkl"

    pen_cm = el.loadObj(config.PEN_STL_F_NAME)

    '''
    init planner
    '''
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, 'lft')
    motion_planner_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, 'rgt')

    objrelpos, objrelrot, path_cam2place = rsu.load_motion_sgl("cam2place_pen", folder_name, id_list)
    _, _, path_draw = rsu.load_motion_sgl("draw_msd", folder_name, id_list)
    objmat4_cam = motion_planner_lft.load_objmat4(config.PEN_STL_F_NAME.split(".stl")[0], id_list[1])
    grasp = motion_planner_lft.load_grasp(config.PEN_STL_F_NAME.split(".stl")[0], id_list[0])

    transmat = motion_planner_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, config.PEN_STL_F_NAME,
                                                           objmat4_cam, load=True, armjnts=path_cam2place[0],
                                                           toggledubug=False)
    objrelpos_new, objrelrot_new = \
        motion_planner_lft.refine_relpose_by_transmat(objrelpos, objrelrot, np.linalg.inv(transmat))

    '''
    show result
    '''
    path_real_1 = pickle.load(open(f"./{exp_name}_1.pkl", "rb"))[1]
    path_real_2 = pickle.load(open(f"./{exp_name}_2.pkl", "rb"))[1]
    # path_mask_1 = pickle.load(open(f"./{exp_name}_path_mask_1.pkl", "rb"))
    # path_mask_2 = pickle.load(open(f"./{exp_name}_path_mask_2.pkl", "rb"))
    #
    # path_mask = (np.array(path_mask_1) * np.array(path_mask_2)).tolist()
    # print(Counter(path_mask_1), Counter(path_mask_2))
    # print(len(path_draw), len(path_real_1), len(path_real_2))
    # path_draw = path_filter(path_draw, path_mask)
    # path_real_1 = path_padding(path_real_1, path_mask_1)
    # path_real_1 = path_filter(path_real_1, path_mask)
    # path_real_2 = path_padding(path_real_2, path_mask_2)
    # path_real_2 = path_filter(path_real_2, path_mask)
    # print(len(path_draw), len(path_real_1), len(path_real_2))

    pentip_sim_1 = []
    pentip_sim_2 = []
    pentip_real_1 = []
    pentip_real_2 = []

    for i, a in enumerate(path_draw):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        pos = pen_objmat4[:3, 3]
        base.pggen.plotSphere(base.render, pos, rgba=(1, 1, 0, 1))
        pentip_sim_1.append(pos)

        if i in [0, 10]:
            motion_planner_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0))
            motion_planner_lft.ah.show_armjnts(rgba=(1, 1, 0, 0.5), armjnts=a)

    for i, a in enumerate(path_draw):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=a)
        pos = pen_objmat4[:3, 3]
        base.pggen.plotSphere(base.render, pos, rgba=(1, 0, 0, 1))
        pentip_sim_2.append(pos)

        if i in [0, 10]:
            motion_planner_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0))
            motion_planner_lft.ah.show_armjnts(rgba=(1, 0, 0, 0.5), armjnts=a)

    for i, a in enumerate(path_real_1):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        pos = pen_objmat4[:3, 3]
        base.pggen.plotSphere(base.render, pos, rgba=(0, 0, 1, 1))
        pentip_real_1.append(pos)
        if i in [0, 10]:
            motion_planner_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 0, 1))
            motion_planner_lft.ah.show_armjnts(rgba=(0, 0, 1, 0.5), armjnts=a)

    for i, a in enumerate(path_real_2):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        pos = pen_objmat4[:3, 3]
        base.pggen.plotSphere(base.render, pos, rgba=(0, 1, 0, 1))
        pentip_real_2.append(pos)
        if i in [0, 10]:
            motion_planner_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 1, 0))
            motion_planner_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 1, 0))

    # rmse, fitness, transmat_1 = o3dh.registration_ptpt(np.asarray(pentip_sim), np.asarray(pentip_real_1),
    #                                                    toggledebug=False)
    # print(rmse, fitness)
    # rmse, fitness, transmat_2 = o3dh.registration_ptpt(np.asarray(pentip_sim), np.asarray(pentip_real_2),
    #                                                    toggledebug=False)
    # print(rmse, fitness)
    # pentip_real_1 = pcdu.transform_pcd(np.asarray(pentip_real_1), transmat_1)
    # pentip_real_2 = pcdu.transform_pcd(np.asarray(pentip_real_2), transmat_2)

    diff_1 = get_diff_list(pentip_sim_1, pentip_real_1, nn=True)
    diff_2 = get_diff_list(pentip_sim_2, pentip_real_2, nn=True)
    plot(exp_name, diff_1, diff_2, f_name="error_2")

    diff_1 = get_diff_list(pentip_sim_2, pentip_real_1, nn=True)
    diff_2 = get_diff_list(pentip_sim_2, pentip_real_2, nn=True)
    plot(exp_name, diff_1, diff_2, f_name="error_1")
    # base.run()
