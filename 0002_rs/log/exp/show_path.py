import pickle

import numpy as np

import config
from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.run_script_utils as rsu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl


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


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex

    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    exp_name = "cylinder_cad"
    exp_id = "2"
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

    # paintingobj_info = \
    #     ru.get_obj_from_phoxiinfo_nobgf(phxilocator, phoxi_f_name=phoxi_f_name, load=True, reconstruct_surface=True,
    #                                     x_range=(600, 1080), y_range=(-100, 300), z_range=(790, 1000))
    # pcdu.show_pcd(paintingobj_info["pcd"], colors=(1, 1, 1, .5))

    transmat = motion_planner_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, config.PEN_STL_F_NAME,
                                                           objmat4_cam, load=True, armjnts=path_cam2place[0],
                                                           toggledubug=False)

    _, _, _, path_draw_refine, path_mask_refine = \
        motion_planner_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_draw, grasp, pen_cm, transmat)

    objrelpos_new, objrelrot_new = \
        motion_planner_lft.refine_relpose_by_transmat(objrelpos, objrelrot, np.linalg.inv(transmat))

    '''
    show result
    '''
    path_real = pickle.load(open(f"./{exp_name}_{exp_id}.pkl", "rb"))[1]
    # path_mask_real = pickle.load(open(f"./{exp_name}_path_mask_{exp_id}.pkl", "rb"))
    # path_mask = (np.array(path_mask_refine) * np.array(path_mask_real)).tolist()
    # print(len(path_draw), len(path_draw_refine), len(path_real))
    #
    # print(Counter(path_mask_real))
    # print(Counter(path_mask))
    #
    # path_draw_refine = path_padding(path_draw_refine, path_mask_refine)
    # path_real = path_padding(path_real, path_mask_real)
    #
    # path_draw_refine = path_filter(path_draw_refine, path_mask)
    # path_real = path_filter(path_real, path_mask)
    # print(len(path_draw), len(path_draw_refine), len(path_real))

    for i, a in enumerate(path_draw):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 1, 0, 1))
        if i == 1:
            motion_planner_lft.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 1, 0))
            motion_planner_lft.show_armjnts(rgba=(1, 1, 0, 0.5), armjnts=a, jawwidth=15)

    for i, a in enumerate(path_draw_refine):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 0, 0, 1))
        if i == 1:
            motion_planner_lft.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0))
            motion_planner_lft.show_armjnts(rgba=(1, 0, 0, 0.5), armjnts=a, jawwidth=15)

    for i, a in enumerate(path_real):
        pen_objmat4 = motion_planner_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(0, 1, 0, 1))
        if i == 1:
            motion_planner_lft.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 1, 0))
            motion_planner_lft.show_armjnts(rgba=(0, 1, 0, 0.5), armjnts=a, jawwidth=15)

    base.run()
