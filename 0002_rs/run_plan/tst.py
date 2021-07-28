import pickle

import numpy as np

import config
from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.run_utils as ru
import utils.phoxi as phoxi
import utils.phoxi_locator as pl

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)

    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    exp_name = "cube"
    id_list = config.ID_DICT[exp_name]
    folder_name = "/motionscript/real_exp_" + exp_name + "/"
    phoxi_f_name = "phoxi_tempdata_" + exp_name + ".pkl"
    phoxi_f_name_grasp = "phoxi_tempdata_grasp_" + exp_name + ".pkl"

    pen_cm = el.loadObj(config.PEN_STL_F_NAME)

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, 'lft')
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, 'rgt')

    paintingobj_info = \
        ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_name, load=True, reconstruct_surface=True,
                            x_range=(600, 1080), y_range=(-100, 300), z_range=(790, 1000))

    objmat4_cam = mp_lft.load_objmat4(config.PEN_STL_F_NAME.split(".stl")[0], id_list[1])
    grasp = mp_lft.load_grasp(config.PEN_STL_F_NAME.split(".stl")[0], id_list[0])

    path_gotopick = pickle.load(open(config.ROOT + folder_name + "gotopick_pen.pkl", "rb"))
    path_cam2place = pickle.load(open(config.ROOT + folder_name + "cam2place_pen.pkl", "rb"))

    objrelpos, objrelrot, path_cam2place = path_cam2place[id_list[0]][id_list[1]]
    _, _, path_gotopick = path_gotopick[id_list[0]]

    tcp_pos, tcp_rot = mp_lft.get_ee()

    transmat = mp_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, config.PEN_STL_F_NAME,
                                               objmat4_cam, load=True, armjnts=path_cam2place[0],
                                               toggledubug=False)
    # mp_lft.show_armjnts(armjnts=path_cam2place[0])
    # transmat = pickle.load(open("transmat_temp.pkl", "rb"))

    # objmat4_init = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=path_gotopick[-1])
    # mp_lft.show_objmat4(pen_cm, objmat4_init, color=(1, 1, 0))
    # mp_lft.show_objmat4(pen_cm, np.dot(objmat4_init, np.linalg.inv(transmat)), color=(1, 0, 0))

    _, _, path_draw = pickle.load(open(config.ROOT + folder_name + "draw_msd.pkl", "rb"))[id_list[0]]
    _, _, _, path_draw_new, path_mask = \
        mp_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_draw, grasp, pen_cm, transmat)

    # primitive_armjnts = mp_lft.get_tool_primitive_armjnts(path_draw[0], objrelrot)
    # mp_lft.show_armjnts(rgba=(0, 1, 1, 0.5), armjnts=primitive_armjnts)
    # path_gotodraw = mp_lft.plan_start2end(end=primitive_armjnts, start=path_cam2place[-1])
    # mp_lft.show_animation(path_gotodraw + path_draw)
    # mp_lft.show_animation(path_gotodraw + path_draw_new)

    # mp_lft.show_armjnts_seq(path_draw, rgba=(1, 1, 0, 0.5))
    # objmat4_draw = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=path_draw[0])
    # mp_lft.show_objmat4(pen_cm, objmat4_draw, color=(1, 1, 0), transparency=1)

    objrelpos_new, objrelrot_new = \
        mp_lft.refine_relpose_by_transmat(objrelpos, objrelrot, np.linalg.inv(transmat))

    '''
    show result
    '''
    for i, a in enumerate(path_draw):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 1, 0, 1))
        if i in [0, 10] or i == len(path_draw):
            mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 1, 0, .5))
            mp_lft.ah.show_armjnts(rgba=(1, 1, 0, 0.5), armjnts=a)

    for i, a in enumerate(path_draw_new):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 0, 0, 1))
        if i in [0, 10] or i == len(path_draw_new):
            mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0, .5))
            mp_lft.ah.show_armjnts(rgba=(1, 0, 0, 0.5), armjnts=a)

    path_real = pickle.load(open("./helmet_2.pkl", "rb"))[1]
    for i, a in enumerate(path_real):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos_new, objrelrot_new, armjnts=a)
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(0, 1, 0, 1))
        if i in [0, 10] or i == len(path_real):
            mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 1, 0))
            mp_lft.ah.show_armjnts(rgba=(0, 1, 0, 0.5), armjnts=a)

    base.run()
