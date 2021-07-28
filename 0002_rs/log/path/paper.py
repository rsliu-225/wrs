import pickle

import numpy as np

import config
import motionplanner.motion_planner as m_planner
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
from localenv import envloader as el

if __name__ == "__main__":
    '''
    set up env and param
    '''
    # import pandaplotutils.pandactrl as pc
    # base = pc.World(camp=[0, 0, 1700], lookatpos=[0, 0, 1000])

    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    base.pggen.plotAxis(base.render, thickness=2, length=150)

    continuouspath_threshold = 1
    sample_num = 500000
    match_rotz = True
    # drawrec_size = (80, 80)
    drawrec_size = (40, 40)

    penpose_f_name = "bucket_cad_circle.pkl"
    paintingobj_f_name = "bucket"

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    grasp_list = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_pregrasps.pkl", "rb"))

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    objmat4_draw_list = pickle.load(open(config.PENPOSE_REL_PATH + penpose_f_name, "rb"))
    mp_lft.ah.show_objmat4_list(objmat4_draw_list, objcm=pen_cm)
    paintingobj_item = el.loadObjitem(paintingobj_f_name, sample_num=sample_num, pos=(800, 200, 780))
    paintingobj_item.show_objcm()
    path_dict = pickle.load(open('./box_nlopt.pkl', 'rb'))
    time_cost_list = []
    cost_list = []
    print(len(path_dict))
    for k, v in path_dict.items():
        print(k)
        objrelpos, objrelrot, path, time_cost = v
        time_cost_list.append(time_cost)
        cost_list.append(mp_lft.get_path_cost(path))
        mp_lft.ah.show_animation(path)
    base.run()

    print(time_cost_list)
    print('avg. time cost', np.mean(time_cost_list))
    print('avg. cost', np.mean(cost_list))
