import os
import pickle
import time

import numpy as np

import config
import motionplanner.motion_planner as m_planner
import run_config as rconfig
import utils.drawpath_utils as du
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.run_utils as ru
import utiltools.robotmath as rm
from localenv import envloader as el


def get_drawmotion_f_name(folder_path, drawpath_f_name, type="ss"):
    if type == "ss":
        return os.path.join(config.ROOT, folder_path, "draw_" + drawpath_f_name.split(".pkl")[0] + ".pkl")
    else:
        return os.path.join(config.ROOT, folder_path, "draw_" + drawpath_f_name.split(".pkl")[0] + "_ms.pkl")


def show_result(mp, path, objmat4_list, grasp, show_objax=False):
    for objmat4 in objmat4_list:
        objrelpos, objrelrot = mp.get_rel_posrot(grasp, objmat4[:3, 3], objmat4[:3, :3])
        if objrelpos is not None:
            break
    if show_objax:
        for objmat4 in objmat4_list:
            base.pggen.plotSphere(base.render, objmat4[:3, 3], radius=2, rgba=(1, 0, 0, 1))
            mp.rbth.draw_axis(objmat4[:3, 3], objmat4[:3, :3], rgba=(1, 0, 0, .2))
        for armjnts in path:
            objmat4_real = mp.get_world_objmat4(objrelpos, objrelrot, armjnts)
            mp.rbth.draw_axis(objmat4_real[:3, 3], objmat4_real[:3, :3], rgba=(0, 1, 0, .2))
    mp.ah.show_objmat4_list(objmat4_list, objcm=pen_cm, fromrgba=(1, 1, 1, .5))
    mp.ah.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
    base.run()


def get_objmat4_draw_list(mp, drawpath_f_name, paintingobj_item, drawrec_size, type="ss", mode="DI",
                          prj_direction=np.asarray((0, 0, 1))):
    if type == "ss":
        drawpath = du.load_drawpath(drawpath_f_name)
        # drawpath = drawpath[36:] + drawpath[:36]
        objmat4_draw_list = \
            ru.get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=drawrec_size,
                                                color=(0, 1, 0), mode=mode, direction=prj_direction)

        return mp.objmat4_list_inp(objmat4_draw_list)
    else:
        drawpath_ms = du.load_drawpath(drawpath_f_name)
        objmat4_draw_list_ms = \
            ru.get_pen_objmat4_list_by_drawpath(drawpath_ms, paintingobj_item, color=(0, 1, 0),
                                                drawrec_size=drawrec_size, mode=mode, direction=prj_direction)
        return objmat4_draw_list_ms


def move_objmat4_list(objmat4_list, dx=0, dy=0, dz=0):
    return [rm.homobuild(objmat4[:3, 3] + np.asarray([dx, dy, dz]), objmat4[:3, :3]) for objmat4 in objmat4_list]


if __name__ == '__main__':
    '''
    set up env and param
    '''
    # import pandaplotutils.pandactrl as pc
    # base = pc.World(camp=[0, 0, 1700], lookatpos=[0, 0, 1000])

    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    base.pggen.plotAxis(base.render, thickness=2, length=150)

    continuouspath_threshold = 1
    sample_num = 5000000
    # drawrec_size = (80, 80)
    # drawrec_size = (45, 45)
    drawrec_size = (60, 60)
    # prj_direction = np.asarray((0, -1, 0))
    prj_direction = np.asarray((0, 0, 1))
    exp_name = "helemet"

    paintingobj_f_name = "bucket"
    penpose_f_name = "bucket_cad_circle.pkl"
    # paintingobj_f_name = "box"
    # penpose_f_name = "box_cad_circle.pkl"
    # penpose_f_name = None

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    grasp_list = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_pregrasps.pkl", "rb"))

    drawpath_f_name = "circle.pkl"
    type = "ss"

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")

    '''
    process image
    '''
    if paintingobj_f_name in rconfig.OBJ_POS.keys():
        paintingobj_item = el.loadObjitem(paintingobj_f_name, sample_num=sample_num,
                                          pos=rconfig.OBJ_POS[paintingobj_f_name])
    else:
        paintingobj_item = el.loadObjitem(paintingobj_f_name, sample_num=sample_num, pos=(800, 100, 780))

    '''
    show obj
    '''
    try:
        paintingobj_item.show_objcm(rgba=(1, 1, 1, .5))
    except:
        pass

    '''
    get pen pose
    '''
    time_start = time.time()

    # if penpose_f_name is not None:
    #     objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + penpose_f_name, "rb"))
    # else:
    #     if paintingobj_f_name in rconfig.PRJ_INFO.keys():
    #         paintingobj_item.set_drawcenter(rconfig.PRJ_INFO[paintingobj_f_name]['draw_center'])
    #         prj_direction = np.asarray(rconfig.PRJ_INFO[paintingobj_f_name]['prj_direction'])
    #     else:
    #         prj_direction = np.asarray((0, 0, 1))
    #     objmat4_list = get_objmat4_draw_list(mp_lft, drawpath_f_name, paintingobj_item, drawrec_size, type=type,
    #                                          mode="EI", prj_direction=prj_direction)
    #     # pickle.dump(objmat4_list, open(config.PENPOSE_REL_PATH + penpose_f_name, "wb"))
    #     print('time cost(projection&slerp)', time.time() - time_start, 's')
    objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + penpose_f_name, "rb"))
    print("drawing path length:", len(objmat4_list))
    # mp_lft.ah.show_objmat4(pen_cm, objmat4_list[0], rgba=(1, 0, 0, 1))
    # mp_lft.ah.show_objmat4_list(objmat4_list, fromrgba=(1, 1, 1, .2), torgba=(1, 1, 1, .2), objcm=pen_cm)
    mp_lft.ah.show_objmat4_list_pos(objmat4_list, rgba=(1, 0, 0, 1))
    base.run()
    col_ps = paintingobj_item.gen_colps(radius=40, show=False)

    success_cnt = 0
    time_start = time.time()
    msc = 'default'

    for grasp_id in range(62):
        grasp = grasp_list[grasp_id]
        path_draw = \
            mp_lft.get_continuouspath_nlopt(msc, grasp, objmat4_list, grasp_id=grasp_id,
                                            col_ps=col_ps, roll_limit=50, pos_limit=1e-2, add_mvcon=False,
                                            dump_f_name='boxcolw50')
        # if path_draw is None:
        #     continue
        # objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objmat4_list[0][:3, 3], objmat4_list[0][:3, :3])
        # break

    print(f"time cost(loop all grasp): {time.time() - time_start}")
    print(f"{success_cnt} of {len(grasp_list)}")
    # mp_lft.ah.show_animation_hold(path_draw, pen_cm, objrelpos, objrelrot, jawwidth=18)
    # mp_lft.ah.show_animation_hold(path_draw2, pen_cm, objrelpos, objrelrot, jawwidth=18)

    base.run()
