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
import utils.run_script_utils as rmu
import utils.run_utils as ru
from localenv import envloader as el


def get_drawmotion_f_name(folder_path, drawpath_f_name, note='', type="ss"):
    if type == "ss":
        return os.path.join(config.ROOT, folder_path, "draw_" + drawpath_f_name.split(".pkl")[0] + note + ".pkl")
    else:
        return os.path.join(config.ROOT, folder_path, "draw_" + drawpath_f_name.split(".pkl")[0] + note + "_ms.pkl")


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
        objmat4_list = ru.get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=drawrec_size,
                                                           color=(0, 1, 0), mode=mode, direction=prj_direction)

        return mp.objmat4_list_inp(objmat4_list)
    else:
        drawpath_ms = du.load_drawpath(drawpath_f_name)
        objmat4_draw_list_ms = \
            ru.get_pen_objmat4_list_by_drawpath(drawpath_ms, paintingobj_item, color=(0, 1, 0),
                                                drawrec_size=drawrec_size, mode=mode, direction=prj_direction)
        return objmat4_draw_list_ms


def dump_drawpath_ss(mp, motion_f_path, objmat4_list, grasp, grasp_id, msc=None, col_ps=None, threshold=1.0,
                     method='ik'):
    try:
        draw_dict = pickle.load(open(motion_f_path, "rb"))
    except:
        draw_dict = {}

    time_start = time.time()

    print(f"---------------planing draw path {str(grasp_id)}---------------")
    # sample_range = [(0, 0, 0)]
    # sample_range = mp.get_rpy_list((-180, 180, 1), (-15, 15 + 1, 15), (-15, 15 + 1, 15))
    # sample_range = mp.get_rpy_list((0, 0, 0), (-5, 5 + 1, 1), (-5, 5 + 1, 1))
    # sample_range = mp.get_rpy_list((0, 0, 0), (-5, 5 + 1, 1), (-5, 5 + 1, 1))
    # path_draw = mp.get_continuouspath_opt1(grasp, grasp_id, objmat4_list, sample_range, msc=msc)
    msc = mp_lft.get_draw_sconfig(objmat4_list[0], grasp)
    if method == 'ik':
        path_draw = mp.get_continuouspath_ik(msc, grasp, objmat4_list, grasp_id=grasp_id, threshold=threshold)
    else:
        # dump_f_name = "bucketcol"
        path_draw = mp.get_continuouspath_nlopt(msc, grasp, objmat4_list, grasp_id=grasp_id, col_ps=col_ps,
                                                roll_limit=60, pos_limit=1e-2, add_mvcon=False, dump_f_name=None)
    objrelpos, objrelrot = mp.get_rel_posrot(grasp, objmat4_list[0][:3, 3], objmat4_list[0][:3, :3])

    if path_draw is not None:
        # show_result(mp, path_draw, objmat4_list, grasp)
        time_cost = time.time() - time_start
        draw_dict[grasp_id] = [objrelpos, objrelrot, path_draw]
        pickle.dump(draw_dict, open(motion_f_path, "wb"))
        print('time cost(get ik)', time_cost, 's')
        return True
    else:
        print("planning drawing motion failed!")
        return False


def dump_drawpath_ms(mp, motion_f_path, objmat4_draw_list_ms, grasp, grasp_id, msc=None):
    if os.path.exists(motion_f_path):
        draw_dict = pickle.load(open(motion_f_path, "rb"))
    else:
        draw_dict = {}

    draw_dict[grasp_id] = {}

    if msc is None:
        msc = mp.initjnts

    print(f"---------------planing draw path {str(grasp_id)}---------------")
    status = True
    for i, objmat4_draw_list_stroke in enumerate(objmat4_draw_list_ms):
        path_draw_stroke = mp.get_continuouspath(msc, grasp, objmat4_draw_list_stroke,
                                                 threshold=continuouspath_threshold)
        if path_draw_stroke is not None:
            draw_dict[grasp_id]["stroke_" + str(i)] = [objrelpos, objrelrot, path_draw_stroke]
            pickle.dump(draw_dict, open(motion_f_path, "wb"))
        else:
            print("stroke_" + str(i), "failed!")
            status = False

    return status


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
    sample_num = 500000
    match_rotz = True
    # drawrec_size = (80, 80)
    drawrec_size = (40, 40)
    # drawrec_size = (25, 25)
    # exp_name = "bucket"
    # exp_name = "cylinder_cad"
    exp_name = "leg"
    # exp_name = "box"
    # exp_name = "cube"
    # folder_path = os.path.join(config.MOTIONSCRIPT_REL_PATH + f"exp_{exp_name}/")
    phoxi_f_path = f"phoxi_tempdata_{exp_name}.pkl"
    # phoxi_f_path = None
    folder_path = f"exp_{exp_name}/"

    # paintingobj_f_name = "box"
    # paintingobj_f_name = "bucket"
    # paintingobj_f_name = "cylinder"
    # paintingobj_f_name = "bunny"
    # paintingobj_f_name = "cube"
    paintingobj_f_name = None

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    grasp_list = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_pregrasps.pkl", "rb"))

    drawpath_f_name = "spiral.pkl"
    type = "ss"
    penpose_f_name = f"{exp_name}_{drawpath_f_name.split('.pkl')[0]}.pkl"

    motion_f_path = get_drawmotion_f_name(folder_path, drawpath_f_name, type=type)
    cam2place_pen_dict = rmu.load_motion_f(folder_path, "cam2place_pen.pkl")
    picknplace_pen_dict = rmu.load_motion_f(folder_path, "picknplace_pen.pkl")
    grasp_id_list = [k for k in list(cam2place_pen_dict.keys()) if cam2place_pen_dict[k] != {}]
    # grasp_id_list = [k for k in list(picknplace_pen_dict.keys()) if picknplace_pen_dict[k] != []]

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")

    '''
    process image
    '''
    if phoxi_f_path is None:
        if paintingobj_f_name in rconfig.OBJ_POS.keys():
            paintingobj_item = el.loadObjitem(paintingobj_f_name, sample_num=sample_num,
                                              pos=rconfig.OBJ_POS[paintingobj_f_name])
        else:
            paintingobj_item = el.loadObjitem(paintingobj_f_name, sample_num=sample_num, pos=(800, 100, 780))
    else:
        if paintingobj_f_name is None:
            paintingobj_item = \
                ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True, reconstruct_surface=True,
                                    x_range=(400, 1000), y_range=(-100, 300), z_range=(790, 1000))
        else:
            paintingobj_item = \
                ru.get_obj_from_phoxiinfo_withmodel(phxilocator, paintingobj_f_name, phoxi_f_name=phoxi_f_path,
                                                    load=True, match_rotz=match_rotz,
                                                    x_range=(200, 900), y_range=(-100, 300), z_range=(790, 1000))
    if exp_name in rconfig.PRJ_INFO.keys():
        paintingobj_item.set_drawcenter(rconfig.PRJ_INFO[exp_name]['draw_center'])
        prj_direction = np.asarray(rconfig.PRJ_INFO[exp_name]['prj_direction'])
    else:
        prj_direction = np.asarray((0, 0, 1))

    '''
    show obj
    '''
    try:
        paintingobj_item.show_objcm(rgba=(1, 1, 1, .5))
        # paintingobj_item.show_objpcd(rgba=(1, 1, 0, .5))
    except:
        pass

    '''
    get pen pose
    '''
    time_start = time.time()
    print('time cost(projection&slerp)', time.time() - time_start, 's')
    objmat4_draw_list = get_objmat4_draw_list(mp_lft, drawpath_f_name, paintingobj_item, drawrec_size, type=type,
                                              mode="EI", prj_direction=prj_direction)
    pickle.dump(objmat4_draw_list, open(config.PENPOSE_REL_PATH + penpose_f_name, "wb"))

    '''
    load pen pose
    '''
    objmat4_draw_list = pickle.load(open(config.PENPOSE_REL_PATH + penpose_f_name, "rb"))
    print("drawing path length:", len(objmat4_draw_list))
    mp_lft.ah.show_objmat4(pen_cm, objmat4_draw_list[0], rgba=(1, 0, 0, 1))
    mp_lft.ah.show_objmat4_list(objmat4_draw_list, fromrgba=(1, 1, 1, .2), objcm=pen_cm)
    col_ps = paintingobj_item.gen_colps(radius=40, show=True, max_smp=400)
    # col_ps = None
    # base.run()

    success_cnt = 0
    time_start = time.time()
    motion_f_path = os.path.join(config.MOTIONSCRIPT_REL_PATH, folder_path,
                                 f'draw_{drawpath_f_name.split(".pkl")[0]}.pkl')
    for grasp_id in grasp_id_list:
        # objmat4_id = list(cam2place_pen_dict[grasp_id].keys())[0]
        # objrelpos, objrelrot, path_cam2place_pen = cam2place_pen_dict[grasp_id][objmat4_id]
        objrelpos, objrelrot, path_picknplace_pen = picknplace_pen_dict[grasp_id]
        msc = path_picknplace_pen[-1]

        # motion_f_path = os.path.join(config.ROOT, 'log/path/', f'{paintingobj_f_name}.pkl')
        grasp = grasp_list[grasp_id]
        # msc = None
        '''
        plan pen draw motion 
        '''
        if type == "ss":
            status = dump_drawpath_ss(mp_lft, motion_f_path, objmat4_draw_list, grasp, grasp_id, msc=msc, col_ps=col_ps,
                                      threshold=continuouspath_threshold)
            if status:
                success_cnt += 1
        else:
            dump_drawpath_ms(mp_lft, motion_f_path, objmat4_draw_list, grasp, grasp_id)

    print(f"time cost(loop all grasp): {time.time() - time_start}")
    print(f"{success_cnt} of {len(grasp_list)}")

    if type == "ss":
        rmu.show_drawmotion_ss(mp_lft, pen_cm, motion_f_path, grasp_id_list, jawwidth=18)
    else:
        rmu.show_drawmotion_ms(mp_lft, pen_cm, motion_f_path, grasp_id_list, jawwidth=18)

    base.run()
