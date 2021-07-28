import os
import pickle
import random
import time

import run_utils as ru

import config
import graspplanner.graspmap_utils as gu
import motionplanner.motion_planner as m_planner
import utils.drawpath_utils as du
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utiltools.robotmath as rm
from localenv import envloader as el

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    continuouspath_threshold = 1
    exp_name = "cylinder_mtp"
    folder_path = os.path.join(config.MOTIONSCRIPT_REL_PATH + "exp_" + exp_name + "/")
    phoxi_f_path = f"phoxi_tempdata_{exp_name}.pkl"

    paintingobj_f_name = "cylinder"
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    match_rotz = False
    load = False

    sample_num = 1000000
    resolution = 1
    drawrec_size = (40, 40)

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    init planner
    '''
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")

    '''
    process image
    '''
    y_range_list = [(250, 325), (325, 400), (400, 475), (475, 550)]
    pen_info_list = []
    for y_range in y_range_list:
        rgba = (random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), 1)
        # base.pggen.plotArrow(base.render, spos=(780, y_range[0], 830), epos=(1000, y_range[0], 830), rgba=rgba)
        # base.pggen.plotArrow(base.render, spos=(780, y_range[1], 830), epos=(1000, y_range[1], 830), rgba=rgba)
        pen_info = \
            ru.get_obj_from_phoxiinfo_withmodel(phxilocator, config.PEN_STL_F_NAME, phoxi_f_name=phoxi_f_path,
                                                match_rotz=match_rotz, load=load, resolution=resolution,
                                                x_range=(680, 980), y_range=y_range, z_range=(810, 860))
        pen_info_list.append(pen_info)
        pcdu.show_pcd(pen_info["pcd"])
        motion_planner_lft.show_objmat4(pen_cm, pen_info["objmat4"], rgba=(0, 1, 0, .5), showlocalframe=True)

    if paintingobj_f_name is None:
        paintingobj_item = \
            ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True, resolution=resolution,
                                x_range=(200, 900), y_range=(-100, 300), z_range=(780, 1000))
    else:
        paintingobj_item = \
            ru.get_obj_from_phoxiinfo_withmodel(phxilocator, paintingobj_f_name + ".stl", resolution=resolution,
                                                phoxi_f_name=phoxi_f_path, match_rotz=match_rotz, load=True,
                                                x_range=(200, 900), y_range=(-100, 300), z_range=(780, 1000))
    # paintingobj_item.set_drawcenter((0, 0, 0))

    '''
    set obj start/goal position/rotation
    '''
    # draw the object at the initial object pose
    objmat4_init_list = []

    for pen_info in pen_info_list:
        objmat4_init = pen_info["objmat4"]
        pen_cm = pen_info["model"]
        objmat4_init_list.append(objmat4_init)
        print("Pen origin position:", objmat4_init[:3, 3])
        motion_planner_lft.show_objmat4(pen_cm, objmat4_init, rgba=(0, 1, 0, .5))
        pcdu.show_pcd(pen_info["pcd"], rgba=(1, 1, 0, 1))

    # draw the object at the final object pose
    objpos_final = paintingobj_item["draw_center"] + [0, 0, 50]
    objrot_final = rm.rodrigues([0, 1, 0], -90)
    objmat4_final = rm.homobuild(objpos_final, objrot_final)

    # motion_planner_lft.show_objmat4(pen_cm, objmat4_final, color=(0, 1, 0, .5), showlocalframe=True)

    '''
    show painting obj
    '''
    try:
        paintingobj = paintingobj_item["model"]
        paintingobj.sethomomat(paintingobj_item["objmat4"])
        paintingobj.reparentTo(base.render)
    except:
        pass
    pcdu.show_pcd(paintingobj_item["pcd"], rgba=(1, 1, 0, 1))
    # base.run()
    '''
    get draw path
    '''
    drawpath = du.gen_circle(interval=5)
    objmat4_draw_list = motion_planner_lft.objmat4_list_inp(
        ru.get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=drawrec_size))
    motion_planner_lft.ah.show_objmat4_list(objmat4_draw_list, pen_cm, rgba=(1, 0, 0, .5))

    '''
    add collision model to obscmlist
    '''
    # motion_planner_lft.add_obs(paintingobj_info["model"])
    # motion_planner_x_lft.add_obs(paintingobj_info["model"])

    '''
    get available grasp
    '''
    time_start = time.time()
    pen_grasplist_all = \
        pickle.load(
            open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_pregrasps.pkl", "rb"))
    print("Num of defined grasps:", len(pen_grasplist_all))

    pen_graspidlist_remain = \
        motion_planner_lft.filter_gid_by_objmat4_list(pen_grasplist_all, pen_cm,
                                                      objmat4_init_list + [objmat4_final],
                                                      candidate_list=None)
    pen_graspidlist_remain = \
        motion_planner_lft.get_available_graspid_by_objmat4_list_msc(pen_grasplist_all, pen_cm, objmat4_draw_list,
                                                                     available_graspid_list=pen_graspidlist_remain,
                                                                     threshold=continuouspath_threshold)
    pen_graspidlist_available = pen_graspidlist_remain
    time_get_grasp = time.time()
    print("Available grasp id:", pen_graspidlist_available)
    print('time cost(get grasp)', time_get_grasp - time_start, 's')

    '''
    plan pen grasp motion
    '''
    gotopick_pen0_dict = {}
    gotopick_pen1_dict = {}
    gotopick_pen2_dict = {}
    gotopick_pen3_dict = {}

    picknplace_pen0_dict = {}
    picknplace_pen1_dict = {}
    picknplace_pen2_dict = {}
    picknplace_pen3_dict = {}

    gotodraw_dict = {}
    draw_rh_dict = {}
    draw_dict = {}

    pen_graspidlist_available_final = []

    for i in pen_graspidlist_available:
        grasp = pen_grasplist_all[i]
        objrelpos, objrelrot = motion_planner_lft.get_rel_posrot(grasp, objpos_final, objrot_final)
        if objrelpos is None:
            continue
        is_all_pen_success = True
        for pen_id, pen_info in enumerate(pen_info_list):
            print("===============pen " + str(pen_id) + "===============")
            objmat4_init = pen_info["objmat4"]
            print("---------------init to pick " + str(i) + "---------------")
            path_init2pick = motion_planner_lft.plan_gotopick(grasp, objmat4_init, pen_cm, objrelpos, objrelrot)
            if path_init2pick is None:
                is_all_pen_success = False
                break

            print("---------------pick and place pen " + str(i) + "---------------")
            path_picknplace_pen = \
                motion_planner_lft.plan_picknplace(grasp, [objmat4_init, objmat4_final], pen_cm, objrelpos, objrelrot,
                                                   start=path_init2pick[-1])
            if path_picknplace_pen is None:
                is_all_pen_success = False
                break

            if pen_id == 0:
                gotopick_pen0_dict[i] = [objrelpos, objrelrot, path_init2pick]
                picknplace_pen0_dict[i] = [objrelpos, objrelrot, path_picknplace_pen]
                pickle.dump(gotopick_pen0_dict, open(folder_path + "gotopick_pen0.pkl", "wb"))
                pickle.dump(picknplace_pen0_dict, open(folder_path + "picknplace_pen0.pkl", "wb"))
            elif pen_id == 1:
                gotopick_pen1_dict[i] = [objrelpos, objrelrot, path_init2pick]
                picknplace_pen1_dict[i] = [objrelpos, objrelrot, path_picknplace_pen]
                pickle.dump(gotopick_pen1_dict, open(folder_path + "gotopick_pen1.pkl", "wb"))
                pickle.dump(picknplace_pen1_dict, open(folder_path + "picknplace_pen1.pkl", "wb"))
            elif pen_id == 2:
                gotopick_pen2_dict[i] = [objrelpos, objrelrot, path_init2pick]
                picknplace_pen2_dict[i] = [objrelpos, objrelrot, path_picknplace_pen]
                pickle.dump(gotopick_pen2_dict, open(folder_path + "gotopick_pen2.pkl", "wb"))
                pickle.dump(picknplace_pen2_dict, open(folder_path + "picknplace_pen2.pkl", "wb"))
            elif pen_id == 3:
                gotopick_pen3_dict[i] = [objrelpos, objrelrot, path_init2pick]
                picknplace_pen3_dict[i] = [objrelpos, objrelrot, path_picknplace_pen]
                pickle.dump(gotopick_pen3_dict, open(folder_path + "gotopick_pen3.pkl", "wb"))
                pickle.dump(picknplace_pen3_dict, open(folder_path + "picknplace_pen3.pkl", "wb"))

        print("Is all pen success:", i, is_all_pen_success)
        if not is_all_pen_success:
            continue

        print("---------------go to draw " + str(i) + "---------------")
        path_gotodraw = \
            motion_planner_lft.plan_start2end_hold(grasp, [objmat4_final, objmat4_draw_list[0]],
                                                   pen_cm, objrelpos, objrelrot)
        if path_gotodraw is None:
            continue

        print("---------------draw path msd " + str(i) + "---------------")
        path_draw = \
            motion_planner_lft.get_continuouspath(path_gotodraw[-1], grasp, objmat4_draw_list,
                                                  threshold=continuouspath_threshold)
        if path_draw is None:
            continue

        gotodraw_dict[i] = [objrelpos, objrelrot, path_gotodraw]
        draw_dict[i] = [objrelpos, objrelrot, path_draw]
        pickle.dump(gotodraw_dict, open(folder_path + "gotodraw.pkl", "wb"))
        pickle.dump(draw_dict, open(folder_path + "draw_msd.pkl", "wb"))

        print("---------------path saved " + str(i) + "---------------")
        pen_graspidlist_available_final.append(i)
        print('time cost(first motion)', time.time() - time_start, 's')

    time_end_pnp = time.time()
    print('time cost(all motion)', time_end_pnp - time_start, 's')

    pick2cam_pen0_dict = {}
    pick2cam_pen1_dict = {}
    pick2cam_pen2_dict = {}
    pick2cam_pen3_dict = {}

    cam2place_pen_dict = {}

    graspmap = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_graspmap.pkl", "rb"))
    objmat4_cam_list = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_objmat4_list.pkl", "rb"))

    print("Available grasp id:", pen_graspidlist_available)
    print("Available grasp id(after planing):", pen_graspidlist_available_final)

    gu.cnt_pos_in_gmap(graspmap)
    gu.cnt_pos_in_gmap(graspmap, pen_graspidlist_available_final)
    success_cnt = 0
    for i in pen_graspidlist_available_final:

        grasp = pen_grasplist_all[i]
        objrelpos, objrelrot = motion_planner_lft.get_rel_posrot(grasp, objpos_final, objrot_final)

        if objrelpos is None:
            continue
        pick2cam_pen0_dict[i] = {}
        pick2cam_pen1_dict[i] = {}
        pick2cam_pen2_dict[i] = {}
        pick2cam_pen3_dict[i] = {}
        cam2place_pen_dict[i] = {}

        for objmat4_cam_id, availible in graspmap[i].items():
            if not availible:
                is_all_pen_success = False
                continue

            is_all_pen_success = True
            for pen_id, pen_info in enumerate(pen_info_list):
                print("===============pen " + str(pen_id) + "===============")
                objmat4_init = pen_info["objmat4"]

                print("---------------pick to cam", i, objmat4_cam_id, "---------------")
                path_pick2cam = motion_planner_lft.plan_picknplace(grasp, [objmat4_init,
                                                                           objmat4_cam_list[objmat4_cam_id]],
                                                                   pen_cm, objrelpos, objrelrot,
                                                                   use_placedownprim=False)
                if path_pick2cam is None:
                    is_all_pen_success = False
                    break

                if pen_id == 0:
                    pick2cam_pen0_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_pick2cam]
                    pickle.dump(pick2cam_pen0_dict, open(folder_path + "pick2cam_pen0.pkl", "wb"))
                if pen_id == 1:
                    pick2cam_pen1_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_pick2cam]
                    pickle.dump(pick2cam_pen1_dict, open(folder_path + "pick2cam_pen1.pkl", "wb"))
                if pen_id == 2:
                    pick2cam_pen2_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_pick2cam]
                    pickle.dump(pick2cam_pen2_dict, open(folder_path + "pick2cam_pen2.pkl", "wb"))
                if pen_id == 3:
                    pick2cam_pen3_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_pick2cam]
                    pickle.dump(pick2cam_pen3_dict, open(folder_path + "pick2cam_pen3.pkl", "wb"))

            print("Is all pen success:", objmat4_cam_id, is_all_pen_success)
            if is_all_pen_success:

                print("---------------cam to place", i, objmat4_cam_id, "---------------")
                path_cam2place = motion_planner_lft.plan_picknplace(grasp, [objmat4_cam_list[objmat4_cam_id],
                                                                            objmat4_final],
                                                                    pen_cm, objrelpos, objrelrot, use_msc=False,
                                                                    use_pickupprim=False)
                if path_cam2place is None:
                    continue

                cam2place_pen_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_cam2place]
                pickle.dump(cam2place_pen_dict, open(folder_path + "cam2place_pen.pkl", "wb"))

                success_cnt += 1
                print("---------------path saved", i, objmat4_cam_id, "---------------")
                print('time cost(take photo motion)', time.time() - time_end_pnp, 's')

        print("cam pose success count:", success_cnt)
        gu.cnt_pos_in_gmap(graspmap, pen_graspidlist_available_final)

    print('time cost(all)', time.time() - time_start)
    print('time cost(all planning)', time.time() - time_get_grasp)
