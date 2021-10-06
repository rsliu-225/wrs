import copy
import os
import pickle
import time

import numpy as np

import config
import graspplanner.graspmap_utils as gu
import motionplanner.motion_planner as m_planner
import utils.drawpath_utils as du
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.run_utils as ru
import utiltools.robotmath as rm
from localenv import envloader as el

if __name__ == '__main__':
    '''
    set up env and param
    '''

    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')

    exp_name = 'bucket'
    folder_path = os.path.join(config.MOTIONSCRIPT_REL_PATH + 'exp_' + exp_name + '/')
    # phoxi_f_path = 'phoxi_tempdata_' + exp_name + '.pkl'
    phoxi_f_path = None

    paintingobj_f_name = 'bucket'
    # paintingobj_f_name = None
    match_rotz = False
    load = True

    continuouspath_threshold = 1
    sample_num = 1000000
    # drawrec_size = (25, 25)
    drawrec_size = (40, 40)
    max_inp = 10

    # drawpath = du.gen_circle(interval=5)
    drawpath = du.load_drawpath('circle.pkl')
    # drawpath_ms = du.gen_grid(side_len=drawrec_size[0])
    draw_motion_f_name = 'draw_circle.pkl'

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='lft')
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='rgt')

    '''
    process image
    '''

    if phoxi_f_path is None:
        pen_item = el.loadObjitem(config.PEN_STL_F_NAME, sample_num=2000, pos=(900, 400, 810), rot=(0, 0, 0))
        paintingobj_item = el.loadObjitem(paintingobj_f_name + '.stl', sample_num=sample_num, pos=(800, 100, 780))

    else:
        pen_item = \
            ru.get_obj_from_phoxiinfo_withmodel(phxilocator, config.PEN_STL_F_NAME, phoxi_f_name=phoxi_f_path,
                                                match_rotz=match_rotz, load=load,
                                                x_range=(780, 1000), y_range=(300, 400), z_range=(810, 870))

        if paintingobj_f_name is None:
            paintingobj_item = \
                ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True, reconstruct_surface=True,
                                    x_range=(400, 1000), y_range=(-100, 300), z_range=(790, 1000))
        else:
            paintingobj_item = \
                ru.get_obj_from_phoxiinfo_withmodel(phxilocator, paintingobj_f_name + '.stl',
                                                    phoxi_f_name=phoxi_f_path, match_rotz=match_rotz, load=True,
                                                    x_range=(200, 900), y_range=(-100, 300), z_range=(790, 1000))
    if paintingobj_f_name == 'bucket':
        paintingobj_item.set_drawcenter((0, 100, 50))  # bucket
        prj_direction = np.asarray((0, -1, 0))

    elif paintingobj_f_name == 'bunny':
        paintingobj_item.set_drawcenter((-5, 55, 0))  # bunny
    else:
        prj_direction = np.asarray((0, 0, 1))

    col_ps = paintingobj_item.gen_colps(radius=30, show=False)

    '''
    set obj start/goal position/rotation
    '''
    # draw the object at the initial object pose
    objmat4_init = pen_item.objmat4
    print('Pen origin position:', objmat4_init[:3, 3])

    # draw the object at the final object pose
    objpos_final = paintingobj_item.drawcenter + [0, 0, 50]
    objrot_final = rm.rodrigues([0, 1, 0], -90)
    objmat4_final = rm.homobuild(objpos_final, objrot_final)

    '''
    show obj
    '''
    try:
        paintingobj = paintingobj_item.objcm
        # paintingobj_item.show_objcm()
    except:
        pass

    mp_lft.ah.show_objmat4(pen_item.objcm, objmat4_init, rgba=(1, 1, 0, 1), showlocalframe=True)
    mp_lft.ah.show_objmat4(pen_item.objcm, objmat4_final, rgba=(1, 0, 1, 1), showlocalframe=True)
    # paintingobj_item.show_objpcd(rgba=(1, 1, 0, 1))
    # pen_item.show_objpcd(rgba=(1, 1, 0, 1))

    '''
    get draw path
    '''
    objmat4_draw_list = mp_lft.__objmat4_list_inp(
        ru.get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=drawrec_size,
                                            color=(1, 0, 0), mode='EI', direction=prj_direction, show=False),
        max_inp=max_inp)
    # mp_lft.ah.show_objmat4_list(objmat4_draw_list, objcm=pen_item.objcm, rgba=(0, 1, 0, .5))

    # objmat4_final = copy.deepcopy(objmat4_draw_list[0])
    # objmat4_final[2, 3] = objmat4_final[2, 3] + 50
    # objmat4_final[1, 3] = objmat4_final[1, 3] + 20
    # mp_lft.ah.show_objmat4(pen_item.objcm, objmat4_final, color=(0, 1, 0), transparency=0.5)
    # base.run()

    '''
    add collision model to obscmlist
    '''
    # mp_lft.add_obs(paintingobj_item.objcm)

    '''
    get available grasp
    '''
    time_start = time.time()
    pen_grasplist = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_pregrasps.pkl', 'rb'))
    print('Num of defined grasps:', len(pen_grasplist))

    mp_lft.ah.show_objmat4(pen_item.objcm, rm.homobuild([500, -500, 780], np.eye(3)), rgba=(.7, .7, .7, 1),
                           showlocalframe=True)
    mp_lft.ah.show_objmat4(pen_item.objcm, rm.homobuild([0, -500, 780], np.eye(3)), rgba=(.7, .7, .7, 1),
                           showlocalframe=True)

    pen_graspidlist_remain, time_cost_dict_gp = \
        mp_lft.filter_gid_by_objmat4_list(pen_grasplist, pen_item.objcm, [objmat4_init, objmat4_final],
                                          candidate_list=None, toggledebug=True)

    for i, grasp in enumerate(pen_grasplist):
        _, _, hndmat4 = grasp
        mp_lft.ah.show_hnd_sgl(np.dot(rm.homobuild([500, -500, 780], np.eye(3)), hndmat4), rgba=(.7, .7, .7, .2))
        if i in pen_graspidlist_remain:
            mp_lft.ah.show_hnd_sgl(np.dot(rm.homobuild([0, -500, 780], np.eye(3)), hndmat4), rgba=(0, 1, 0, .2))

    base.run()

    time_get_grasp = time.time()
    print('Available grasp id:', pen_graspidlist_remain)
    print('time cost(get grasp)', time_get_grasp - time_start, 's')

    '''
    plan pen grasp motion
    '''
    gotopick_pen_dict = {}
    picknplace_pen_dict = {}
    gotodraw_dict = {}
    draw_dict = {}

    pen_graspidlist_final = []
    time_cost_dict_mp = {}

    for i in pen_graspidlist_remain:
        time_start_tmp = time.time()
        grasp = pen_grasplist[i]
        objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objpos_final, objrot_final)
        if objrelpos is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------init to pick {str(i)}---------------')
        path_init2pick = mp_lft.plan_gotopick(grasp, objmat4_init, pen_item.objcm, objrelpos, objrelrot)
        if path_init2pick is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------pick and place {str(i)}---------------')
        path_picknplace = mp_lft.plan_picknplace(grasp, [objmat4_init, objmat4_final], pen_item.objcm, objrelpos,
                                                 objrelrot, start=copy.deepcopy(path_init2pick[-1]))
        if path_picknplace is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue
        mp_lft.ah.show_animation_hold(path_picknplace, pen_item.objcm, objrelpos, objrelrot)
        base.run()

        print(f'---------------go to draw {str(i)}---------------')
        path_gotodraw = \
            mp_lft.plan_start2end_hold(grasp, [objmat4_final, objmat4_draw_list[0]], pen_item.objcm, objrelpos,
                                       objrelrot, start=copy.deepcopy(path_picknplace[-1]))
        if path_gotodraw is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------draw path {str(i)}---------------')
        time_start_draw = time.time()
        path_draw = mp_lft.get_continuouspath_ik(path_gotodraw[-1], grasp, objmat4_draw_list,
                                                 threshold=continuouspath_threshold)
        if path_draw is None:
            time_cost_dict_mp[i] = {'flag': False, 'flag_draw': False,
                                    'time_cost': time.time() - time_start_tmp,
                                    'time_cost_draw': time.time() - time_start_draw}
            continue

        time_cost_dict_mp[i] = {'flag': False, 'flag_draw': True,
                                'time_cost': time.time() - time_start_tmp,
                                'time_cost_draw': time.time() - time_start_draw}
        print(i, time_cost_dict_mp[i])

        gotopick_pen_dict[i] = [objrelpos, objrelrot, path_init2pick]
        picknplace_pen_dict[i] = [objrelpos, objrelrot, path_picknplace]
        gotodraw_dict[i] = [objrelpos, objrelrot, path_gotodraw]
        draw_dict[i] = [objrelpos, objrelrot, path_draw]

        pickle.dump(gotopick_pen_dict, open(folder_path + 'gotopick_pen.pkl', 'wb'))
        pickle.dump(picknplace_pen_dict, open(folder_path + 'picknplace_pen.pkl', 'wb'))
        pickle.dump(gotodraw_dict, open(folder_path + 'gotodraw.pkl', 'wb'))
        pickle.dump(draw_dict, open(folder_path + draw_motion_f_name, 'wb'))

        print('---------------path saved ' + str(i) + '---------------')
        pen_graspidlist_final.append(i)
        print('time cost(first motion)', time.time() - time_start, 's')

    time_end_pnp = time.time()
    print('time cost(all motion)', time_end_pnp - time_start, 's')

    pick2cam_pen_dict = {}
    cam2place_pen_dict = {}

    graspmap = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_graspmap.pkl', 'rb'))
    objmat4_cam_list = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_objmat4_list.pkl',
             'rb'))

    print(pen_graspidlist_remain)
    print(pen_graspidlist_final)

    gu.cnt_pos_in_gmap(graspmap)
    gu.cnt_pos_in_gmap(graspmap, pen_graspidlist_final)
    success_cnt = 0

    for i in pen_graspidlist_final:
        time_start_tmp = time.time()
        grasp = pen_grasplist[i]
        pick2cam_pen_dict[i] = {}
        cam2place_pen_dict[i] = {}
        objrelpos, objrelrot, path_init2pick = gotopick_pen_dict[i]

        for objmat4_cam_id, availible in graspmap[i].items():
            if availible:
                print('---------------pick to cam', i, objmat4_cam_id, '---------------')
                path_pick2cam = \
                    mp_lft.plan_picknplace(grasp, [objmat4_init, objmat4_cam_list[objmat4_cam_id]],
                                           pen_item.objcm, objrelpos, objrelrot,
                                           use_placedownprim=False, start=copy.deepcopy(path_init2pick[-1]))
                if path_pick2cam is None:
                    continue

                print('---------------cam to place', i, objmat4_cam_id, '---------------')
                path_cam2place = \
                    mp_lft.plan_picknplace(grasp, [objmat4_cam_list[objmat4_cam_id], objmat4_final],
                                           pen_item.objcm, objrelpos, objrelrot, use_pickupprim=False,
                                           start=copy.deepcopy(path_pick2cam[-1]))
                print(len(path_pick2cam), path_pick2cam[-1])
                if path_cam2place is None:
                    continue
                time_cost_dict_mp[i] = \
                    {'flag': True, 'flag_draw': True,
                     'time_cost': time_cost_dict_mp[i]['time_cost'] + time.time() - time_start_tmp,
                     'time_cost_draw': time_cost_dict_mp[i]['time_cost_draw']}

                pick2cam_pen_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_pick2cam]
                cam2place_pen_dict[i][objmat4_cam_id] = [objrelpos, objrelrot, path_cam2place]
                pickle.dump(pick2cam_pen_dict, open(folder_path + 'pick2cam_pen.pkl', 'wb'))
                pickle.dump(cam2place_pen_dict, open(folder_path + 'cam2place_pen.pkl', 'wb'))
                success_cnt += 1
                print('---------------path saved', i, objmat4_cam_id, '---------------')
                print('time cost(take photo motion)', time.time() - time_end_pnp, 's')
                break
        if not time_cost_dict_mp[i]['flag']:
            time_cost_dict_mp[i]['time_cost'] += time.time() - time_start_tmp
