import copy
import os
import pickle
import time

import config
import graspplanner.graspmap_utils as gu
import motionplanner.motion_planner as m_planner
import utils.drawpath_utils as du
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.run_utils as ru
from localenv import envloader as el

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')

    continuouspath_threshold = 1
    folder_path = os.path.join(config.ROOT + '/motionscript/tc_bunny_nlopt/')

    phoxi_f_path = 'phoxi_tempdata_0524.pkl'

    pen_f_name = 'pentip'
    paintingobj_f_name = 'bunny'
    # paintingobj_f_name = None
    draw_motion_f_name = 'draw_circle.pkl'

    match_rotz = False
    load = True

    sample_num = 1000000
    resolution = 1
    # drawrec_size = (40, 40)
    drawrec_size = (25, 25)
    max_inp = 10
    drawpath = du.gen_circle(interval=5)

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='lft')

    '''
    process image
    '''
    pen_item = ru.get_obj_from_phoxiinfo_withmodel(phxilocator, pen_f_name + '.stl', phoxi_f_name=phoxi_f_path,
                                                   match_rotz=match_rotz, load=load, resolution=resolution,
                                                   x_range=(780, 1080), y_range=(300, 400), z_range=(810, 870))
    if paintingobj_f_name is None:
        paintingobj_item = \
            ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True, resolution=resolution,
                                reconstruct_surface=True,
                                x_range=(400, 1080), y_range=(-100, 300), z_range=(790, 1000))
    else:
        # paintingobj_item = el.loadObjitem(paintingobj_f_name + '.stl', sample_num=sample_num, pos=(800, 100, 780))
        paintingobj_item = el.loadObjitem(paintingobj_f_name + '.stl', sample_num=sample_num, pos=(840, 220, 800))
        paintingobj_item.set_drawcenter((0, 45, 0))

    col_ps = paintingobj_item.gen_colps(radius=1, show=True)

    '''
    set obj start/goal position/rotation
    '''
    # draw the object at the initial object pose
    objmat4_init = pen_item.objmat4
    objpos_init = objmat4_init[:3, 3]
    objrot_init = objmat4_init[:3, :3]
    print('Pen origin position:', objmat4_init[:3, 3])

    '''
    show obj
    '''
    paintingobj = paintingobj_item.objcm
    paintingobj_item.show_objcm()

    mp_lft.ah.show_objmat4(pen_item.objcm, objmat4_init, rgba=(0, 1, 0, .5), showlocalframe=True)
    paintingobj_item.show_objpcd(rgba=(1, 1, 0, 1))
    pen_item.show_objpcd(rgba=(1, 1, 0, 1))
    # base.run()

    '''
    get draw path
    '''
    objmat4_draw_list = mp_lft.__objmat4_list_inp(
        ru.get_pen_objmat4_list_by_drawpath(drawpath, paintingobj_item, drawrec_size=drawrec_size, color=(1, 0, 0),
                                            mode='EI'), max_inp=max_inp)
    # motion_planner_lft.ah.show_objmat4_list(objmat4_draw_list, objcm=pen_item.objcm, rgba=(0, 1, 0, .5))

    '''
    add collision model to obscmlist
    '''
    # motion_planner_lft.add_obs(paintingobj_item.objcm)
    # motion_planner_x_lft.add_obs(paintingobj_item.objcm)

    '''
    get available grasp
    '''
    time_start = time.time()
    pen_glist = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_pregrasps.pkl', 'rb'))
    print('Num of defined grasps:', len(pen_glist))

    pen_gidlist_res, time_cost_dict_gp = \
        mp_lft.filter_gid_by_objmat4_list(pen_glist, pen_item.objcm, [objmat4_init], candidate_list=None)

    time_get_grasp = time.time()
    print('Available grasp id:', pen_gidlist_res)
    print('time cost(get grasp)', time_get_grasp - time_start, 's')

    '''
    plan pen grasp motion
    '''
    gotopick_pen_dict = {}
    picknplace_pen_dict = {}
    gotodraw_dict = {}
    draw_dict = {}

    pen_gid_list_final = []
    time_cost_dict_mp = {}

    for i in pen_gidlist_res:
        time_start_tmp = time.time()
        grasp = pen_glist[i]
        objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objpos_init, objrot_init)
        if objrelpos is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------init to pick {str(i)}---------------')
        path_init2pick = mp_lft.plan_gotopick(grasp, objmat4_init, pen_item.objcm, objrelpos, objrelrot)
        if path_init2pick is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        draw_start = mp_lft.get_draw_sconfig(objmat4_draw_list[0], grasp)
        if draw_start is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        objmat4_final = mp_lft.get_world_objmat4(objrelpos, objrelrot,
                                                 mp_lft.get_tool_primitive_armjnts(draw_start, objrelrot, length=20))
        objmat4_draw_start = mp_lft.get_world_objmat4(objrelpos, objrelrot, draw_start)

        print(f'---------------pick and place {str(i)}---------------')
        path_picknplace = mp_lft.plan_picknplace(grasp, [objmat4_init, objmat4_final], pen_item.objcm, objrelpos,
                                                 objrelrot,
                                                 start=copy.deepcopy(path_init2pick[-1]))
        if path_picknplace is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------go to draw {str(i)}---------------')
        path_gotodraw = \
            mp_lft.plan_start2end_hold(grasp, [objmat4_final, objmat4_draw_start], pen_item.objcm, objrelpos, objrelrot,
                                       start=copy.deepcopy(path_picknplace[-1]))
        if path_gotodraw is None:
            time_cost_dict_mp[i] = {'flag': False, 'time_cost': time.time() - time_start_tmp}
            continue

        print(f'---------------draw path {str(i)}---------------')
        time_start_draw = time.time()

        path_draw = mp_lft.get_continuouspath_nlopt(draw_start, grasp, objmat4_draw_list, grasp_id=i,
                                                    col_ps=col_ps, roll_limit=1e-2, pos_limit=1e-2, add_mvcon=False)

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
        pen_gid_list_final.append(i)
        print('time cost(first motion)', time.time() - time_start, 's')

    time_end_pnp = time.time()
    print('time cost(all motion)', time_end_pnp - time_start, 's')

    pick2cam_pen_dict = {}
    cam2place_pen_dict = {}

    gmap = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_graspmap.pkl', 'rb'))
    objmat4_cam_list = pickle.load(
        open(config.GRASPMAP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_objmat4_list.pkl', 'rb'))

    print(pen_gidlist_res)
    print(pen_gid_list_final)

    gu.cnt_pos_in_gmap(gmap)
    gu.cnt_pos_in_gmap(gmap, pen_gid_list_final)
    success_cnt = 0

    for i in pen_gid_list_final:
        time_start_tmp = time.time()
        grasp = pen_glist[i]
        pick2cam_pen_dict[i] = {}
        cam2place_pen_dict[i] = {}
        objrelpos, objrelrot, path_init2pick = gotopick_pen_dict[i]
        _, _, path_gotodraw = gotodraw_dict[i]

        for objmat4_cam_id, availible in gmap[i].items():
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
                    mp_lft.plan_start2end_hold_armj([path_pick2cam[-1], path_gotodraw[-1]], pen_item.objcm,
                                                    objrelpos, objrelrot)
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

    '''
    time cost summary
    '''
    time_cost_dict_final = {}
    for k, v in time_cost_dict_gp.items():
        if k in time_cost_dict_mp.keys():
            time_cost_dict_final[k] = \
                {'time_cost_gp': time_cost_dict_gp[k]['time_cost'],
                 'time_cost_mp': time_cost_dict_mp[k]['time_cost'],
                 'flag_gp': time_cost_dict_gp[k]['flag'],
                 'flag_mp': time_cost_dict_mp[k]['flag']}
            if 'time_cost_draw' in time_cost_dict_mp[k].keys():
                time_cost_dict_final[k]['time_cost_draw'] = time_cost_dict_mp[k]['time_cost_draw']
                time_cost_dict_final[k]['flag_draw'] = time_cost_dict_mp[k]['flag_draw']
            else:
                time_cost_dict_final[k]['time_cost_draw'] = 0
                time_cost_dict_final[k]['flag_draw'] = False
        else:
            time_cost_dict_final[k] = \
                {'time_cost_gp': time_cost_dict_gp[k]['time_cost'],
                 'time_cost_mp': 0,
                 'time_cost_draw': 0,
                 'flag_draw': False,
                 'flag_gp': False,
                 'flag_mp': False}
    for k, v in time_cost_dict_final.items():
        print(k, v)
    pickle.dump(time_cost_dict_final, open(folder_path + 'time_cost.pkl', 'wb'))

    print('cam pose success count:', success_cnt)
    gu.cnt_pos_in_gmap(gmap, pen_gid_list_final)
    print('time cost(all)', time.time() - time_start)
    print('time cost(all planning)', time.time() - time_get_grasp)
