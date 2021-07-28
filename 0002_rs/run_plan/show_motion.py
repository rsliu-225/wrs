import motionplanner.motion_planner as m_planner
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
from utils.run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    # base, env = el.loadEnv_wrs(camp=[4000, -1200, 1700], lookatpos=[200, 0, 1200])
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')
    amat_f_name = '/phoxi_calibmat_210527.pkl'

    '''
    init planner
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='rgt')
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='lft')
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=amat_f_name)

    '''
    param
    '''
    paintingobj_stl_f_name = 'cylinder.stl'
    # paintingobj_stl_f_name = None
    resolution = 1
    exp_name = 'cylinder_cad'
    phoxi_f_name = 'phoxi_tempdata_' + exp_name + '.pkl'
    # folder_name = 'tc_' + exp_name + '_ik/'
    folder_name = f'exp_{exp_name}/'
    motion_seq = ['gotopick_pen', 'pick2cam_pen', 'cam2place_pen', 'gotodraw', 'draw_circle']
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    id_list = config.ID_DICT[exp_name]

    motion_dict = load_motion_f(folder_name, f_name='pick2cam_pen.pkl')
    solvable_glist = list(motion_dict.keys())
    cam_objmat4_id_list = []
    for k, v in motion_dict.items():
        if v != {}:
            print(k, list(v.keys()))
            cam_objmat4_id_list.append(list(v.keys())[0])

    glist = mp_lft.load_all_grasp(config.PEN_STL_F_NAME.split('.stl')[0])

    '''
    show grasp
    '''
    hndmat4_list = []
    for i in solvable_glist[:1]:
        _, _, hndmat4 = mp_lft.load_grasp(config.PEN_STL_F_NAME.split('.stl')[0], grasp_id=i)
        print(i, hndmat4)
        hndmat4_list.append(hndmat4)

        cam_objmat4_list = mp_lft.load_all_objmat4(config.PEN_STL_F_NAME.split('.stl')[0], grasp_id=i)
        for objmat4 in cam_objmat4_list:
            mp_lft.ah.show_objmat4(pen_cm, objmat4, rgba=(1, 1, 0, .4))
            # mp_lft.ah.show_hnd_sgl(np.dot(objmat4, hndmat4))

        cam_objmat4_list_f = mp_lft.load_all_objmat4_failed(config.PEN_STL_F_NAME.split('.stl')[0], grasp_id=i)
        for objmat4 in cam_objmat4_list_f:
            mp_lft.ah.show_objmat4(pen_cm, objmat4, rgba=(0.8, 0.8, 0.8, .1))

    '''
    show motion
    '''
    # mp_lft.ah.show_armjnts()
    paintingobj_item = el.loadObjitem(paintingobj_stl_f_name, sample_num=1000000, pos=(800, 100, 780))

    # setting_real_simple(phoxi_f_name, phxilocator.amat)
    setting_real(phxilocator, phoxi_f_name, config.PEN_STL_F_NAME, paintingobj_stl_f_name, resolution)
    # setting_sim(stl_f_name=paintingobj_stl_f_name, pos=(800, 100, 780))

    objrelpos, objrelrot, path = load_motion_seq(motion_seq, folder_name, id_list)

    _, _, path_gotopick = load_motion_sgl('gotopick_pen', folder_name, id_list)
    _, _, path_pick2cam_pen = load_motion_sgl('pick2cam_pen', folder_name, id_list)
    _, _, path_cam2place_pen = load_motion_sgl('cam2place_pen', folder_name, id_list)
    _, _, path_gotodraw = load_motion_sgl('gotodraw', folder_name, id_list)
    _, _, path_draw = load_motion_sgl('draw_circle', folder_name, id_list)
    for i, a in enumerate(path_draw):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=a)
        # mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0, 1 - 0.005 * i))
        # base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 0, 0, 1 - .005 * i))
        mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0, .2))

    # mp_lft.ah.show_path(path_gotopick, fromrgba=[0, 1, 1, .5], torgba=[1, 1, 0, .5], genmnp=True)
    # mp_lft.ah.show_path_hold(path_pick2cam_pen, pen_cm, objrelpos, objrelrot, fromrgba=[0, 1, 0, .5],
    #                          torgba=[1, 1, 0, .5], genmnp=True, jawwidth=20)
    # mp_lft.ah.show_path_hold(path_cam2place_pen, pen_cm, objrelpos, objrelrot, fromrgba=[1, 1, 0, .5],
    #                          torgba=[1, 0, 0, .5], genmnp=True, jawwidth=20)
    # mp_lft.ah.show_path_hold(path_draw, pen_cm, objrelpos, objrelrot, fromrgba=[1, 0, 0, .5], torgba=[1, 0, 0, .5],
    #                          genmnp=True, jawwidth=20)

    # mp_lft.ah.show_path_end(path_gotopick, fromrgba=[0, 1, 1, 1], torgba=[1, 1, 0, 1])
    # mp_lft.ah.show_path_end(path_pick2cam_pen, fromrgba=[0, 1, 0, 1], torgba=[1, 1, 0, 1])
    # mp_lft.ah.show_path_end(path_cam2place_pen, fromrgba=[1, 1, 0, 1], torgba=[1, 0, 0, 1])
    # mp_lft.ah.show_path_end(path_draw, fromrgba=[1, 0, 0, 1], torgba=[1, 0, 0, 1])

    # mp_lft.ah.show_armjnts(rgba=(0, 1, 0, .2), armjnts=path_gotopick[-1], jawwidth=20)
    mp_lft.ah.show_armjnts(rgba=(1, 1, 0, .4), armjnts=path_cam2place_pen[0], jawwidth=20)
    # base.run()

    # mp_lft.ah.show_armjnts(rgba=(1, 0, 0, .4), armjnts=path_draw[0], jawwidth=20)
    # base.run()

    # pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=path_gotopick[-1])
    # mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(0, 1, 0, 1))
    # mp_lft.ah.show_hnd_sgl(np.dot(pen_objmat4, hndmat4_list[0]), rgba=(0, 1, 0, .2))

    pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=path_cam2place_pen[0])
    mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 1, 0, 1))
    base.run()
    # mp_lft.ah.show_hnd_sgl(np.dot(pen_objmat4, hndmat4_list[0]), rgba=(1, 1, 0, .2))

    pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=path_draw[0])
    mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 0, 0, 1))
    # mp_lft.ah.show_hnd_sgl(np.dot(pen_objmat4, hndmat4_list[0]), rgba=(1, 0, 0, .2))

    base.run()
