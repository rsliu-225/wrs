import config
import motionplanner.motion_planner as m_planner
import run_config as rconfig
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
from utils.run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')

    '''
    init planner
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='rgt')
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='lft')
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    param
    '''
    # paintingobj_f_name = 'cube'
    # paintingobj_f_name = 'cylinder'
    # paintingobj_f_name = 'bucket'
    paintingobj_f_name = None

    # exp_name = 'bucket'
    # exp_name = 'cylinder_cad'
    # exp_name = 'helmet'
    # exp_name = 'cube'
    # exp_name = 'force'
    # exp_name = 'raft'
    exp_name = 'leg'
    # exp_name = 'bunny'
    phoxi_f_name = f'phoxi_tempdata_{exp_name}.pkl'
    # phoxi_f_name = None
    folder_name = f'exp_{exp_name}/'
    # folder_name = 'tc_' + exp_name + '_nlopt/'
    motion_seq = ['gotopick_pen', 'ClOSEGRIPPER', 'pick2cam_pen', 'TRIGGERFRAME', 'cam2place_pen',
                  'gotodraw', 'draw_spiral']
    # motion_seq = ['gotopick_pen', 'ClOSEGRIPPER', 'picknplace_pen', 'gotodraw', 'draw_circle']
    # motion_seq = ['draw_circle']

    '''
    run sim
    '''
    id_list = input_script_id(folder_name)
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)

    if paintingobj_f_name is not None:
        if paintingobj_f_name in rconfig.OBJ_POS.keys():
            setting_sim(stl_f_name=paintingobj_f_name, pos=rconfig.OBJ_POS[paintingobj_f_name])
        else:
            setting_sim(stl_f_name=paintingobj_f_name, pos=(800, 100, 780))
    else:
        setting_real_simple(phoxi_f_name, phxilocator.amat)
        # setting_real(phxilocator, phoxi_f_name, config.PEN_STL_F_NAME, paintingobj_f_name)

    objrelpos, objrelrot, path = load_motion_seq(motion_seq, folder_name, id_list)
    # show_drawmotion_ms(mp_lft, pen_cm, os.path.join(config.ROOT, 'motionscript', folder_name, 'draw_pig_ms.pkl'),
    #                    [id_list[0]])
    # _, _, path_gotopick = load_motion_sgl('gotopick_pen', folder_name, id_list)
    # _, _, path_pick2cam_pen = load_motion_sgl('pick2cam_pen', folder_name, id_list)
    # _, _, path_cam2place_pen = load_motion_sgl('cam2place_pen', folder_name, id_list)
    # _, _, path_picknplace_pen = load_motion_sgl('picknplace_pen', folder_name, id_list)
    # _, _, path_gotodraw = load_motion_sgl('gotodraw', folder_name, id_list)
    _, _, path_draw = load_motion_sgl(motion_seq[-1], folder_name, id_list)

    # mp_lft.ah.show_animation_hold(path_pick2cam_pen, pen_cm, objrelpos, objrelrot)
    # mp_lft.ah.show_armjnts(armjnts=path_pick2cam_pen[-1], rgba=(1, 0, 0, .5))
    # mp_lft.ah.show_armjnts(armjnts=path_pick2cam_pen[-2], rgba=(0, 1, 0, .5))
    # base.run()

    for i, a in enumerate(path_draw):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=a)
        mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 1, 1, .1))
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 0, 0, .005 * i))
    mp_lft.ah.show_armjnts_with_obj(path_draw[0], pen_cm, objrelpos, objrelrot, rgba=(1, 0, 0, .5))
    mp_lft.ah.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
    print(len(path), mp_lft.get_path_cost(path))
    mp_lft.rbth.plot_armjnts(path)
    base.run()

    '''
    run dual script
    '''
    # pen, egg = setting_egg()
    #
    # # load motion script
    # motion_seq_lft = ['gotopick_pen', 'ClOSEGRIPPER', 'picknplace_pen', 'draw']
    # motion_seq_rgt = ['gotopick_paintingobj', 'ClOSEGRIPPER', 'picknplace_paintingobj']
    #
    # folder_name = 'egg_circle/withmodel/'
    # id_list_lft = input_script_id(folder_name, armname='lft')
    # id_list_rgt = input_script_id(folder_name, armname='rgt')
    #
    # objrelpos_rgt, objrelrot_rgt, path_rgt = load_motion_seq(motion_seq_rgt, folder_name, id_list_rgt, 'rgt')
    # path_rgt = [[rbt.initlftjnts, jnts] for jnts in path_rgt]
    #
    # objrelpos_lft, objrelrot_lft, path_lft = load_motion_seq(motion_seq_lft, folder_name, id_list_lft, 'lft')
    # path_lft = [[jnts, path_rgt[-1][1]] for jnts in path_lft]
    #
    # path_dual = path_rgt + path_lft
    # mp_lft.show_animation_hold_dual(path_dual, pen, objrelpos_lft, objrelrot_lft, egg, objrelpos_rgt, objrelrot_rgt)
