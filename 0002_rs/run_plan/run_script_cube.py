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
    paintingobj_f_name = 'cube'
    exp_name = 'cube'
    phoxi_f_name = None
    folder_name = f'exp_{exp_name}/'
    motion_name = 'draw_circle'

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
        # setting_real(phxilocator, phoxi_f_name, pen_stl_f_name, paintingobj_f_name, resolution)

    motion_dict = pickle.load(open(f'{config.MOTIONSCRIPT_REL_PATH}/{folder_name}/{motion_name}.pkl', 'rb'))
    for k, v in motion_dict.items():
        print(k, len(v[2]))
        objrelpos, objrelrot, path = v
        mp_lft.ah.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
        mp_lft.rbth.plot_armjnts(path)
        grasp = mp_lft.load_grasp(config.PEN_STL_F_NAME.split(".stl")[0], k)
        # path_new, _ = mp_lft.refine_continuouspath_by_posdiff(objrelpos, objrelrot, path, grasp, pen_cm,
        #                                                       np.asarray([1, 1, 10]))
        # mp_lft.ah.show_animation_hold(path_new, pen_cm, objrelpos, objrelrot)
        # path_new = mp_lft.refine_continuouspath_rgt(path, 30, objrelpos, objrelrot, [0, 0, 1], grasp, pen_cm)
        # path_new = mp_lft.refine_continuouspath_lft(path, 16, objrelpos, objrelrot, [0, 0, 1], grasp, pen_cm)
        # mp_lft.ah.show_animation_hold(path_new, pen_cm, objrelpos, objrelrot)

        # base.run()

    base.run()

    for i, a in enumerate(path):
        pen_objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=a)
        mp_lft.ah.show_objmat4(pen_cm, pen_objmat4, rgba=(1, 1, 1, .1))
        base.pggen.plotSphere(base.render, pen_objmat4[:3, 3], rgba=(1, 0, 0, .005 * i))

    print(len(path), mp_lft.get_path_cost(path))
    base.run()
