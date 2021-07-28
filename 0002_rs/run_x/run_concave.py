import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
from utils.run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    exp_name = "box"
    folder_name = "exp_" + exp_name + "/"
    phoxi_f_name = "phoxi_tempdata_" + exp_name + ".pkl"

    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    id_list = config.ID_DICT[exp_name]

    '''
    init class
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    run real robot
    '''
    draw_path_name = 'draw_circle'
    # draw_path_name = 'draw_star'

    model_name = config.PEN_STL_F_NAME.split(".stl")[0]
    grasp = mp_x_lft.load_grasp(model_name, id_list[0])

    objrelpos, objrelrot, path_gotopick_pen = load_motion_sgl("gotopick_pen", folder_name, id_list)
    _, _, path_picknplace_pen = load_motion_sgl("picknplace_pen", folder_name, id_list)
    _, _, path_gotodraw = load_motion_sgl("gotodraw", folder_name, id_list)
    _, _, path_draw = load_motion_sgl(draw_path_name, folder_name, id_list)

    '''
    go to init
    '''
    mp_x_rgt.goto_init_x()
    rbtx.opengripper(armname="lft")
    mp_x_lft.goto_init_x()

    '''
    pick and place
    '''
    mp_x_lft.movepath(path_gotopick_pen)
    print("close gripper")
    rbtx.closegripper(forcepercentage=100, armname="lft")
    mp_x_lft.movepath(path_picknplace_pen)

    draw_primitive_armjnts = mp_lft.get_tool_primitive_armjnts(path_draw[0], objrelrot, length=20)
    mp_x_lft.goto_armjnts_x(draw_primitive_armjnts)

    path_draw_new, path_mask_new = \
        mp_x_lft.refine_path_by_attatchfirm(objrelpos, objrelrot, path_draw, pen_cm, grasp, path_mask=None,
                                            forcethreshold=2.5)

    toolrelpose = mp_lft.homomat2vec(objrelpos, objrelrot)
    # mp_x_lft.force_controller.passive_move(path_draw_new, toolrelpose)
    mp_x_lft.movepath(path_draw_new)
    mp_x_lft.move_up_x(pen_cm, objrelpos, objrelrot, direction=[0, -1, 0])
