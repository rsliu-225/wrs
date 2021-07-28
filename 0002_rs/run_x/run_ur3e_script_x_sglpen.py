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

    # exp_name = "cylinder_cad"
    # exp_name = "force"
    exp_name = "raft"
    # exp_name = "leg"
    # exp_name = "bunny"
    # exp_name = "helmet"
    folder_name = "exp_" + exp_name + "/"
    phoxi_f_name = "phoxi_tempdata_" + exp_name + ".pkl"
    phoxi_f_name_grasp = "phoxi_tempdata_grasp_" + exp_name + ".pkl"

    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    id_list = config.ID_DICT[exp_name]

    '''
    init class
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
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
    objmat4_cam = mp_x_lft.load_objmat4(model_name, id_list[1])

    objrelpos, objrelrot, path_gotopick_pen = load_motion_sgl("gotopick_pen", folder_name, id_list)
    _, _, path_pick2cam_pen = load_motion_sgl("pick2cam_pen", folder_name, id_list)
    _, _, path_cam2place_pen = load_motion_sgl("cam2place_pen", folder_name, id_list)
    _, _, path_picknplace_pen = load_motion_sgl("picknplace_pen", folder_name, id_list)
    _, _, path_gotodraw = load_motion_sgl("gotodraw", folder_name, id_list)
    _, _, path_draw = load_motion_sgl(draw_path_name, folder_name, id_list)
    path_draw.append(path_draw[0])

    # '''
    # go to init
    # '''
    # mp_x_rgt.goto_init_x()
    # rbtx.opengripper(armname="lft")
    # mp_x_lft.goto_init_x()
    #
    # '''
    # pick and place
    # '''
    # mp_x_lft.movepath(path_gotopick_pen)
    # print("close gripper")
    # rbtx.closegripper(forcepercentage=100, armname="lft")
    # mp_x_lft.movepath(path_picknplace_pen)

    '''
    draw with pre-saved transfer matrix
    '''
    # transmat = \
    #     mp_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, config.PEN_STL_F_NAME, objmat4_cam, load=True,
    #                                     armjnts=path_cam2place_pen[0], toggledubug=False)
    transmat = np.eye(4)
    # transmat = pickle.load(open("./transmat_temp.pkl", "rb"))
    grasp_refined, objrelpos_refined, objrelrot_refined, path_draw_new, _ = \
        mp_x_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_draw, grasp, pen_cm, transmat)
    path_draw_new = path_draw_new[50:]
    draw_primitive_armjnts = mp_lft.get_tool_primitive_armjnts(path_draw_new[0], objrelrot, length=20)
    mp_x_lft.goto_armjnts_x(draw_primitive_armjnts)
    # mp_x_lft.goto_armjnts_x(path_draw_new[0])
    # time.sleep(3)
    for armjnts in path_draw:
        penmat4 = mp_lft.get_world_objmat4(objrelpos_refined, objrelrot_refined, armjnts=armjnts)
        base.pggen.plotSphere(base.render, penmat4[:3, 3], rgba=(1, 0, 0, 1))

    path_draw_new, path_mask_new = \
        mp_x_lft.refine_path_by_attatchfirm(objrelpos_refined, objrelrot_refined, path_draw_new,
                                            pen_cm, grasp, path_mask=None, forcethreshold=3)

    # for armjnts in path_draw_new:
    #     penmat4 = mp_lft.get_world_objmat4(objrelpos_refined, objrelrot_refined, armjnts=armjnts)
    #     base.pggen.plotSphere(base.render, penmat4[:3, 3], rgba=(0, 0, 1, 1))
    # for armjnts in path_draw_new:
    #     penmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=armjnts)
    #     base.pggen.plotSphere(base.render, penmat4[:3, 3], rgba=(1, 0, 1, 1))
    # for armjnts in path_draw:
    #     penmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=armjnts)
    #     base.pggen.plotSphere(base.render, penmat4[:3, 3], rgba=(0, 1, 0, 1))

    toolrelpose = mp_x_lft.homomat2vec(objrelpos_refined, objrelrot_refined)

    mp_x_lft.force_controller.passive_move(path_draw_new, toolrelpose)
    # mp_lft.ah.show_animation(path)
    # base.run()
    # mp_x_lft.movepath(path_draw_new)

    mp_x_lft.move_up_x(pen_cm, objrelpos_refined, objrelrot_refined, direction=[0, 0, 1])
    base.run()

    '''
    mulit-stroke
    '''
    # draw_path_name = "draw_pig_ms"
    # draw_dict = load_motion_dict(draw_path_name, folder_name, id_list)
    # transmat = pickle.load(open("transmat_temp.pkl", "rb"))
    #
    # path = []
    # for stroke_key, v in draw_dict.items():
    #     print("------", stroke_key, "--------")
    #     objrelpos, objrelrot, path_stroke = v
    #     grasp_refined, objrelpos_refined, objrelrot_refined, path_stroke = \
    #         motion_planner_x_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_stroke, grasp,
    #                                                                pen_cm, transmat)
    #
    #     draw_primitive_armjnts = motion_planner_lft.get_tool_primitive_armjnts(path_stroke[0], objrelrot, length=10)
    #     motion_planner_x_lft.goto_armjnts_x(draw_primitive_armjnts)
    #     path_stroke = motion_planner_x_lft.refine_path_by_attatchfirm(objrelpos, objrelrot, path_stroke,
    #                                                                   pen_cm, grasp)
    #     if path != []:
    #         path_gotodraw = motion_planner_lft.plan_start2end(start=path[-1], end=path_stroke[0])
    #     else:
    #         path_gotodraw = []
    #
    #     path_up = motion_planner_lft.get_moveup_path(path_stroke[-1], pen_cm, objrelpos, objrelrot, length=30)
    #     path.extend(path_gotodraw + path_stroke + path_up)
    #     # motion_planner_x_lft.movepath(path_stroke + path_up)
    #     toolrelpose = motion_planner_lft.homomat2vec(objrelpos_refined, objrelrot_refined)
    #     motion_planner_x_lft.passive_move(path_stroke, toolrelpose)
    #     motion_planner_x_lft.movepath(path_up)
    #
    # motion_planner_x_lft.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
    # base.run()
