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

    exp_name = "cylinder_mtp"
    phoxi_f_name = f"phoxi_tempdata_{exp_name}.pkl"
    phoxi_f_name_grasp = f"phoxi_tempdata_grasp_{exp_name}.pkl"
    folder_name = f"real_exp_{exp_name}/"

    id_list = config.ID_DICT[exp_name]
    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    pen_model_name = config.PEN_STL_F_NAME.split(".stl")[0]

    '''
    init class
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    grasp = mp_x_lft.load_grasp(pen_model_name, id_list[0])
    objmat4_cam = mp_x_lft.load_objmat4(pen_model_name, id_list[1])

    paintingobj_info = \
        ru.get_obj_from_phoxiinfo_withmodel(phxilocator, "cylinder.stl", load=True,
                                            phoxi_f_name=phoxi_f_name, match_rotz=False,
                                            x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
    paintingobj_cm = paintingobj_info.objcm

    '''
    run all
    '''
    draw_path_name_list = ["draw_D_ms", "draw_R_ms", "draw_A_ms", "draw_W_ms"]
    # draw_path_name_list = ["draw_circle"]
    grayimg, depthnparray_float32, pcd = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name_grasp, load=False)

    mp_x_lft.goto_init_x()
    # motion_planner_x_rgt.goto_init_x()
    rbtx.opengripper(speedpercentage=100, armname="lft")

    objrelpos, objrelrot, path_cam2place = load_motion_sgl("cam2place_pen", folder_name, id_list)

    for i, draw_path_name in enumerate(draw_path_name_list):
        objrelpos, objrelrot, path_gotopick = load_motion_sgl("gotopick_pen" + str(i), folder_name, id_list)
        _, _, path_pick2cam = load_motion_sgl("pick2cam_pen" + str(i), folder_name, id_list)
        _, _, path_picknplace_pen = load_motion_sgl("picknplace_pen" + str(i), folder_name, id_list)
        draw_dict = load_motion_sgl(draw_path_name, folder_name, id_list)

        if i > 0:
            path_up = mp_lft.get_moveup_path(path_gotopick[-1], pen_cm, objrelpos, objrelrot, length=50)
            path_gotopick_new = mp_lft.plan_start2end(end=path_up[-1],
                                                      start=mp_x_lft.get_armjnts())
            mp_x_lft.movepath(path_gotopick_new)
            mp_x_lft.movepath(path_up[::-1])

        else:
            mp_x_lft.movepath(path_gotopick)

        print("close gripper")
        rbtx.closegripper(forcepercentage=50, armname="lft")
        mp_x_lft.movepath(path_pick2cam)

        '''
        trigger frame
        '''
        grayimg, depthnparray_float32, pcd = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name_grasp, load=False)
        tcppos, tcprot = mp_x_lft.get_ee()

        mp_x_lft.movepath(path_cam2place)

        '''
        refine draw path(multiple stroke)
        '''
        # transmat = pickle.load(open("transmat_temp.pkl", "rb"))
        transmat = mp_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, config.PEN_STL_F_NAME,
                                                   objmat4_cam, load=True, armjnts=path_cam2place[0],
                                                   toggledubug=False)
        path = []
        for stroke_key, v in draw_dict.items():
            print("------", stroke_key, "--------")
            objrelpos, objrelrot, path_stroke = v
            grasp_refined, objrelpos_refined, objrelrot_refined, path_stroke_new, path_mask = \
                mp_x_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_stroke, grasp, pen_cm,
                                                           transmat)
            draw_primitive_armjnts = mp_lft.get_tool_primitive_armjnts(path_stroke[0], objrelrot, length=20)
            mp_x_lft.goto_armjnts_x(draw_primitive_armjnts)
            path_draw_new, path_mask_new = \
                mp_x_lft.refine_path_by_attatchfirm(objrelpos_refined, objrelrot_refined, path_stroke_new,
                                                    pen_cm, grasp_refined, path_mask=path_mask, forcethreshold=2.5)
            if path != []:
                path_gotodraw = mp_lft.plan_start2end(start=mp_x_lft.get_armjnts(),
                                                      end=path_stroke[0])
            else:
                path_gotodraw = []

            path_up = mp_lft.get_moveup_path(path_stroke[-1], pen_cm, objrelpos, objrelrot, length=50)
            path.extend(path_gotodraw + path_stroke + path_up)

            toolrelpose = mp_lft.homomat2vec(objrelpos_refined, objrelrot_refined)
            mp_x_lft.force_controller.passive_move(path_stroke, toolrelpose)
            # motion_planner_x_lft.movepath(path_stroke)
            mp_x_lft.movepath(path_up)

        '''
        return pen
        '''
        mp_lft.add_obs(paintingobj_cm)
        mp_x_lft.add_obs(paintingobj_cm)

        start = mp_x_lft.get_armjnts()
        goal = path_picknplace_pen[0]
        objmat4_start = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=start)
        objmat4_goal = mp_x_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=goal)
        path_return_pen = mp_x_lft.plan_picknplace(grasp, [objmat4_start, objmat4_goal], pen_cm, objrelpos,
                                                   objrelrot, start=start, pickupprim_len=10)
        # motion_planner_x_lft.goto_armjnts_x(path_picknplace_pen[::-1][0])
        # motion_planner_x_lft.movepath(path_picknplace_pen[::-1])
        mp_x_lft.movepath(path_return_pen)
        mp_lft.init_obs()
        mp_x_lft.init_obs()

        print("open gripper")
        rbtx.opengripper(forcepercentage=100, armname="lft")

        mp_x_lft.move_up_x(pen_cm, objrelpos, objrelrot, length=70, direction=[0, 0, 1])
        mp_lft.ah.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
    base.run()
