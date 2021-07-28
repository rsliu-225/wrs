from run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    pen_stl_f_name = "pentip.stl"
    folder_name = "real_exp_cylinder/"
    phoxi_f_name = "phoxi_tempdata_cylinder.pkl"
    phoxi_f_name_grasp = "phoxi_tempdata_grasp_cylinder.pkl"
    amat_f_name = "/phoxi_calibmat_0615.pkl"

    pen_cm = el.loadObj(pen_stl_f_name)

    '''
    init class
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    motion_planner_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    motion_planner_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=amat_f_name)

    '''
    run real robot
    '''
    id_pair_list = [[42, 413], [1, 13]]  # real_exp_cylinder
    draw_path_name = "draw_pig_ms"
    model_name = pen_stl_f_name.split(".stl")[0]

    rbtx.opengripper(armname="lft")
    # motion_planner_x_rgt.goto_init_x()
    motion_planner_x_lft.goto_init_x()

    for i, id_list in enumerate(id_pair_list):
        grasp = motion_planner_x_lft.load_grasp(model_name, id_list[0])
        objmat4_cam = motion_planner_x_lft.load_objmat4(model_name, id_list[1])

        objrelpos, objrelrot, path_gotopick = load_motion_sgl("gotopick_pen", folder_name, id_list)
        if i > 0:
            path_up = motion_planner_lft.get_moveup_path(path_gotopick[-1], pen_cm, objrelpos, objrelrot, length=80)
            path_gotopick_new = motion_planner_lft.plan_start2end(end=path_up[-1],
                                                                  start=motion_planner_x_lft.get_armjnts())
            motion_planner_x_lft.movepath(path_gotopick_new)
            motion_planner_x_lft.movepath(path_up[::-1])
        else:
            motion_planner_x_lft.movepath(path_gotopick)

        print("close gripper")
        rbtx.closegripper(forcepercentage=100, armname="lft")

        objrelpos, objrelrot, path_pick2cam_pen = load_motion_sgl("pick2cam_pen", folder_name, id_list)
        motion_planner_x_lft.movepath(path_pick2cam_pen)

        '''
        trigger frame
        '''
        grayimg, depthnparray_float32, pcd = ru.get_phoxi_info(phoxi_f_name=phoxi_f_name_grasp, load=False)
        tcppos, tcprot = motion_planner_x_lft.get_ee()

        objrelpos, objrelrot, path_cam2place = load_motion_sgl("cam2place_pen", folder_name, id_list)
        motion_planner_x_lft.movepath(path_cam2place)

        '''
        mulit-stroke
        '''
        draw_dict = load_motion_sgl(draw_path_name, folder_name, id_list)
        transmat = motion_planner_x_lft.get_transmat_by_vision(phxilocator, phoxi_f_name_grasp, pen_stl_f_name,
                                                               objmat4_cam, load=True, armjnts=path_cam2place[0],
                                                               toggledubug=False)
        # transmat = np.eye(4)
        path = []
        if i > 0:
            draw_dict_items = list(draw_dict.items())[10:]
        else:
            draw_dict_items = list(draw_dict.items())[:10]

        for stroke_key, v in draw_dict_items:
            print("------", stroke_key, "--------")
            objrelpos, objrelrot, path_stroke = v
            grasp_refined, objrelpos_refined, objrelrot_refined, path_stroke = \
                motion_planner_x_lft.refine_continuouspath_by_transmat(objrelpos, objrelrot, path_stroke, grasp,
                                                                       pen_cm, transmat)

            draw_primitive_armjnts = motion_planner_lft.get_tool_primitive_armjnts(path_stroke[0], objrelrot, length=15)
            motion_planner_x_lft.goto_armjnts_x(draw_primitive_armjnts)
            path_stroke = motion_planner_x_lft.refine_path_by_attatchfirm(objrelpos, objrelrot, path_stroke,
                                                                          pen_cm, grasp)
            if path != []:
                path_gotodraw = motion_planner_lft.plan_start2end(start=path[-1], end=path_stroke[0])
            else:
                path_gotodraw = []

            path_up = motion_planner_lft.get_moveup_path(path_stroke[-1], pen_cm, objrelpos, objrelrot, length=30)
            path.extend(path_gotodraw + path_stroke + path_up)
            # motion_planner_x_lft.movepath(path_stroke + path_up)
            toolrelpose = motion_planner_lft.homomat2vec(objrelpos_refined, objrelrot_refined)
            motion_planner_x_lft.passive_move(path_stroke, toolrelpose)
            motion_planner_x_lft.movepath(path_up)

        # motion_planner_x_lft.show_animation_hold(path, pen_cm, objrelpos, objrelrot)
        # base.run()

        '''
        return pen
        '''
        objrelpos, objrelrot, path_picknplace_pen = load_motion_sgl("picknplace_pen", folder_name, id_list)

        motion_planner_x_lft.goto_armjnts_x(path_picknplace_pen[::-1][0])
        motion_planner_x_lft.movepath(path_picknplace_pen[::-1])

        print("open gripper")
        rbtx.opengripper(armname="lft")

        motion_planner_x_lft.move_up_x(pen_cm, objrelpos, objrelrot, length=80)
