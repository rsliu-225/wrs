from utils.run_script_utils import *
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.phoxi as phoxi
import utils.phoxi_locator as pl

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    '''
    init planner
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    param
    '''
    paintingobj_stl_f_name = "cylinder"
    # paintingobj_stl_f_name = None
    resolution = 1

    exp_name = "cylinder_mtp"
    folder_path = os.path.join(config.MOTIONSCRIPT_REL_PATH + f"exp_{exp_name}/")
    phoxi_f_name = f"phoxi_tempdata_{exp_name}.pkl"
    phoxi_f_name_grasp = f"phoxi_tempdata_grasp_{exp_name}.pkl"
    folder_name = f"exp_{exp_name}/"

    '''
    run sim
    '''
    id_list = get_script_id_dict_multiplepen(folder_name)
    grasp_id = id_list[0]
    objmat4_id = id_list[1]

    pen_cm = el.loadObj(f_name=config.PEN_STL_F_NAME)

    setting_real_simple(phoxi_f_name, phxilocator.amat)
    # setting_real(phxilocator, phoxi_f_name, config.PEN_STL_F_NAME, paintingobj_stl_f_name+".stl", resolution)
    path_all = []

    path_draw_name_list = ["draw_D_ms", "draw_R_ms", "draw_A_ms", "draw_W_ms"]
    # path_draw_name_list = ["draw_circle", "draw_circle", "draw_circle", "draw_circle"]

    for pen_id, path_draw_name in enumerate(path_draw_name_list):
        pen_id = str(pen_id)
        motion_seq_sub = ["pick2cam_pen" + pen_id, "cam2place_pen", "gotodraw"]
        objrelpos, objrelrot, path_gotopick = load_motion_seq(["gotopick_pen" + pen_id], folder_name, id_list)

        if path_all != []:
            grasp = motion_planner_lft.load_grasp(config.PEN_STL_F_NAME.split(".stl")[0], grasp_id)
            path_up = motion_planner_lft.get_moveup_path(path_gotopick[-1], pen_cm, objrelpos, objrelrot, length=50)
            path_gotopick_new = motion_planner_lft.plan_start2end(end=path_up[-1], start=path_all[-1])
            if path_gotopick_new is None:
                print("planning failed")
                path_all.extend(path_gotopick)
            else:
                path_all.extend(path_gotopick_new + path_up[::-1])

        _, _, path = load_motion_seq(motion_seq_sub, folder_name, id_list)
        path_all.extend(path)

        # path_draw = load_motion_sgl(path_draw_name, folder_name, id_list)
        # for k, v in path_draw.items():
        #     _, _, path_stroke = v
        #     path_up = motion_planner_lft.get_moveup_path(path_stroke[-1], pen_cm, objrelpos, objrelrot, length=50)
        #     path_all.extend(path_stroke + path_up)

        _, _, path_draw = load_motion_sgl(path_draw_name, folder_name, id_list)
        path_up = motion_planner_lft.get_moveup_path(path_draw[-1], pen_cm, objrelpos, objrelrot, length=50)
        path_all.extend(path_draw + path_up)

        motion_seq_sub = ["picknplace_pen" + pen_id]
        _, _, path = load_motion_seq(motion_seq_sub, folder_name, id_list)

        path_all.extend(path[::-1])
        path_all.extend(motion_planner_lft.get_moveup_path(path_all[-1], pen_cm, objrelpos, objrelrot, length=50))

    motion_planner_lft.ah.show_animation_hold(path_all, pen_cm, objrelpos, objrelrot)
    base.run()
