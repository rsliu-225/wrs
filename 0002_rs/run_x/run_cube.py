import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
from utils.run_script_utils import *
import utiltools.robotmath as rm

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    exp_name = "cube"
    exp_id = "rot"
    folder_name = f"exp_{exp_name}/"
    total_len = 39600

    pen_cm = el.loadObj(config.PEN_STL_F_NAME)
    cube_cm = el.loadObj('cube.stl', pos=(800, 400, 780))

    id_list = config.ID_DICT[exp_name]

    '''
    init class
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="rgt")
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    mp_lft.add_obs(cube_cm)

    # mp_x_lft.goto_init_x()
    # mp_x_rgt.goto_init_x()

    '''
    run real robot
    '''
    draw_path_name = 'draw_circle'
    model_name = config.PEN_STL_F_NAME.split(".stl")[0]
    grasp = mp_x_lft.load_grasp(model_name, id_list[0])

    motion_dict = pickle.load(open(f'{config.MOTIONSCRIPT_REL_PATH}/{folder_name}/{draw_path_name}.pkl', 'rb'))
    objrelpos, objrelrot, path_draw = motion_dict[id_list[0]]
    # mp_lft.ah.show_animation_hold(path_draw, pen_cm, objrelpos, objrelrot)

    if os.path.exists(f"{config.ROOT}/log/exp/{exp_name}_state.pkl"):
        state = pickle.load(open(f"{config.ROOT}/log/exp/{exp_name}_state.pkl", "rb"))
        path_draw_sid = int(np.ceil(sum([int(s) for s in state]) / total_len * len(path_draw))) + 1
        print("load drawing state:", state, path_draw_sid)
        if path_draw_sid >= len(path_draw) - 1:
            state = []
            path_draw_sid = 0
            pickle.dump(state, open(f"{config.ROOT}/log/exp/{exp_name}_state.pkl", "wb"))
            print("Start again")
        else:
            try:
                path_draw = pickle.load(open(f"{config.ROOT}/log/exp/{exp_name}_path.pkl", "rb"))
            except:
                pass

            mp_x_lft.move_up_x(pen_cm, objrelpos, objrelrot, length=50)
            info = pickle.load(open(f"{config.ROOT}/log/exp/{exp_name + '_' + exp_id}.pkl", "rb"))
            objmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, path_draw[path_draw_sid])
            baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
            # f = np.dot(baserot, info[0][-1][:3])
            # f = np.asarray(np.dot(objmat4[:3, :3], info[0][-1][:3]))
            f = np.asarray([0, 0, 1])
            n0 = objmat4[:3, 0]
            # path_draw = mp_lft.refine_continuouspath_lft(path_draw, path_draw_sid, objrelpos, objrelrot, n0,
            #                                                         grasp, pen_cm)
            path_draw = mp_lft.refine_continuouspath_rgt(path_draw, path_draw_sid, objrelpos, objrelrot, f,
                                                                    grasp, pen_cm)
            mp_lft.ah.show_animation_hold(path_draw, pen_cm, objrelpos, objrelrot)
            # base.run()
            # '''
            # go to next point
            # '''
            # skip_num = 3
            # path_draw = path_draw[path_draw_sid + skip_num:]
            # state.append(int(total_len * skip_num / len(path_draw)))
            # pickle.dump(state, open(f"{config.ROOT}/log/exp/{exp_name}_state.pkl", "wb"))
            # print('dump state:', state)
    else:
        path_draw_sid = 0

    '''
    refine draw path
    '''
    draw_primitive_armjnts = mp_lft.get_tool_primitive_armjnts(path_draw[0], objrelrot, length=35)
    mp_x_lft.goto_armjnts_hold_x(grasp, pen_cm, objrelpos, objrelrot, draw_primitive_armjnts)
    mp_x_lft.goto_armjnts_x(path_draw[0])
    time.sleep(2)

    # path_draw, _ = \
    #     mp_x_lft.refine_path_by_attatchfirm(objrelpos, objrelrot, path_draw, pen_cm, grasp, forcethreshold=3)
    # time.sleep(1)

    toolrelpose = mp_lft.homomat2vec(objrelpos, objrelrot)
    mp_x_lft.force_controller.passive_move(path_draw, toolrelpose)
    mp_x_lft.move_up_x(pen_cm, objrelpos, objrelrot, length=35)
