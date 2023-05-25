import numpy as np

import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import basis.robot_math as rm
from utils.run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.lft_arm_hnd.open_gripper()
    rbt.rgt_arm_hnd.open_gripper()

    pen = el.loadObj(config.PEN_STL_F_NAME)
    cube = el.loadObjitem('cube.stl', pos=(600, 200, 780))
    cube.show_objcm()

    '''
    init class
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="rgt_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    # mp_x_lft.goto_init_x()
    # mp_x_rgt.goto_init_x()

    '''
    get available grasp
    '''
    time_start = time.time()
    pen_grasplist = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split('.stl')[0] + '_pregrasps.pkl', 'rb'))
    print('Num of defined grasps:', len(pen_grasplist))
    objmat4_list = []
    for x in range(600, 800, 5):
        penpos = np.asarray((x, 200, 780))/1000
        objmat4_list.append(rm.homomat_from_posrot(penpos, rm.rotmat_from_axangle((0, 1, 0), -90)))
    for y in range(200, 400, 5):
        penpos = np.asarray((800, y, 780))/1000
        objmat4_list.append(rm.homomat_from_posrot(penpos, rm.rotmat_from_axangle((0, 1, 0), -90)))

    mp_lft.ah.show_objmat4_list(objmat4_list, pen, showlocalframe=True)
    for i, grasp in enumerate(pen_grasplist):
        print(i)
        path = mp_lft.get_continuouspath_ik(None, grasp, objmat4_list, grasp_id=i)
        objrelpos, objrelrot = mp_lft.get_rel_posrot(grasp, objmat4_list[0][:3, 3], objmat4_list[0][:3, :3])
        if path is None:
            continue
        mp_lft.ah.show_animation_hold(path, pen, objrelpos, objrelrot)
        mp_x_lft.goto_armjnts_x(path[0])
        time.sleep(3)
        toolrelpose = mp_lft.homomat2vec(objrelpos, objrelrot)
        mp_x_lft.force_controller.passive_move(path, toolrelpose)
        pickle.dump([objrelpos, objrelrot, path], open(config.MOTIONSCRIPT_REL_PATH + 'exp_force/draw_L.pkl', 'wb'))

        base.run()
