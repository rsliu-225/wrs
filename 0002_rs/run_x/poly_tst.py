import numpy as np

import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import basis.robot_math as rm
from utils.run_script_utils import *

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt = el.loadUr3e()
    rbtx = el.loadUr3ex()

    '''
    init class
    '''
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="rgt_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")

    mp_x_lft.goto_init_x()
    mp_x_rgt.goto_init_x()
