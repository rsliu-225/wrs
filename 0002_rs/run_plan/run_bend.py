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

    '''
    init class
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    base.run()
