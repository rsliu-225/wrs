import numpy as np

import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import basis.robot_math as rm
from utils.run_script_utils import *
import visualization.panda.world as wd

if __name__ == '__main__':
    """
    set up env and param
    """
    base = wd.World(cam_pos=[4, 0, 1.7], lookat_pos=[0, 0, 1])
    env = None
    rbt = el.loadYumi()
    rbtx = el.loadYumiX()

    """
    init class
    """
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="rgt_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")

    mp_x_lft.goto_init_x()
    mp_x_rgt.goto_init_x()

    mp_x_rgt.goto_armjnts_x(armjnts=np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
                                              0.13788101, 1.51669112]))
