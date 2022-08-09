import numpy as np

import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import basis.robot_math as rm
from utils.run_script_utils import *
import visualization.panda.world as wd
import bendplanner.BendSim as b_sim
import bendplanner.bender_config as bconfig

if __name__ == '__main__':
    """
    set up env and param
    """
    base = wd.World(cam_pos=[4, 0, 1.7], lookat_pos=[0, 0, 1])
    env = None
    rbt = el.loadYumi(showrbt=True)
    rbtx = el.loadYumiX()

    f_name = 'chair'
    folder_name = 'stick'
    transmat4 = rm.homomat_from_posrot((.4, .3, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    """
    init class
    """
    bs = b_sim.BendSim(show=True)
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="rgt_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")

    mp_x_lft.goto_init_x()
    mp_x_rgt.goto_init_x()

    # mp_x_rgt.goto_armjnts_x(armjnts=np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
    #                                           0.13788101, 1.51669112]))
    # eepos, eerot = mp_x_lft.get_ee()
    # mp_x_lft.move_up_x()
    # mp_x_rgt.move_up_x(direction=np.array([1, 1, 0]), length=.1)

    '''
    show result
    '''
    pathseq_list = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_pathseq.pkl', 'rb'))

    grasp, pathseq = pathseq_list[1]
    for path in pathseq:
        print(path)
        if len(path) == 1:
            mp_x_lft.goto_armjnts_x(path[0])
        else:
            mp_x_lft.movepath(path)
        time.sleep(10)
    base.run()
