import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm
import numpy as np
import math

if __name__ == '__main__':
    import os

    base = wd.World(cam_pos=[10, 0, 5], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    rbt = ur3ed.UR3EDual()
    hnd = rtqhe.RobotiqHE()
    # rbt.gen_meshmodel().attach_to(base)
    hnd.gen_meshmodel().attach_to(base)
    hnd.grip_at_with_jcpose()
    base.run()