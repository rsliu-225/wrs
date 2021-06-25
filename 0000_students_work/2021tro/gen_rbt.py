import math

import numpy as np

import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=[10, 0, 5], lookat_pos=[0, 0, 1])
    gm.gen_frame().attach_to(base)

    pencm = cm.CollisionModel(initor="./objects/pentip.stl")
    pencm.set_rgba([.3, .3, .3, .3])
    pencm.set_rotmat(rm.rotmat_from_euler(math.pi, 0, 0))
    pencm.attach_to(base)

    cylindercm = cm.CollisionModel(initor="./objects/cylinder.stl")
    cylindercm.set_rgba([.3, .3, .3, .3])
    cylindercm.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
    cylindercm.attach_to(base)

    rbt = ur3ed.UR3EDual()
    # rbt.gen_meshmodel().attach_to(base)

    hnd = rtqhe.RobotiqHE()
    hnd.grip_at_with_jcpose((.1, 0, 0), np.eye(3), .02)
    hnd.gen_meshmodel().attach_to(base)

    # hnd.grip_at_with_jcpose()
    gm.gen_cone(radius=.05).attach_to(base)
    base.run()
