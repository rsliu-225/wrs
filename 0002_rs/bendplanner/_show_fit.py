import bendplanner.BendSim as bsim
import bendplanner.bend_utils as bu
import numpy as np
import pickle
import config
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqe
import visualization.panda.world as wd
import basis.robot_math as rm


def show_pseq(pseq):
    pseq = np.asarray(pseq)
    for i in range(len(pseq) - 1):
        gm.gen_stick(pseq[i], pseq[i + 1], thickness=.0001).attach_to(base)


def gen_torsion(lm, rotm, step):
    pseq = [[0, -.08, 0]]
    rotseq = [np.eye(3)]
    rstep = rotm / (lm / step)
    for i in range(int(lm / step)):
        pseq.append([0, i * step, 0])
        rotseq.append(rm.rotmat_from_axangle((0, 1, 0), rstep * i))
    pseq.append([0, lm + .08, 0])
    rotseq.append(rotseq[-1])
    return np.asarray(pseq), np.asarray(rotseq)


if __name__ == '__main__':
    lfthnd = rtqe.RobotiqHE()
    rgthnd = rtqe.RobotiqHE()

    grasp_list_lft = pickle.load(open(config.PREGRASP_REL_PATH + "plate_pregrasps.pkl", "rb"))
    grasp_list_rgt = pickle.load(open(config.PREGRASP_REL_PATH + "plate_rgt_pregrasps.pkl", "rb"))
    _, _, _, lfthnd_pos, lfthnd_rotmat = grasp_list_lft[0]
    _, _, _, rgthnd_pos, rgthnd_rotmat = grasp_list_rgt[-1]

    base = wd.World(cam_pos=[.2, .2, 0], lookat_pos=[0, 0, 0])
    bs = bsim.BendSim(show=False, cm_type='surface', granularity=np.pi / 90)
    # goal_pseq, goal_rotseq = pickle.load(open(config.ROOT + f'/data/bend/rotpseq/skull2.pkl', 'rb'))

    # goal_pseq = bu.gen_sgl_curve(pseq=np.asarray([[0, 0, 0], [.018, .03, 0], [.06, .06, 0], [.12, 0, 0]]))
    # goal_rotseq = bu.get_rotseq_by_pseq(goal_pseq)

    # goal_pseq = [goal_pseq[0] - (goal_pseq[1] - goal_pseq[0]) * 50] + \
    #             list(goal_pseq) + \
    #             [goal_pseq[-1] - (goal_pseq[-2] - goal_pseq[-1]) * 50]
    # goal_rotseq = bu.get_rotseq_by_pseq(goal_pseq)
    # goal_rotseq = [r.dot(rm.rotmat_from_axangle((0, 1, 0), np.pi / 2)) for r in goal_rotseq]

    goal_pseq, goal_rotseq = gen_torsion(lm=.12, rotm=np.pi / 3, step=.001)

    bs.reset(goal_pseq, goal_rotseq, extend=False)
    bs.show(rgba=(1, 0, 0, 1))

    # gm.gen_frame(goal_pseq[0], goal_rotseq[0], length=.05).attach_to(base)
    objmat4_lft = rm.homomat_from_posrot(goal_pseq[1], goal_rotseq[1])
    lfthndmat4 = np.dot(objmat4_lft, rm.homomat_from_posrot(lfthnd_pos, lfthnd_rotmat))
    lfthnd.fix_to(lfthndmat4[:3, 3], lfthndmat4[:3, :3], jawwidth=.002)
    lfthnd.gen_meshmodel().attach_to(base)

    # gm.gen_frame(goal_pseq[-1], goal_rotseq[-1], length=.05).attach_to(base)
    objmat4_rgt = rm.homomat_from_posrot(goal_pseq[-2], goal_rotseq[-2])
    rgthndmat4 = np.dot(objmat4_rgt, rm.homomat_from_posrot(rgthnd_pos, rgthnd_rotmat))
    rgthnd.fix_to(rgthndmat4[:3, 3], rgthndmat4[:3, :3], jawwidth=.002)
    rgthnd.gen_meshmodel().attach_to(base)

    base.run()

    # plate = cm.CollisionModel(config.ROOT + '/obstacles/plate_curve.stl')
    # plate.set_rgba((.7, .7, .7, 1))
    # plate.attach_to(base)
