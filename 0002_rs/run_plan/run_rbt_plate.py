import copy
import math
import visualization.panda.world as wd
import modeling.geometric_model as gm
import numpy as np
import basis.robot_math as rm
import basis.trimesh as trm
import modeling.collision_model as cm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import time
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import bendplanner.BendRbtPlanner as br_planner
import utils.panda3d_utils as p3u
import bendplanner.InvalidPermutationTree as ip_tree
from scipy import interpolate
import matplotlib.pyplot as plt
import config

if __name__ == '__main__':
    import pickle
    import localenv.envloader as el

    gripper = rtqhe.RobotiqHE()
    # base, env = el.loadEnv_wrs(camp=[.6, -.4, 1.7], lookatpos=[.6, -.4, 1])
    # base, env = el.loadEnv_wrs(camp=[0, 0, 1], lookatpos=[0, 0, 0])
    # base, env = el.loadEnv_wrs()
    # rbt = el.loadUr3e()
    # transmat4 = rm.homomat_from_posrot((.7, -.2, .78 + bconfig.BENDER_H), np.eye(3))

    base, env = el.loadEnv_yumi()
    rbt = el.loadYumi(showrbt=False)
    transmat4 = rm.homomat_from_posrot((.4, .2, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    bs = b_sim.BendSim(show=True)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")

    f_name = 'skull2'

    goal_pseq, goal_rotseq = pickle.load(open(config.ROOT + f'/data/bend/rotpseq/{f_name}.pkl', 'rb'))
    init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    fit_pseq, fit_rotseq = bu.decimate_rotpseq(goal_pseq, goal_rotseq, tor=.0002, toggledebug=False)
    bendset = bu.rotpseq2bendset(fit_pseq, fit_rotseq, bend_r=bs.bend_r, init_l=bs.init_l, toggledebug=True)
    init_rot = fit_rotseq[0]

    bs.reset(init_pseq, init_rotseq, extend=True)
    bs.move_to_org(bu.cal_length(goal_pseq))
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    # bs.show_bendresseq(bendresseq, is_success)

    goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)
    fit_pseq, fit_rotseq = bu.align_with_init(bs, fit_pseq, init_rot, fit_rotseq)

    init_rot = bu.get_init_rot(fit_pseq)
    pickle.dump([goal_pseq, bendset], open(f'{config.ROOT}/bendplanner/planres/plate/{f_name}_bendseq.pkl', 'wb'))
    goal_pseq, bendset = pickle.load(open(f'{config.ROOT}/bendplanner/planres/plate/{f_name}_bendseq.pkl', 'rb'))
    for b in bendset:
        print(b)

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)

    grasp_list = mp.load_all_grasp('plate')
    # grasp_list = grasp_list[:200]

    brp.set_up(bendset, grasp_list, transmat4)
    brp.run(f_name=f_name)
