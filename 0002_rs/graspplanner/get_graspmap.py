import pickle

import config
from localenv import envloader as el
import motionplanner.motion_planner as m_planner

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")

    '''
    init planner
    '''
    motion_planner_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="rgt")
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")

    obj_name = "pentip_short"
    objcm = el.loadObj(obj_name + ".stl")
    pregrasp_list = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/" + obj_name + "_pregrasps.pkl", "rb"))
    folder_path = config.ROOT + "/graspplanner/graspmap/"

    x_range = (650, 800)
    y_range = (200, 350)
    z_range = (1000, 1100)
    roll_range = (0, 20)
    pitch_range = 180
    yaw_range = (0, 180)
    pos_step = 50
    rot_step = 20

    # set obj goal position/rotation
    # objmat4_final_list = gu.get_candidate_objmat4_list(x_range, y_range, z_range, roll_range, pitch_range, yaw_range,
    #                                                    pos_step, rot_step)
    objmat4_final_list = pickle.load(open(folder_path + obj_name + "_objmat4_list.pkl", "rb"))
    motion_planner_lft.show_objmat4_list(objmat4_final_list, objcm, rgba=(0, 1, 0, .5))
    base.run()
    # pickle.dump(objmat4_final_list, open(folder_path + obj_name + "_objmat4_list.pkl", "wb"))
    # objmat4finalngrasp_pair_dict = gu.get_graspmap(motion_planner_lft, objcm, objmat4_final_list, pregrasp_list)
    #
    # pickle.dump(objmat4finalngrasp_pair_dict, open(folder_path + obj_name + "_graspmap.pkl", "wb"))
