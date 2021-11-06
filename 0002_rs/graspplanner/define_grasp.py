import config
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import graspplanner.grasp_planner as gp
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    gripper = rtqhe.RobotiqHE()
    grasp_planner = gp.GraspPlanner(gripper)

    '''
    pentip.stl
    '''
    # stl_f_name = "pentip.stl"
    # pregrasp_list = []
    # for i in range(20, 160, 40):
    #     pregrasp_list.extend(
    #         grasp_planner.define_grasp_with_rotation(grasp_coordinate=(i / 1000, 0, 0), finger_normal=(1, 0, 0),
    #                                                  hand_normal=(0, 0, -1), jawwidth=.02,
    #                                                  obj=grasp_planner.load_objcm(stl_f_name),
    #                                                  rotation_range=(-75, 75), toggledebug=True))
    #
    # print(len(pregrasp_list))
    # grasp_planner.write_pregrasps(stl_f_name, pregrasp_list)
    # base.run()

    # '''
    # calibboard.stl
    # '''
    # stl_f_name = "calibboard.stl"
    # pregrasp_list = grasp_planner.define_grasp(grasp_coordinate=(0, -20, 0), finger_normal=(0, 0, 1),
    #                                            hand_normal=(0, 1, 0), jawwidth=10,
    #                                            obj=grasp_planner.load_objcm(stl_f_name), toggledebug=True)
    #
    # grasp_planner.write_pregrasps(stl_f_name, pregrasp_list)
    #
    # base.run()

    '''
    for test
    '''
    stl_f_name = "pentip.stl"
    obj = grasp_planner.load_objcm(stl_f_name)
    grasp_list = grasp_planner.load_pregrasp(stl_f_name)
    grasp_planner.show_grasp(grasp_list[:30], obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)

    base.run()
