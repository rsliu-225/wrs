import config
import manipulation.grip.robotiqhe.robotiqhe as rtqhe
import graspplanner.grasp_planner as gp
import pandaplotutils.pandactrl as pc

if __name__ == '__main__':
    base = pc.World(camp=[4000, 500, 3000], lookatpos=[0, 0, 0])
    hndfa = rtqhe.HandFactory()
    grasp_planner = gp.GraspPlanner(hndfa)

    '''
    egg.stl
    '''
    # stl_f_name = "egg.stl"
    # pregrasp_list = \
    #     grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, 5), finger_normal=(0, 1, 0),
    #                                              hand_normal=(0, 0, -1), jawwidth=40,
    #                                              obj=grasp_planner.load_objcm(stl_f_name),
    #                                              rotation_range=(-75, 75), toggledebug=True)

    '''
    pen.stl
    '''
    # stl_f_name = "pen.stl"
    # pregrasp_list = []
    # for i in range(-40, 20, 20):
    #     pregrasp_list.extend(
    #         grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, i), finger_normal=(0, 1, 0),
    #                                                  hand_normal=(-1, 0, 0), jawwidth=20,
    #                                                  obj=grasp_planner.load_objcm(stl_f_name),
    #                                                  rotation_range=(-45, 45), toggledebug=True))

    '''
    pentip.stl
    '''
    # stl_f_name = "pentip.stl"
    # pregrasp_list = []
    # for i in range(20, 160, 40):
    #     pregrasp_list.extend(
    #         grasp_planner.define_grasp_with_rotation(grasp_coordinate=(i, 0, -10), finger_normal=(0, 1, 0),
    #                                                  hand_normal=(0, 0, -1), jawwidth=20,
    #                                                  obj=grasp_planner.load_objcm(stl_f_name),
    #                                                  rotation_range=(-75, 75), toggledebug=True))
    #
    # print(len(pregrasp_list))
    # grasp_planner.write_pregrasps(stl_f_name, pregrasp_list, mode="hndovr")
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
    grasp_planner.show_grasp(grasp_list[:30], obj, rgba=(1, 0, 0, .2))

    base.run()
