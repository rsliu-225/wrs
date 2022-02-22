import numpy as np

import config
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import graspplanner.grasp_planner as gp
import visualization.panda.world as wd
import modeling.collision_model as cm
import basis.robot_math as rm
import modeling.geometric_model as gm

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
    plate
    '''
    # obj = cm.gen_box(extent=np.asarray([.0015, .1, .01]))
    # obj = cm.gen_stick(epos=np.asarray([0, .1, 0]), thickness=.0015, sections=180)
    # obj.attach_to(base)
    #
    # pregrasp_list = grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, 0), finger_normal=(0, 0, 1),
    #                                                          hand_normal=(0, 1, 0), jawwidth=.02,
    #                                                          obj=obj, rotation_ax=(1, 0, 0),
    #                                                          rotation_range=(-75, 75), toggledebug=True)
    #
    # grasp_planner.write_pregrasps('plate', pregrasp_list)
    # grasp_list = grasp_planner.load_pregrasp('plate')
    # grasp_planner.show_grasp(grasp_list[:30], obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)
    #
    # base.run()

    '''
    stick
    '''
    gm.gen_frame(length=.2).attach_to(base)
    obj = cm.gen_stick(epos=np.asarray([0, .01, 0]), thickness=.0015, sections=180, rgba=(.7, .7, .7, .7))
    obj.attach_to(base)
    pregrasp_list = []
    finger_normal = (0, 0, 1)
    hand_normal = (0, 1, 0)

    for i in np.linspace(0, 360, 12):
        print(i)
        tmp_rotmat = rm.rotmat_from_axangle((1, 0, 0), np.radians(360/12))
        hand_normal = np.dot(tmp_rotmat, hand_normal)
        finger_normal = np.dot(tmp_rotmat, finger_normal)
        pregrasp_list.extend(grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, 0),
                                                                      finger_normal=finger_normal,
                                                                      hand_normal=hand_normal, jawwidth=.01,
                                                                      obj=obj, rot_ax=(0, 1, 0),
                                                                      rot_range=(-180, 181),
                                                                      rot_interval=30,
                                                                      toggledebug=True))

    grasp_planner.write_pregrasps('stick', pregrasp_list)
    grasp_list = grasp_planner.load_pregrasp('stick')
    grasp_planner.show_grasp(grasp_list, obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)
    print(len(grasp_list))
    base.run()
