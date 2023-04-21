import numpy as np

import config
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yumigripper
import graspplanner.grasp_planner as gp
import visualization.panda.world as wd
import modeling.collision_model as cm
import basis.robot_math as rm
import modeling.geometric_model as gm

if __name__ == '__main__':
    base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
    # gripper = rtqhe.RobotiqHE()
    gripper = yumigripper.YumiGripper()
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
    # obj = cm.gen_box(extent=np.asarray([.0015, .1, .01]), homomat=rm.homomat_from_posrot([0, .05, 0], np.eye(3)))
    # obj = cm.gen_box(extent=np.asarray([.0015, .01, .01]), homomat=rm.homomat_from_posrot([0, .01, 0], np.eye(3)))
    # obj.attach_to(base)
    #
    # pregrasp_list = grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, 0), finger_normal=(0, 0, 1),
    #                                                          hand_normal=(0, -1, 0), jawwidth=.005, rot_interval=10,
    #                                                          obj=obj, rot_ax=(1, 0, 0),
    #                                                          rot_range=(-75, 75), toggledebug=True)
    #
    # grasp_planner.write_pregrasps('plate', pregrasp_list)
    # grasp_list = grasp_planner.load_pregrasp('plate')
    # grasp_planner.show_grasp(grasp_list, obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)

    # base.run()

    '''
    plate
    '''
    obj = cm.gen_box(extent=np.asarray([.0015, .01, .01]), homomat=rm.homomat_from_posrot([0, .01, 0], np.eye(3)))
    obj.attach_to(base)

    pregrasp_list = grasp_planner.define_grasp_with_rotation(grasp_coordinate=(0, 0, 0), finger_normal=(0, 0, 1),
                                                             hand_normal=(0, 1, 0), jawwidth=.005, rot_interval=10,
                                                             obj=obj, rot_ax=(1, 0, 0),
                                                             rot_range=(-75, 75), toggledebug=True)

    grasp_planner.write_pregrasps('plate_yumi', pregrasp_list)
    grasp_list = grasp_planner.load_pregrasp('plate_yumi')
    grasp_planner.show_grasp(grasp_list, obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)

    base.run()

    '''
    stick
    '''
    # import grasping.annotation.utils as gau
    #
    # gm.gen_frame(length=.2).attach_to(base)
    # obj = cm.gen_stick(epos=np.asarray([0, .001, 0]), thickness=.01, sections=180, rgba=(.7, .7, .7, .7))
    # obj.attach_to(base)
    # pregrasp_list = []
    #
    # for i in range(0, 181, 15):
    #     angle = np.radians(i)
    #     pregrasp_list.extend(gau.define_grasp_with_rotation(gripper, obj, gl_jaw_center_pos=(0, 0, 0),
    #                                                         gl_jaw_center_z=
    #                                                         np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]), angle),
    #                                                                rm.unit_vector(np.array([0, 1, 0]))),
    #                                                         gl_jaw_center_y=
    #                                                         np.dot(rm.rotmat_from_axangle(np.array([1, 0, 0]), angle),
    #                                                                np.array([0, 0, 1])),
    #                                                         jaw_width=.03,
    #                                                         rotation_interval=np.radians(15),
    #                                                         gl_rotation_ax=np.array([0, 1, 0]),
    #                                                         toggle_debug=False))
    #
    # grasp_planner.write_pregrasps('stick_yumi', pregrasp_list)
    # pregrasp_list = grasp_planner.load_pregrasp('stick_yumi')
    # grasp_planner.show_grasp(pregrasp_list[:1], obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False)
    # print(len(pregrasp_list))
    # base.run()
