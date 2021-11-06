import copy
import os
import pickle
import math
import numpy as np

import config
import modeling.collision_model as cm
import modeling.geometric_model as gm
import basis.robot_math as rm
import grasping.annotation.utils as gau


class GraspPlanner(object):
    def __init__(self, gripper_s):
        self.gripper_s = gripper_s

    def load_objcm(self, stl_f_name):
        obj = cm.CollisionModel(initor=os.path.join(config.ROOT, "obstacles/", stl_f_name))
        obj.set_rgba((1, 1, 0, 1))
        return obj

    def write_pregrasps(self, stl_f_name, pregrasps, mode=""):
        if mode == "":
            pickle.dump(pregrasps,
                        open(config.ROOT + f"/graspplanner/pregrasp/{stl_f_name.split('.stl')[0]}_pregrasps.pkl", 'wb'))
        elif mode == "hndovr":
            pickle.dump(pregrasps,
                        open(config.ROOT + f"/graspplanner/pregrasp_hndovr/{stl_f_name.split('.stl')[0]}_pregrasps.pkl",
                             'wb'))

    def load_pregrasp(self, stl_f_name, mode=""):
        if mode == "":
            return pickle.load(
                open(config.ROOT + "/graspplanner/pregrasp/" + stl_f_name.split(".stl")[0] + '_pregrasps.pkl', 'rb'))
        elif mode == "hndovr":
            return pickle.load(
                open(config.ROOT + "/graspplanner/pregrasp_hndovr/" + stl_f_name.split(".stl")[0] + '_pregrasps.pkl',
                     'rb'))

    def define_grasp(self, grasp_coordinate, finger_normal, hand_normal, jawwidth, obj, toggledebug=False):
        effect_grasp = gau.define_grasp(self.gripper_s, obj,
                                        gl_jaw_center_pos=grasp_coordinate,
                                        gl_jaw_center_z=hand_normal,
                                        gl_jaw_center_y=finger_normal,
                                        jaw_width=jawwidth,
                                        toggle_flip=True, toggle_debug=toggledebug)

        return effect_grasp

    def define_grasp_with_rotation(self, grasp_coordinate, finger_normal, hand_normal, jawwidth, obj,
                                   rotation_ax=np.array([0, 1, 0]), rotation_interval=15, rotation_range=(-90, 90),
                                   toggledebug=False):
        if toggledebug:
            gm.gen_frame().attach_to(base)
        effect_grasp = \
            gau.define_grasp_with_rotation(self.gripper_s, obj,
                                           gl_jaw_center_pos=np.asarray(grasp_coordinate),
                                           gl_jaw_center_z=np.asarray(hand_normal),
                                           gl_jaw_center_y=np.asarray(finger_normal),
                                           jaw_width=jawwidth,
                                           gl_rotation_ax=rotation_ax,
                                           rotation_interval=math.radians(rotation_interval),
                                           rotation_range=(
                                               math.radians(rotation_range[0]), math.radians(rotation_range[1])),
                                           toggle_flip=True, toggle_debug=toggledebug)

        return effect_grasp

    def show_grasp(self, grasp_list, obj, rgba=None, toggle_tcpcs=False, toggle_jntscs=False, mode='mesh'):
        obj.attach_to(base)
        for grasp in grasp_list:
            self.gripper_s.jaw_to(grasp[0])
            self.gripper_s.fix_to(grasp[3], grasp[4])
            if mode == 'mesh':
                self.gripper_s.gen_meshmodel(rgba=rgba, toggle_tcpcs=toggle_tcpcs,
                                             toggle_jntscs=toggle_jntscs).attach_to(base)
            else:
                self.gripper_s.gen_stickmodel(toggle_tcpcs=toggle_tcpcs, toggle_jntscs=toggle_jntscs).attach_to(base)
