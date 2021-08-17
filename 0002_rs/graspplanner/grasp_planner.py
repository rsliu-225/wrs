import copy
import os
import pickle

import numpy as np

import config
import environment.bulletcdhelper as bch
import environment.collisionmodel as cm
import basis.robot_math as rm


class GraspPlanner(object):
    def __init__(self, hndfa):
        self.hndfa = hndfa
        self.bcdchecker = bch.MCMchecker(toggledebug=False)

    def load_objcm(self, stl_f_name):
        obj = cm.CollisionModel(objinit=os.path.join(config.ROOT, "obstacles/", stl_f_name))
        obj.setColor(1, 1, 0, 1)
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

    def checkhndenvcollision(self, hnd, jawwidth, homomat, inputobj):
        handtmp = hnd
        handtmp.sethomomat(homomat)
        handtmp.setjawwidth(jawwidth + 10)
        iscollided = self.bcdchecker.isMeshListMeshListCollided(handtmp.cmlist, [copy.deepcopy(inputobj)])
        return iscollided

    def define_grasp(self, grasp_coordinate, finger_normal, hand_normal, jawwidth, obj, toggledebug=False):
        effect_grasp = []
        hnd = self.hndfa.genHand(ftsensoroffset=36)
        base.pggen.plotAxis(hnd.handnp, length=15, thickness=2)
        grasp = hnd.approachAt(grasp_coordinate[0], grasp_coordinate[1], grasp_coordinate[2],
                               finger_normal[0], finger_normal[1], finger_normal[2],
                               hand_normal[0], hand_normal[1], hand_normal[2], jawwidth=jawwidth)
        if not self.checkhndenvcollision(self.hndfa.genHand(ftsensoroffset=36), grasp[0], grasp[2], obj):
            if toggledebug:
                hnd.setColor(1, 0, 0, .5)
                hnd.reparentTo(base.render)
                obj.reparentTo(base.render)
            effect_grasp.append(grasp)

        hnd_reverse = self.hndfa.genHand(ftsensoroffset=36)
        base.pggen.plotAxis(hnd_reverse.handnp, length=15, thickness=2)
        grasp_reverse = hnd_reverse.approachAt(grasp_coordinate[0], grasp_coordinate[1], grasp_coordinate[2],
                                               -finger_normal[0], -finger_normal[1], -finger_normal[2],
                                               hand_normal[0], hand_normal[1], hand_normal[2], jawwidth=jawwidth)
        if not self.checkhndenvcollision(self.hndfa.genHand(ftsensoroffset=36), grasp[0], grasp[2], obj):
            if toggledebug:
                hnd_reverse.setColor(1, 0, 0, .5)
                hnd_reverse.reparentTo(base.render)
                obj.reparentTo(base.render)
            effect_grasp.append(grasp_reverse)

        return effect_grasp

    def define_grasp_with_rotation(self, grasp_coordinate, finger_normal, hand_normal, jawwidth, obj,
                                   rotation_interval=15, rotation_range=(-90, 90), toggledebug=False):
        effect_grasp = []
        rotation_list = []
        for i in range(rotation_range[0], rotation_range[1] + 1):
            if i % rotation_interval == 0:
                rotation_list.append(i)
        for rotate_angle in rotation_list:
            hnd = self.hndfa.genHand(ftsensoroffset=36)
            # base.pggen.plotAxis(hnd.handnp, length=15, thickness=2)
            rotate = np.dot(np.array([hand_normal[0], hand_normal[1], hand_normal[2]]),
                            rm.rodrigues(finger_normal, rotate_angle))
            grasp = hnd.approachAt(grasp_coordinate[0], grasp_coordinate[1], grasp_coordinate[2], finger_normal[0],
                                   finger_normal[1], finger_normal[2], rotate[0], rotate[1], rotate[2],
                                   jawwidth=jawwidth)
            if not self.checkhndenvcollision(self.hndfa.genHand(ftsensoroffset=36), grasp[0], grasp[2], obj):
                if toggledebug:
                    hnd.setColor(1, 0, 0, .2)
                    hnd.reparentTo(base.render)
                    obj.reparentTo(base.render)
                effect_grasp.append(grasp)

            hnd_reverse = self.hndfa.genHand(ftsensoroffset=36)
            # base.pggen.plotAxis(hnd_reverse.handnp, length=15, thickness=2)
            grasp_reverse = hnd_reverse.approachAt(grasp_coordinate[0], grasp_coordinate[1], grasp_coordinate[2],
                                                   -finger_normal[0], -finger_normal[1], -finger_normal[2], rotate[0],
                                                   rotate[1], rotate[2], jawwidth=jawwidth)
            if not self.checkhndenvcollision(self.hndfa.genHand(ftsensoroffset=36), grasp_reverse[0],
                                             grasp_reverse[2], obj):
                if toggledebug:
                    hnd_reverse.setColor(1, 0, 0, .2)
                    hnd_reverse.reparentTo(base.render)
                    obj.reparentTo(base.render)
                effect_grasp.append(grasp_reverse)

        return effect_grasp

    def show_grasp(self,grasp_list, obj, rgba=(1, 0, 0, .2)):
        obj.reparentTo(base.render)
        for grasp in grasp_list:
            hnd = self.hndfa.genHand(ftsensoroffset=36)
            hnd.sethomomat(grasp[2])
            hnd.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
            hnd.reparentTo(base.render)

