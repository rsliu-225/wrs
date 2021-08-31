import copy

import numpy as np

import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import motionplanner.robot_helper as rbt_helper
import basis.robot_math as rm


class AnimationHelper(object):
    def __init__(self, env, rbt, armname="lft"):
        self.rbt = rbt
        self.env = env
        self.obscmlist = env.getstationaryobslist() + env.getchangableobslist()
        self.armname = armname
        self.rbth = rbt_helper.RobotHelper(self.env, self.rbt, self.armname)
        self.hndfa = rtqhe.RobotiqHE()

    def draw_axis(self, pos, rot, rgba=None, length=50, thickness=5):
        base.pggen.plotAxis(base.render, spos=pos, srot=rot, length=length, rgba=rgba, thickness=thickness)
        base.pggen.plotSphere(base.render, pos, 10, rgba=(1, 1, 0, 1))

    def draw_axis_uneven(self, pos, rot, scale=1, thickness=5):
        base.pggen.plotArrow(base.render, spos=pos, epos=pos + scale * rot[:3, 0], rgba=(1, 0, 0, 1),
                             thickness=thickness)
        base.pggen.plotArrow(base.render, spos=pos, epos=pos + scale * rot[:3, 1], rgba=(0, 1, 0, 1),
                             thickness=thickness)
        base.pggen.plotArrow(base.render, spos=pos, epos=pos + scale * rot[:3, 2], rgba=(0, 0, 1, 1),
                             thickness=thickness)

    def show_path_hold(self, path, objcm, objrelpos, objrelrot, fromrgba=(1, 0, 0, .5), torgba=(1, 1, 0, .5),
                       toggleendcoord=False, genmnp=True, jawwidth=50, toggleobjcoord=False):
        objcm = copy.deepcopy(objcm)
        rgbdiff = np.asarray(torgba) - np.asarray(fromrgba)
        pos_pre = None
        for i, armjnts in enumerate(path):
            if i % 3 != 0:
                continue
            rgba = (fromrgba[0] + rgbdiff[0] * i / len(path),
                    fromrgba[1] + rgbdiff[1] * i / len(path),
                    fromrgba[2] + rgbdiff[2] * i / len(path),
                    1,)
            self.rbt.movearmfk(armjnts, self.armname)
            objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
            objmat4 = rm.homomat_from_posrot(objpos, objrot)
            self.show_objmat4(objcm, objmat4, rgba=(rgba[0], rgba[1], rgba[2], rgba[3]))
            if pos_pre is not None:
                base.pggen.plotStick(base.render, spos=pos_pre, epos=objmat4[:3, 3], rgba=(0,0,0,1), thickness=5)
            pos_pre = objmat4[:3, 3]

            if toggleobjcoord:
                objcm.showlocalframe()
            # if genmnp:
            #     self.__genmnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord, jawwidth=jawwidth)
            # else:
            #     self.__gensnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord)

    def show_path_end(self, path, fromrgba=(1, 0, 0, 1), torgba=(1, 1, 0, 1), toggleendcoord=False, jawwidth=20):
        rgbdiff = np.asarray(torgba) - np.asarray(fromrgba)
        pos_pre = None

        for i in range(len(path)):
            if i % 1 != 0:
                continue
            rgba = (fromrgba[0] + rgbdiff[0] * i / len(path),
                    fromrgba[1] + rgbdiff[1] * i / len(path),
                    fromrgba[2] + rgbdiff[2] * i / len(path),
                    fromrgba[3] + rgbdiff[3] * i / len(path))
            self.rbt.movearmfk(path[i], self.armname)
            # if i % 5 == 0:
            # self.__genmnp_by_armname(rgba=(rgba[0], rgba[1], rgba[2], .5), toggleendcoord=False, jawwidth=jawwidth)
            # self.__genmnp_by_armname(rgba=(1, 1, 1, .2), toggleendcoord=False, jawwidth=jawwidth)
            eepos, eerot = self.rbt.getee(armname=self.armname)
            if toggleendcoord:
                base.pggen.plotAxis(base.render, spos=eepos, srot=eerot, length=30, thickness=5)
            # if pos_pre is not None:
            #     base.pggen.plotStick(base.render, spos=pos_pre, epos=eepos, rgba=rgba, thickness=2)
            # base.pggen.plotSphere(base.render, eepos, rgba=rgba, radius=10)
            # pos_pre = eepos

    def show_rbt(self):
        self.rbtmg.genmnp(self.rbt).reparentTo(base.render)

    def show_objmat4(self, obj, objmat4, rgba=(1, 1, 1, .5), showlocalframe=False):
        obj_final = copy.deepcopy(obj)
        obj_final.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        obj_final.setMat(base.pg.np4ToMat4(objmat4))
        if showlocalframe:
            obj_final.showlocalframe()
        obj_final.reparentTo(base.render)

    def show_objmat4_list(self, objmat4_list, objcm=None, fromrgba=(1, 1, 1, .5), torgba=(1, 1, 1, .5),
                          showlocalframe=False):
        rgbdiff = np.asarray(torgba) - np.asarray(fromrgba)
        if objmat4_list is None:
            print("show_objmat4_list, objmat4_list is None!")
        else:
            for i, objmat4 in enumerate(objmat4_list):
                if objcm is not None:
                    obj_show = copy.deepcopy(objcm)
                    obj_show.setColor(fromrgba[0] + rgbdiff[0] * i / len(objmat4_list),
                                      fromrgba[1] + rgbdiff[1] * i / len(objmat4_list),
                                      fromrgba[2] + rgbdiff[2] * i / len(objmat4_list),
                                      fromrgba[3] + rgbdiff[3] * i / len(objmat4_list))
                    obj_show.setMat(base.pg.np4ToMat4(objmat4))
                    obj_show.reparentTo(base.render)
                    if showlocalframe:
                        obj_show.showlocalframe()
                else:
                    base.pggen.plotArrow(base.render, spos=objmat4[:3, 3], epos=objmat4[:3, 3] + 10 * objmat4[:3, 0],
                                         rgba=fromrgba)

    def show_objmat4_list_pos(self, objmat4_list, rgba=(1, 1, 1, .5), showlocalframe=False):
        if objmat4_list is None:
            print("show_objmat4_list_pos, objmat4_list is None!")
        else:
            for objmat4 in objmat4_list:
                base.pggen.plotSphere(base.render, pos=objmat4[:3, 3], rgba=rgba)

    def show_path(self, path, fromrgba=(1, 0, 0, .5), torgba=(1, 1, 0, .5), toggleendcoord=False, genmnp=True):
        rgbdiff = np.asarray(torgba) - np.asarray(fromrgba)
        for i, armjnts in enumerate(path):
            if i % 3 != 0:
                continue
            self.rbt.movearmfk(armjnts, self.armname)
            if genmnp:
                self.__genmnp_by_armname(rgba=(fromrgba[0] + rgbdiff[0] * i / len(path),
                                               fromrgba[1] + rgbdiff[1] * i / len(path),
                                               fromrgba[2] + rgbdiff[2] * i / len(path),
                                               fromrgba[3] + rgbdiff[3] * i / len(path)),
                                         toggleendcoord=toggleendcoord)
            else:
                self.__gensnp_by_armname(rgba=(fromrgba[0] + rgbdiff[0] * i / len(path),
                                               fromrgba[1] + rgbdiff[1] * i / len(path),
                                               fromrgba[2] + rgbdiff[2] * i / len(path),
                                               fromrgba[3] + rgbdiff[3] * i / len(path)),
                                         toggleendcoord=toggleendcoord)

    def show_armjnts_with_obj(self, armjnts, obj, objrelpos, objrelrot, toggleendcoord=False, rgba=(1, 1, 1, .5)):
        self.rbt.movearmfk(armjnts, self.armname)
        self.__genmnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord)
        objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
        objmat4 = rm.homomat_from_posrot(objpos, objrot)
        base.pggen.plotSphere(base.render, pos=objmat4[:3, 3], rgba=(1, 1, 0, 1))
        obj.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        obj.setMat(base.pg.np4ToMat4(objmat4))
        obj.reparentTo(base.render)
        obj.showlocalframe()

    def show_armjnts(self, rgba=None, armjnts=None, toggleendcoord=False, jawwidth=50, genmnp=True):
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, self.armname)
        if genmnp:
            self.__genmnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord, jawwidth=jawwidth)
        else:
            self.__gensnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord)

    def show_hnd(self, rgba=None, armjnts=None, toggleendcoord=False, jawwidth=50):
        eepos, eerot = self.rbth.get_tcp(armjnts=armjnts)
        hnd = self.hndfa.gen_meshmodel(jawwidth=jawwidth, ftsensoroffset=36, toggle_tcpcs=toggleendcoord)
        if rgba is not None:
            hnd.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        hnd.sethomomat(rm.homomat_from_posrot(eepos, eerot))
        hnd.reparentTo(base.render)

    def show_hnd_sgl(self, hndmat4, rgba=None, toggleendcoord=False, jawwidth=50):
        hnd = self.hndfa.gen_meshmodel(jawwidth=jawwidth, ftsensoroffset=36, toggle_tcpcs=toggleendcoord)
        if rgba is not None:
            hnd.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        hnd.sethomomat(hndmat4)
        hnd.reparentTo(base.render)

    def show_animation(self, path):
        rbtmnp = [None, None]
        motioncounter = [0]
        taskMgr.doMethodLater(0.05, self.__update, "update",
                              extraArgs=[rbtmnp, motioncounter, self.rbt, path, self.armname],
                              appendTask=True)

    def show_animation_hold(self, path, obj, objrelpos, objrelrot, jawwidth=50):
        rbtmnp = [None, None]
        motioncounter = [0]
        taskMgr.doMethodLater(0.05, self.__update_hold, "update",
                              extraArgs=[rbtmnp, motioncounter, self.rbt, path, self.armname, obj,
                                         objrelpos, objrelrot, jawwidth],
                              appendTask=True)

    def show_animation_hold_dual(self, path, obj_lft, objrelpos_lft, objrelrot_lft,
                                 obj_rgt, objrelpos_rgt, objrelrot_rgt):
        rbtmnp = [None, None]
        motioncounter = [0]
        taskMgr.doMethodLater(0.05, self.__update_hold_dual, "update",
                              extraArgs=[rbtmnp, motioncounter, self.rbt, path, self.rbtmg, self.rbtball,
                                         obj_lft, objrelpos_lft, objrelrot_lft, obj_rgt, objrelpos_rgt, objrelrot_rgt],
                              appendTask=True)

    def __genmnp_by_armname(self, rgba, toggleendcoord=False, jawwidth=50):
        self.rbt.closegripper(armname=self.armname, jawwidth=jawwidth)
        if self.armname == "lft":
            self.rbtmg.genmnp(self.rbt, togglejntscoord=False, toggleendcoord=toggleendcoord, rgbalft=rgba). \
                reparentTo(base.render)
        else:
            self.rbtmg.genmnp(self.rbt, togglejntscoord=False, toggleendcoord=toggleendcoord, rgbargt=rgba). \
                reparentTo(base.render)

    def __gensnp_by_armname(self, rgba, toggleendcoord=False):
        self.rbt.closegripper(armname=self.armname)
        if self.armname == "lft":
            self.rbtmg.gensnp(self.rbt, togglejntscoord=False, toggleendcoord=toggleendcoord, lftrgba=rgba). \
                reparentTo(base.render)
        else:
            self.rbtmg.gensnp(self.rbt, togglejntscoord=False, toggleendcoord=toggleendcoord, rgtrgba=rgba). \
                reparentTo(base.render)

    def __update(self, rbtmnp, motioncounter, rbt, path, armname, task):
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
            armjnts = path[motioncounter[0]]
            rbt.movearmfk(armjnts, armname=armname)
            rbtmnp[0] = rbt.gen_meshmodel()
            rbtmnp[0].attach_to(base)

            # rbtball.showfullcn(rbt)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again

    def __update_hold(self, rbtmnp, motioncounter, rbt, path, armname,
                      obj, objrelpos, objrelrot, jawwidth, task):
        self.rbt.closegripper(armname=self.armname, jawwidth=jawwidth)
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
            armjnts = path[motioncounter[0]]
            rbt.movearmfk(armjnts, armname=armname)
            rbtmnp[0] = rbt.gen_meshmodel()
            rbtmnp[0].attach_to(base)

            objpos, objrot = self.rbt.getworldpose(objrelpos, objrelrot, self.armname)
            objmat4 = rm.homomat_from_posrot(objpos, objrot)
            obj.setColor(1, 0, 0, 1)
            obj.setMat(base.pg.np4ToMat4(objmat4))
            obj.reparentTo(base.render)
            obj.showlocalframe()
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again

    def __update_hold_dual(self, rbtmnp, motioncounter, rbt, path, rbtmesh, rbtball,
                           obj_lft, objrelpos_lft, objrelrot_lft, obj_rgt, objrelpos_rgt, objrelrot_rgt, task):

        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()

            rbt.movearmfk(path[motioncounter[0]][0], armname="lft")
            rbt.movearmfk(path[motioncounter[0]][1], armname="rgt")

            rbtmnp[0] = rbtmesh.genmnp(rbt)
            rbtmnp[0].reparentTo(base.render)
            rbtball.showfullcn(rbt)

            objpos_lft, objrot_lft = self.rbt.getworldpose(objrelpos_lft, objrelrot_lft, "lft")
            objmat4 = rm.homomat_from_posrot(objpos_lft, objrot_lft)
            obj_lft.setColor(1, 0, 0, 1)
            obj_lft.setMat(base.pg.np4ToMat4(objmat4))
            obj_lft.reparentTo(base.render)
            # obj_lft.showlocalframe()

            objpos_rgt, objrot_rgt = self.rbt.getworldpose(objrelpos_rgt, objrelrot_rgt, "rgt")
            objmat4 = rm.homomat_from_posrot(objpos_rgt, objrot_rgt)
            obj_rgt.setColor(1, 0, 0, 1)
            obj_rgt.setMat(base.pg.np4ToMat4(objmat4))
            obj_rgt.reparentTo(base.render)
            # obj_rgt.showlocalframe()

            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again


if __name__ == '__main__':
    '''
    set up env and param
    '''
    from localenv import envloader as el

    base, env = el.loadEnv_wrs()

    rbt, rbtmg, rbtball = el.loadUr3e(showrbt=False)
    ah = AnimationHelper(env, rbt, rbtmg, rbtball, "lft")
