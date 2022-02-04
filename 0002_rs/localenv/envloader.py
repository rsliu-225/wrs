import copy
import os

import numpy as np
import open3d as o3d

import config
import modeling.collision_model as cm
import modeling.geometric_model as gm
import localenv.item as item
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import visualization.panda.world as wd
import robot_con.ur.ur3e_dual_x as ur3ex
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3edual
import basis.trimesh.sample as ts
import basis.robot_math as rm
import basis.o3dhelper as o3d_helper
from basis.trimesh.primitives import Box


# import utils.pcd_utils as pcdu
# import utils.comformalmapping_utils as cu


class Env_wrs(object):
    def __init__(self, boundingradius=.01, betransparent=False):
        """
        load obstacles model
        separated by category

        :param base:
        author: weiwei
        date: 20181205
        """

        self.__this_dir, _ = os.path.split(__file__)

        # table
        self.__tablepath = os.path.join(self.__this_dir, "../obstacles", "ur3edtable.stl")
        self.__tablecm = cm.CollisionModel(self.__tablepath, expand_radius=boundingradius, btransparency=betransparent)
        self.__tablecm.set_pos((.18, 0, 0))
        self.__tablecm.set_rgba((.32, .32, .3, 1.0))

        self.__battached = False
        self.__changableobslist = []

    def reparentTo(self, nodepath):
        if not self.__battached:
            # table
            self.__tablecm.attach_to(nodepath)
            # housing
            self.__battached = True

    def loadobj(self, name):
        self.__objpath = os.path.join(self.__this_dir, "../../0000_srl/objects", name)
        self.__objcm = cm.CollisionModel(self.__objpath, cdprimit_type="ball")
        return self.__objcm

    def getstationaryobslist(self):
        """
        generate the collision model for stationary obstacles

        :return:

        author: weiwei
        date: 20180811
        """

        stationaryobslist = [self.__tablecm]
        return stationaryobslist

    def getchangableobslist(self):
        """
        get the collision model for changable obstacles

        :return:

        author: weiwei
        date: 20190313
        """
        return self.__changableobslist

    def addchangableobs(self, base, objcm, pos, rot):
        """

        :param objcm: CollisionModel
        :param pos: nparray 1x3
        :param rot: nparray 3x3
        :return:

        author: weiwei
        date: 20190313
        """

        self.__changableobslist.append(objcm)
        objcm.attach_to(base)
        objcm.set_homomat(rm.homomat_from_posrot(pos, rot))

    def addchangableobscm(self, objcm):
        self.__changableobslist.append(objcm)

    def removechangableobs(self, objcm):
        if objcm in self.__changableobslist:
            objcm.remove()


def loadEnv_wrs(camp=[4, 0, 1.7], lookatpos=[0, 0, 1]):
    # Table width: 120
    # Table long: 1080

    base = wd.World(cam_pos=camp, lookat_pos=lookatpos)
    env = Env_wrs(boundingradius=.007)
    env.reparentTo(base)

    # phoxi
    phoxicam = cm.CollisionModel(initor=Box(box_extents=[.55, .2, .1]))
    phoxicam.set_rgba((.32, .32, .3, 1))
    env.addchangableobs(base, phoxicam, [.65, 0, 1.76], np.eye(3))

    # desk
    desk = cm.CollisionModel(initor=Box(box_extents=[1.08, .4, .76]))
    desk.set_rgba((0.7, 0.7, 0.7, 1))
    env.addchangableobs(base, desk, [.54, .8, .38], np.eye(3))

    # penframe
    # penframe = cm.CollisionModel(initor=Box(box_extents=[.2, .32, .1]))
    # penframe.set_rgba((0.7, 0.7, 0.7, 1))
    # env.addchangableobs(base.render, penframe, [1.08 - .3 + .1, .6 - .175, .795], np.eye(3))

    return base, env


def __pcd_trans(pcd, amat):
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(amat, homopcd).T
    return realpcd[:, :3]


def loadUr3e(showrbt=False):
    rbt = ur3edual.UR3EDual()
    if showrbt:
        rbt.gen_meshmodel().attach_to(base)

    return rbt


def loadUr3ex(pc_ip='10.2.0.100'):
    rbtx = ur3ex.Ur3EDualUrx(lft_robot_ip='10.0.2.2', rgt_robot_ip='10.0.2.3', pc_ip=pc_ip)

    return rbtx


def loadObj(f_name, pos=(0, 0, 0), rot=(0, 0, 0), color=(1, 1, 1), transparency=0.5):
    obj = cm.CollisionModel(initor=os.path.join(config.ROOT, "obstacles", f_name))
    obj.set_pos(pos)
    obj.set_rgba((color[0], color[1], color[2], transparency))
    obj.set_rpy(rot[0], rot[1], rot[2])

    return obj


def loadObjpcd(f_name, pos=(0, 0, 0), rot=(0, 0, 0), sample_num=100000, toggledebug=False):
    obj = cm.CollisionModel(initor=os.path.join(config.ROOT, "obstacles", f_name))
    rotmat4 = np.zeros([4, 4])
    rotmat4[:3, :3] = rm.rotmat_from_euler(rot[0], rot[1], rot[2], axes="sxyz")
    rotmat4[:3, 3] = pos

    obj_surface = np.asarray(ts.sample_surface(obj.trimesh, count=sample_num))
    # obj_surface = obj_surface[obj_surface[:, 2] > 2]
    obj_surface_real = __pcd_trans(obj_surface, rotmat4)

    if toggledebug:
        obj_surface = o3d_helper.nparray2o3dpcd(copy.deepcopy(obj_surface))
        obj_surface.paint_uniform_color([1, 0.706, 0])
        o3d.visualization.draw_geometries([obj_surface], window_name='loadObjpcd')
        pcddnp = gm.gen_pointcloud(obj_surface_real)
        pcddnp.reparentTo(base.render)

    return obj_surface_real


def update(rbtmnp, motioncounter, robot, path, armname, task):
    if motioncounter[0] < len(path):
        if rbtmnp[0] is not None:
            rbtmnp[0].detachNode()
        pose = path[motioncounter[0]]
        robot.movearmfk(pose, armname)
        rbtmnp[0] = robot.gen_meshmodel()
        rbtmnp[0].reparentTo(base.render)
        motioncounter[0] += 1
    else:
        motioncounter[0] = 0
    return task.again


def loadObjitem(f_name, pos=(0, 0, 0), rot=(0, 0, 0), sample_num=10000, type="box", filter_dir=None):
    if f_name[-3:] != 'stl':
        f_name += '.stl'
    objcm = cm.CollisionModel(initor=os.path.join(config.ROOT, "obstacles", f_name), cdprimit_type=type)
    objcm.objtrm.remove_unreferenced_vertices()
    objcm.objtrm.remove_degenerate_faces()
    print('num of vs:', len(objcm.objtrm.vertices))
    # if len(vs) > 20000:
    #     print('---------------down sample---------------')
    #     vs, faces, nrmls = cu.downsample(vs, faces, 20000/len(vs))
    #     print('num of vs:', len(vs))
    #     objcm = pcdu.reconstruct_surface(vs, radii=[5])

    objmat4 = np.zeros([4, 4])
    objmat4[:3, :3] = rm.rotmat_from_euler(rot[0], rot[1], rot[2], axes="sxyz")
    objmat4[:3, 3] = pos
    objcm.set_homomat(objmat4)
    print("---------------success load---------------")

    return item.Item(objcm=objcm, objmat4=objmat4, sample_num=sample_num, filter_dir=filter_dir)


if __name__ == '__main__':
    base, env = loadEnv_wrs()
    gm.gen_frame().attach_to(base)
    objcm = loadObj('cylinder.stl', pos=(.7, 0, .78), rot=(0, 0, 0), transparency=1)
    objcm.attach_to(base)
    rbt = loadUr3e()
    rbt.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
    print(rbt.manipulability(component_name="lft_arm"))
    mnpax = rbt.manipulability_axmat(component_name="lft_arm")
    tcp_pos, _ = rbt.get_gl_tcp(manipulator_name="lft_arm")
    print(mnpax)
    print(tcp_pos)
    gm.gen_ellipsoid(pos=tcp_pos, axmat=mnpax).attach_to(base)
    gm.gen_arrow(spos=tcp_pos, epos=tcp_pos + mnpax[:, 0], rgba=(1, 0, 0, 1)).attach_to(base)
    gm.gen_arrow(spos=tcp_pos, epos=tcp_pos + mnpax[:, 1], rgba=(0, 1, 0, 1)).attach_to(base)
    gm.gen_arrow(spos=tcp_pos, epos=tcp_pos + mnpax[:, 2], rgba=(0, 0, 1, 1)).attach_to(base)

    base.run()
