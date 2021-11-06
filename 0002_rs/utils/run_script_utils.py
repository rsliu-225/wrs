import copy
import os
import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from direct.stdpy import threading

import config
import utils.pcd_utils as pcdu
import utils.run_utils as ru
from localenv import envloader as el


class ForceRecorder(object):
    def __init__(self, motion_planner_x, f_id):
        self.motion_planner_x = motion_planner_x
        self.f_name = "./log/force_armjnts_log_" + f_id + ".pkl"
        self.__rbtx = self.motion_planner_x.rbtx
        self.__thread_recorder = None
        self.__thread_plot = None
        self.__flag = True
        self.__plotflag = False
        self.__ploton = True
        self.__armname = self.motion_planner_x.armname
        self.__force = []
        # record
        self.__info = []
        self.__force = []
        self.__armjnts = []

        # zero force
        self.__rbtx.zerotcpforce(armname="lft")
        self.__rbtx.zerotcpforce(armname="rgt")

    def start_record(self):
        self.__rbtx.zerotcpforce(armname=self.__armname)

        def recorder():
            while self.__flag:
                self.__force.append(self.__rbtx.getinhandtcpforce(armname=self.__armname))
                self.__armjnts = self.motion_planner_x.get_armjnts()
                self.__info.append([self.__force, self.__armjnts, time.time()])
                self.__plotflag = True
                time.sleep(.1)

        def plot():
            fig = plt.figure(1)
            plt.ion()
            plt.show()
            plt.ylim((-10, 10))
            while self.__ploton:
                if self.__plotflag:
                    plt.clf()
                    # print(len(self.__force))
                    force = [l[:3] for l in self.__force]
                    torque = [l[3:] for l in self.__force]
                    # distance = [np.linalg.norm(self.fixgoal - np.array(dis[1])) for dis in self.__tcp]

                    x = [0.02 * i for i in range(len(force))]
                    plt.xlim((0, max(x)))
                    plt.subplot(211)
                    plt.plot(x, force, label=["x", "y", "z"])
                    plt.legend("xyz", loc='upper left')
                    plt.subplot(212)
                    plt.plot(x, torque, label=["Rx", "Ry", "Rz"])
                    plt.pause(0.005)
                time.sleep(.1)
            plt.savefig(f"{time.time()}.png")
            pickle.dump(self.__info, open(self.f_name, "wb"))
            plt.close(fig)

        self.__thread_recorder = threading.Thread(target=recorder, name="recorder")
        self.__thread_recorder.start()
        self.__thread_plot = threading.Thread(target=plot, name="plot")
        self.__thread_plot.start()
        print("start finish")

    def finish_record(self):
        if self.__thread_recorder is None:
            return
        self.__flag = False
        self.__thread_recorder.join()
        self.__thread_recorder = None

        if self.__thread_plot is None:
            return
        self.__ploton = False
        time.sleep(.05)
        self.__plotflag = False
        self.__thread_plot.join()
        self.__thread_plot = None


def load_motion_f(folder_name, f_name, root=config.MOTIONSCRIPT_REL_PATH):
    return pickle.load(open(os.path.join(root, folder_name, f_name), "rb"))


def load_motion_sgl(motion, folder_name, id_list, root=config.MOTIONSCRIPT_REL_PATH):
    motion_dict = pickle.load(open(root + folder_name + motion + ".pkl", "rb"))
    value = []
    for id in id_list:
        try:
            value = motion_dict[id]
            motion_dict = value
        except:
            continue
    objrelpos, objrelrot, path = value
    return objrelpos, objrelrot, path


def setting_sim(stl_f_name, pos=(600, 0, 780)):
    if stl_f_name[-3:] != 'stl':
        stl_f_name += '.stl'
    objcm = el.loadObj(stl_f_name, pos=pos, rot=(0, 0, 0))
    objcm.attach_to(base)


def setting_real_simple(phoxi_f_path, amat):
    grayimg, depthnparray_float32, pcd = ru.load_phxiinfo(phoxi_f_name=phoxi_f_path, load=True)
    pcd = np.array([p for p in pcdu.trans_pcd(pcd, amat) if 790 < p[2] < 1100])
    pcdu.show_pcd(pcd)


def setting_real(phxilocator, phoxi_f_path, pen_stl_f_name, paintingobj_stl_f_name, resolution=1):
    pen_item = ru.get_obj_from_phoxiinfo_withmodel(phxilocator, pen_stl_f_name, phoxi_f_name=phoxi_f_path,
                                                   x_range=(600, 1000), y_range=(200, 600), z_range=(810, 1000))
    if paintingobj_stl_f_name is not None:
        paintingobj_item = \
            ru.get_obj_from_phoxiinfo_withmodel(phxilocator, paintingobj_stl_f_name, x_range=(400, 1080),
                                                phoxi_f_name=phoxi_f_path, resolution=resolution)
    else:
        paintingobj_item = ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                               resolution=resolution, reconstruct_surface=True)
    paintingobj_item.show_objcm(rgba=(1, 1, 1, .5))
    # paintingobj_item.show_objpcd(rgba=(1, 1, 0, 1))
    # objmat4 = paintingobj_item.objmat4
    # objmat4[:3, 3] = objmat4[:3, 3] + np.asarray([5, 5, 0])
    # paintingobj_item.set_objmat4(objmat4)
    pen_item.show_objcm(rgba=(0, 1, 0, 1))
