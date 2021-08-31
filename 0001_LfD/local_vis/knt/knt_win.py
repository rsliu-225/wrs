import drivers.rpc.frtknt.frtknt_client as fc
import visualization.panda.world as pc
import modeling.geometric_model as gm

import cv2
import time
import pickle
import drivers.rpc.frtknt.kntv2 as kntv2
from drivers.devices.kinect2.pykinect2 import PyKinectV2
import numpy as np
import threading


def init_knt(knt):
    class ThreadKinectCam(threading.Thread):
        def __init__(self, index, create_time, kinect):
            threading.Thread.__init__(self)
            self.index = index
            self.create_time = create_time
            self.kinect = kinect

        def close(self):
            self.kinect.close()

        def run(self):
            self.kinect.runForThread()

    threadKinectCam = ThreadKinectCam(2, time.time(), knt)
    threadKinectCam.start()
    while True:
        if knt.getColorFrame() is None:
            print("initializing color...")
            continue
        if knt.getDepthFrame() is None:
            print("initializing depth...")
            continue
        break
    return True


def load(f_name):
    return pickle.load(open(f'./res/knt/{f_name}', 'rb'))


def map_depth2color(knt, dframe):
    return knt.mapDepthFrameToColorFrame(dframe)


def map_color2depth(knt, dframe, clframe):
    return knt.mapColorFrameToDepthFrame(dframe, clframe)


def clframe2rgbimg(clframe):
    clb = np.flip(np.array(clframe[0::4]).reshape((1080, 1920)), 1)
    clg = np.flip(np.array(clframe[1::4]).reshape((1080, 1920)), 1)
    clr = np.flip(np.array(clframe[2::4]).reshape((1080, 1920)), 1)
    return cv2.merge((clb, clg, clr))


def dframe2depthimg(dframe):
    return dframe.reshape((424, 512, 1))


def record_img(f_name, host="10.0.1.143:183001"):
    depthimg_list = []
    rgbimg_list = []
    pcd_list = []

    knt = fc.FrtKnt(host)
    while True:
        print(len(depthimg_list))
        depthimg = knt.getdepthimg()
        rgbimg = knt.getrgbimg()
        pcd = knt.getpcd()
        pcd = pcd / 1000
        cv2.imshow('rgbimg', rgbimg)
        cv2.imshow('depthimg', depthimg)
        depthimg_list.append(depthimg)
        rgbimg_list.append(rgbimg)
        pcd_list.append(pcd)
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            break
    pickle.dump([rgbimg_list, depthimg_list, pcd_list], open(f'res/knt/{f_name}.pkl', 'wb'))


def record_frame(f_name):
    dframe_list = []
    clframe_list = []
    knt = kntv2.KinectV2(PyKinectV2.FrameSourceTypes_Color |
                         PyKinectV2.FrameSourceTypes_Depth)
    if ku.init_knt(knt):
        while True:
            print(len(dframe_list))
            dframe = knt.getDepthFrame()
            clframe = knt.getColorFrame()
            cv2.imshow('rgbimg', clframe2rgbimg(clframe))
            cv2.imshow('depthimg', dframe2depthimg(dframe))
            dframe_list.append(dframe)
            clframe_list.append(clframe)
            if cv2.waitKey(1) & 0xff == 27:
                break
    pickle.dump([clframe_list, dframe_list], open(f'res/knt/{f_name}_raw.pkl', 'wb'))


if __name__ == '__main__':
    f_name = 'tst'
    record_frame(f_name)

    kinect = kntv2.KinectV2(PyKinectV2.FrameSourceTypes_Color |
                            PyKinectV2.FrameSourceTypes_Depth)
    # f_name = 'tst.pkl'
    # rgbimg_list, depthimg_list, pcd_list = load(f_name)
    # for i, depthimg in enumerate(depthimg_list):
    #     aligned_depthimg = map_color2depth(depthimg, rgbimg_list[i])
    #     cv2.imshow('depth', depthimg)
    #     cv2.imshow('rgb', rgbimg_list[i])
    #     cv2.imshow('aligned depth', aligned_depthimg)
    #     print(depthimg.shape)
    #     print(rgbimg_list[i].shape)
    #     cv2.waitKey(0)

    f_name = 'tst_raw.pkl'
    clframe_list, dframe_list = load(f_name)
    for i in range(len(clframe_list)):
        aligned_rgba = map_color2depth(kinect, dframe_list[i], clframe_list[i])
        # cv2.imshow('rgbimg', ku.clframe2rgbimg(clframe_list[i]))
        cv2.imshow('depthimg', dframe2depthimg(dframe_list[i]))
        cv2.imshow('aligned depth', aligned_rgba)
        print(aligned_rgba.shape)
        print(aligned_rgba)
        cv2.waitKey(0)

