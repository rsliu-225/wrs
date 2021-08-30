import cv2
import time
import pickle
from drivers.devices.kinect2.pykinect2 import PyKinectV2, PyKinectRuntime
import drivers.rpc.frtknt.kntv2 as kntv2
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
