import drivers.rpc.frtknt.frtknt_client as fc
import visualization.panda.world as pc
import modeling.geometric_model as gm

import cv2
import time
import pickle
from drivers.devices.kinect2.pykinect2 import PyKinectV2, PyKinectRuntime
import drivers.rpc.frtknt.kntv2 as kntv2
import numpy as np
import knt_utils as ku

if __name__ == '__main__':
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
    clframe_list, dframe_list = ku.load(f_name)
    for i in range(len(clframe_list)):
        aligned_rgba = ku.map_color2depth(kinect, dframe_list[i], clframe_list[i])
        # cv2.imshow('rgbimg', ku.clframe2rgbimg(clframe_list[i]))
        cv2.imshow('depthimg', ku.dframe2depthimg(dframe_list[i]))
        cv2.imshow('aligned depth', aligned_rgba)
        print(aligned_rgba.shape)
        print(aligned_rgba)
        cv2.waitKey(0)
