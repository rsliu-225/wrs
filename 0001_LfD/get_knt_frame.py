import drivers.rpc.frtknt.frtknt_client as fc
import visualization.panda.world as pc
import modeling.geometric_model as gm

import cv2
import time
import pickle
import drivers.rpc.frtknt.kntv2 as kntv2
from drivers.devices.kinect2.pykinect2 import PyKinectV2
import numpy as np
import knt_utils as ku


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
            cv2.imshow('rgbimg', ku.clframe2rgbimg(clframe))
            cv2.imshow('depthimg', ku.dframe2depthimg(dframe))
            dframe_list.append(dframe)
            clframe_list.append(clframe)
            if cv2.waitKey(1) & 0xff == 27:
                break
    pickle.dump([clframe_list, dframe_list], open(f'res/knt/{f_name}_raw.pkl', 'wb'))


if __name__ == '__main__':
    f_name = 'tst'
    record_frame(f_name)

# pcdcenter = [0, 0, 1.5]
# base = pc.World(cam_pos=[0, 0, -1], lookat_pos=pcdcenter, w=1024, h=768)
#
# pcldnp = [None]
#
# def update(frk, pcldnp, task):
#     if pcldnp[0] is not None:
#         pcldnp[0].detachNode()
#     pcd = frk.getpcd()
#     pcd = pcd/1000
#     print(pcd)
#     pcldnp[0] = gm.gen_pointcloud(pcd)
#     pcldnp[0].attach_to(base)
#     return task.done

# taskMgr.doMethodLater(0.05, update, "update", extraArgs=[frk, pcldnp],
#                       appendTask=True)
#
# base.run()
