import drivers.rpc.frtknt.frtknt_client as fc
import visualization.panda.world as pc
import modeling.geometric_model as gm

import cv2
import time
import pickle


def record(frk, f_name):
    depthimg_list = []
    rgbimg_list = []
    pcd_list = []
    while True:
        print(len(depthimg_list))
        depthimg = frk.getdepthimg()
        rgbimg = frk.getrgbimg()
        pcd = frk.getpcd()
        pcd = pcd / 1000
        cv2.imshow('rgbimg', rgbimg)
        cv2.imshow('depthimg', depthimg)
        depthimg_list.append(depthimg)
        rgbimg_list.append(rgbimg)
        pcd_list.append(pcd)
        if cv2.waitKey(1) & 0xff == 27:  # ESCで終了
            cv2.destroyAllWindows()
            break
    pickle.dump([rgbimg_list, depthimg_list, pcd_list], open(f'./res/knt/{f_name}', 'wb'))


def load(f_name):
    return pickle.load(open(f'./res/knt/{f_name}', 'rb'))


if __name__ == '__main__':
    frk = fc.FrtKnt(host="10.0.1.143:183001")
    f_name = 'tst.pkl'
    # record(frk, f_name)
    rgbimg_list, depthimg_list, pcd_list = load(f_name)
    for i, img in enumerate(depthimg_list):
        cv2.imshow(f_name, img)
        cv2.imshow(f_name, rgbimg_list[i])
        print(img.shape)
        print(rgbimg_list[i].shape)
        cv2.waitKey(0)

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
