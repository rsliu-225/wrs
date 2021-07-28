import os
import pickle

import cv2

import config
import manipulation.grip.robotiqhe.robotiqhe as rtqhe
import robotcon.ur3edual as ur3ex
import utils.phoxi as phoxi


def get_jnt(f_name, img_num, armname="rgt"):
    hndfa = rtqhe.HandFactory()
    rbtx = ur3ex.Ur3EDualUrx(hndfa)
    jnts = rbtx.getjnts(armname)
    i = 0
    while i < img_num:
        if img_num == 1:
            pickle.dump(jnts, open(config.ROOT + "/img/jnts/" + "_".join([f_name, armname]) + ".pkl", "wb"))
        else:
            pickle.dump(jnts, open(config.ROOT + "/img/jnts/" + "_".join([f_name, str(i), armname]) + ".pkl", "wb"))
        i += 1
        print(jnts)


def get_img(f_name, img_num):
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    i = 0
    while i < img_num:
        if img_num == 1:
            grayimg, depthnparray_float32, pcd = phxi.dumpalldata(f_name="img/" + f_name + ".pkl")
        else:
            grayimg, depthnparray_float32, pcd = phxi.dumpalldata(f_name="img/" + "_".join([f_name, str(i)]) + ".pkl")
        cv2.imshow("grayimg", grayimg)
        cv2.waitKey(0)
        i += 1


if __name__ == '__main__':
    f_name = "phoxi_tempdata_raft_circle_f"
    img_num = 1
    get_img(f_name, img_num)
    # get_jnt(f_name, img_num, armname="rgt")
