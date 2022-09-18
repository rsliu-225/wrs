import os
import pickle

import cv2

import config
import utils.phoxi as phoxi


def get_jnt(f_name, img_num, armname="rgt"):
    import robot_con.ur.ur3e_dual_x as ur3ex
    rbtx = ur3ex.Ur3EDualUrx()
    jnts = rbtx.get_jnt_values(armname)
    i = 0
    while i < img_num:
        if img_num == 1:
            pickle.dump(jnts, open(config.ROOT + "/img/jnts/" + "_".join([f_name, armname]) + ".pkl", "wb"))
        else:
            pickle.dump(jnts, open(config.ROOT + "/img/jnts/" + "_".join([f_name, str(i), armname]) + ".pkl", "wb"))
        i += 1
        print(jnts)


def get_img(f_name, img_num, path=''):
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    i = 0
    while i < img_num:
        if img_num == 1:
            grayimg, depthnparray_float32, pcd = phxi.dumpalldata(f_name="img/" + path + f_name + ".pkl")
        else:
            grayimg, depthnparray_float32, pcd = \
                phxi.dumpalldata(f_name="img/" + path + "_".join([f_name, str(i)]) + ".pkl")
        cv2.imshow("grayimg", depthnparray_float32)
        cv2.waitKey(0)
        i += 1


if __name__ == '__main__':
    # {"affine_mat": [[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
    #                 [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
    #                 [-0.202360828, 0.00546017392, -0.96800006, 0.94915224], [0.0, 0.0, 0.0, 1.0]]}
    f_name = "result"
    img_num = 1
    get_img(f_name, img_num, path='phoxi/exp_bend/stick/randomc/')
    # get_jnt(f_name, img_num, armname="rgt")
