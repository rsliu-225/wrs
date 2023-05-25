import os
import pickle

import cv2

import config
import drivers.devices.zivid.zivid_sdk as zivid
import modeling.geometric_model as gm
import utils.phoxi as phoxi
import visualization.panda.world as wd


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
    cam = phoxi.Phoxi(host=config.PHOXI_HOST)
    i = 0
    while i < img_num:
        if img_num == 1:
            grayimg, depthnparray_float32, pcd = cam.dumpalldata(f_name="img/" + path + f_name + ".pkl")
        else:
            grayimg, depthnparray_float32, pcd = \
                cam.dumpalldata(f_name="img/" + path + "_".join([f_name, str(i)]) + ".pkl")
        cv2.imshow("grayimg", depthnparray_float32)
        cv2.waitKey(0)
        i += 1


def get_img_zivid(f_name, path=''):
    cam = zivid.Zivid()
    raw_pcd, pcd_no_nan_indices, rgba = cam.get_pcd_rgba()
    rgbimg = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2RGB)
    pcd = raw_pcd[pcd_no_nan_indices]

    pcd_rgba = rgba.reshape(-1, 4)[pcd_no_nan_indices] / 255
    pickle.dump([pcd, pcd_rgba, rgbimg], open(os.path.join(config.ROOT, path, f_name + '.pkl'), 'wb'))
    cv2.imshow("rgba", rgbimg)
    cv2.waitKey(0)

    gm.gen_pointcloud(pcd, rgbas=pcd_rgba).attach_to(base)
    gm.gen_sphere((0, 0, 0)).attach_to(base)
    zivid_gm = gm.GeometricModel(os.path.join(config.ROOT, 'obstacles', 'zivid.stl'))
    zivid_gm.set_rgba((1, 1, 1, .3))
    zivid_gm.attach_to(base)
    base.run()


if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])

    # {"affine_mat": [[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
    #                 [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
    #                 [-0.202360828, 0.00546017392, -0.96800006, 0.94915224], [0.0, 0.0, 0.0, 1.0]]}
    f_name = "result"
    get_img_zivid(f_name, path='img/zivid/nbc/extrude_1/')
    # get_jnt(f_name, img_num, armname="rgt")
