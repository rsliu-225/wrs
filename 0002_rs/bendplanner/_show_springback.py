import pickle
import os
import config
import numpy as np
import cv2
import basis.robot_math as rm
import utils.pcd_utils as pcdu
import localenv.envloader as el
import basis.o3dhelper as o3dh

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def show_wire(fo, f, z_range=(.15, .18), rgba=(1, 0, 0, 1)):
    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f), 'rb'))
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)

    pcd_crop = pcdu.crop_pcd(pcd, x_range=(.4, 1), y_range=(-.3, .3), z_range=z_range)
    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
    pcdu.show_pcd(pcd_crop, rgba=rgba)


if __name__ == '__main__':
    fo = 'stick/penta'
    base, env = el.loadEnv_yumi()

    show_wire(fo, '1_res.pkl', z_range=(.12, .15), rgba=(1, 0, 0, 1))
    show_wire(fo, '1_goal.pkl', z_range=(.15, .18), rgba=(0, 1, 0, 1))
    show_wire(fo, '2_res.pkl', z_range=(.12, .15), rgba=(1, 0, 0, 1))
    show_wire(fo, '2_goal.pkl', z_range=(.15, .18), rgba=(0, 1, 0, 1))

    base.run()
