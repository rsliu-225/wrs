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

if __name__ == '__main__':
    fo = 'stick'
    f = 'penta'
    base, env = el.loadEnv_yumi()
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f}.pkl'), 'rb'))

    z_range = (.2, .25)
    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f, 'result.pkl'), 'rb'))
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)

    pcd_pix = pcd.reshape(textureimg.shape[0], textureimg.shape[1], 3)
    pcd_crop = pcdu.crop_pcd(pcd, x_range=(.4, 1), y_range=(-.3, .3), z_range=z_range)
    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .5))
    pcdu.show_pcd(pcd_crop, rgba=(1, 0, 0, 1))

    rmse, fitness, trans = o3dh.registration_ptpt(src=goal_pseq, tgt=pcd_crop)
    print(rmse, fitness)
    goal_pseq = rm.homomat_transform_points(trans, goal_pseq)
    pcdu.show_pcd(goal_pseq, rgba=(0, 1, 0, 1))

    base.run()

    mask = np.where(pcd_pix[:, :, 2] < z_range[1], 255, 0).reshape(
        (textureimg.shape[0], textureimg.shape[1], 1)).astype(np.uint8)
    img = cv2.bitwise_and(textureimg, mask)
    cv2.imshow()
