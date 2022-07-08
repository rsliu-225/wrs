import pickle
import os
import cv2
import config
import numpy as np

import utils.vision_utils as vu
import utils.pcd_utils as pcdu
import localenv.envloader as el
from sklearn.mixture import GaussianMixture

import pyransac3d as pyrsc

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    if show:
        for p in gmix.means_:
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)
    return gmix.means_


def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = np.uint8(np.clip((2 * (np.int16(img) - 30) + 60), 0, 255))

    return img


def enhance_grayimg(grayimg):
    if len(grayimg.shape) == 2 or grayimg.shape[2] == 1:
        grayimg = grayimg.reshape(grayimg.shape[:2])
    return cv2.equalizeHist(grayimg)


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    rbt = el.loadYumi(showrbt=True)

    fo = 'springback/steel'
    line_set = []
    clr = 0
    vecs = []
    pcd_color = [(1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1)]
    kpts_color = [(1, 0, 0, 1), (0, 1, 0, 1), (1, 1, 0, 1)]
    for f in os.listdir(os.path.join(config.ROOT, 'img/phoxi', fo)):
        if f[-3:] != 'pkl':
            continue
        print(f'------------{f}------------')
        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
        img = enhance_grayimg(textureimg)
        # cv2.imshow('', img)
        # cv2.waitKey(0)

        pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)
        # pcdu.show_pcd(pcd)
        # base.run()
        pcd_pix = pcd.reshape(textureimg.shape[0], textureimg.shape[1], 3)
        z_range = (.12, .15)
        mask_1 = np.where(pcd_pix[:, :, 2] < z_range[1], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
            np.uint8)
        mask_2 = np.where(pcd_pix[:, :, 2] > z_range[0], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
            np.uint8)
        mask = cv2.bitwise_and(mask_1, mask_2)
        img = cv2.bitwise_and(img, mask)

        pcd_crop = pcdu.crop_pcd(pcd, x_range=(0, 1), y_range=(-1, 1), z_range=z_range)

        line = pyrsc.Line()
        line.fit(pcd_crop, thresh=0.002, maxIteration=1000)
        pcdu.show_pcd(pcd_crop[line.inliers], rgba=pcd_color[clr])
        print(line.A, line.B)
        vecs.append(line.A)
        # gm.gen_stick(spos=line.B, epos=line.A + line.B, rgba=pcd_color[clr]).attach_to(base)
        # pcdu.show_pcd(pcd_crop, rgba=pcd_color[clr])
        # kpts = get_kpts_gmm(pcd_crop, rgba=kpts_color[clr])

        clr += 1
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # cv2.imshow('', mask)
        # cv2.waitKey(0)
        # cv2.imshow('', img)
        # cv2.waitKey(0)

        dst = cv2.Canny(img, 0, 100, None, 3)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 100, None, 0, 0)

        if lines is not None:
            for i in range(len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))

                if x0 > 540 or x0 < 300:
                    continue
                # print(x0, y0)
                print(np.degrees(theta), pt1, pt2)
                cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
                line_set.append((pt1, pt2))

        # cv2.imshow('', dst)
        # cv2.waitKey(0)
        # cv2.imshow('', img)
        # cv2.waitKey(0)

    print(np.degrees(rm.angle_between_vectors(vecs[0], vecs[1])))
    print(np.degrees(rm.angle_between_vectors(vecs[0], vecs[2])))
    base.run()
