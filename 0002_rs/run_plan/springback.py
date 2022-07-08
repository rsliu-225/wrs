import pickle
import os
import cv2
import config
import numpy as np

import utils.vision_utils as vu

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


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


def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


if __name__ == '__main__':
    # import visualization.panda.world as wd
    # import modeling.geometric_model as gm

    # base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    fo = 'springback/alu'
    line_set = []
    for f in os.listdir(os.path.join(config.ROOT, 'img/phoxi', fo)):
        if f[-3:] != 'pkl':
            continue
        print(f'------------{f}------------')
        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))

        img = textureimg
        # img = cv2.cvtColor(textureimg, cv2.COLOR_GRAY2BGR)
        # img = img[0:400, 200:700]

        mask = np.ones((img.shape[0], img.shape[1], 1)).astype(np.uint8)
        mask[50:400, 300:550] = 255

        # pcd = trans_pcd(pcd, affine_mat)
        # pcd = pcd.reshape(textureimg.shape[0],textureimg.shape[1], 3)
        #
        # mask = np.where(pcd[:, :, 2]<.2, 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(np.uint8)
        img = cv2.bitwise_and(img, mask)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = increase_brightness(img, 30)

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

        cv2.imshow('', dst)
        cv2.waitKey(0)
        cv2.imshow('', img)
        cv2.waitKey(0)
