import pickle
import os
import cv2
import config
import numpy as np

import utils.vision_utils as vu
import utils.pcd_utils as pcdu
import localenv.envloader as el
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

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


def hough_lines(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dst = cv2.Canny(img, 0, 100, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 100, None, 0, 0)
    line_set = []

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
    return line_set


def springback_from_img(fo, z_range, line_thresh=0.002, line_size_thresh=300):
    sb_dict = {}
    pcd_color = {'init': (1, 0, 0, 1), 'goal': (0, 1, 0, 1), 'res': (1, 1, 0, 1)}
    kpts_color = {'init': (1, 0, 0, 1), 'goal': (0, 1, 0, 1), 'res': (1, 1, 0, 1)}
    for f in os.listdir(os.path.join(config.ROOT, 'img/phoxi', fo)):
        if f[-3:] != 'pkl':
            continue
        print(f'------------{f}------------')
        if f.split('.pkl')[0] == 'init':
            key = 'init'
            angle = 0
        else:
            angle = f.split('.pkl')[0].split('_')[0]
            if f.split('.pkl')[0].split('_')[1] == 'res':
                key = 'res'
            else:
                key = 'goal'
        if angle not in sb_dict.keys():
            sb_dict[angle] = {}
        sb_dict[angle][key] = []

        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
        img = enhance_grayimg(textureimg)
        # cv2.imshow('', img)
        # cv2.waitKey(0)

        pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)
        pcd_pix = pcd.reshape(textureimg.shape[0], textureimg.shape[1], 3)
        mask_1 = np.where(pcd_pix[:, :, 2] < z_range[1], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
            np.uint8)
        mask_2 = np.where(pcd_pix[:, :, 2] > z_range[0], 255, 0).reshape((img.shape[0], img.shape[1], 1)).astype(
            np.uint8)
        mask = cv2.bitwise_and(mask_1, mask_2)
        img = cv2.bitwise_and(img, mask)

        pcd_crop = pcdu.crop_pcd(pcd, x_range=(0, 1), y_range=(-1, 1), z_range=z_range)
        pcdu.show_pcd(pcd_crop, rgba=(1, 1, 1, .5))

        while 1:
            print(f'------------{len(pcd_crop)}------------')
            line = pyrsc.Line()
            line.fit(pcd_crop, thresh=line_thresh, maxIteration=1000)
            if len(line.inliers) > line_size_thresh:
                pcdu.show_pcd(pcd_crop[line.inliers], rgba=pcd_color[key])
                # gm.gen_sphere(line.B, rgba=(0, 0, 1, 1), radius=.002).attach_to(base)
                print(line.A, line.B)
                pcd_crop = np.delete(pcd_crop, line.inliers, axis=0)
                sb_dict[angle][key].append(line.A)
            else:
                break

        # gm.gen_stick(spos=line.B, epos=line.A + line.B, rgba=pcd_color[clr]).attach_to(base)
        # kpts = get_kpts_gmm(pcd_crop, rgba=kpts_color[clr])

        # cv2.imshow('', mask)
        # cv2.waitKey(0)
        # cv2.imshow('', img)
        # cv2.waitKey(0)
    pickle.dump(sb_dict, open(f'./{fo.split("/")[1]}_springback.pkl', 'wb'))
    return sb_dict


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0])
    rbt = el.loadYumi(showrbt=True)

    fo = 'springback/steel'
    z_range = (.12, .15)
    # sb_dict = springback_from_img(fo, z_range)
    sb_dict = pickle.load(open('./steel_springback.pkl', 'rb'))
    # print(sb_dict)
    X = []
    sb_err = []
    bend_err = []
    for k, v in sb_dict.items():
        if int(k) == 0:
            continue
        # goal = np.degrees(rm.angle_between_vectors(sb_dict[0]['init'][0], sb_dict[k]['goal'][1]))
        # res = np.degrees(rm.angle_between_vectors(sb_dict[0]['init'][0], sb_dict[k]['res'][1]))
        # if goal > 90 and res < 90:
        #     goal = 180 - goal
        # elif goal < 90 and res > 90:
        #     res = 180 - res
        # diff = abs(goal - res)
        # print(goal)
        # print(res)

        res = np.degrees(rm.angle_between_vectors(sb_dict[k]['res'][0], sb_dict[k]['res'][1]))
        goal = np.degrees(rm.angle_between_vectors(sb_dict[k]['goal'][0], sb_dict[k]['goal'][1]))
        print(goal, res)

        if abs(res - int(k)) > 30:
            res = abs(180 - res)
        if abs(goal - int(k)) > 30:
            goal = abs(180 - goal)
        print(int(k) + 15, goal, res)
        sb = goal - res
        bend = int(k) + 15 - goal
        # if diff > 90:
        #     diff = 180 - diff

        print('spring back:', sb)
        print('------------')
        sb_err.append(sb)
        bend_err.append(bend)
        X.append(int(k))

    sort_inx = np.argsort(X)
    X = [X[i] for i in sort_inx]
    sb_err = [sb_err[i] for i in sort_inx]
    bend_err = [bend_err[i] for i in sort_inx]

    plt.grid()
    plt.xticks(X)
    plt.plot(X, sb_err, c='gold')
    plt.plot(X, [np.mean(sb_err)] * len(X), c='gold', linestyle='dashed')
    plt.plot(X, bend_err, c='limegreen')
    plt.plot(X, [np.mean(bend_err)] * len(X), c='limegreen', linestyle='dashed')

    # plt.plot(X, np.asarray(bend_err) + np.asarray(sb_err))
    plt.show()
    base.run()
