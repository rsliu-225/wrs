import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

import config
import localenv.envloader as el
import utils.pcd_utils as pcdu
import utils.vision_utils as vu
from sklearn import linear_model

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


def springback_from_img(fo, z_range, line_thresh=.002, line_size_thresh=300):
    sb_dict = {}
    pcd_color = {'init': (1, 0, 0, 1), 'goal': (0, 1, 0, 1), 'res': (1, 1, 0, 1), 'refine': (0, 0, 1, 1),
                 'refine_goal': (0, 1, 1, 1)}
    ext_str = '.pkl'
    for f in os.listdir(os.path.join(config.ROOT, 'img/phoxi', fo)):
        if f[-3:] != 'pkl':
            continue
        # if f[0] != '0':
        #     continue
        print(f'------------{f}------------')
        if f.split(ext_str)[0] == 'init':
            key = 'init'
            angle = 0
        else:
            angle = f.split(ext_str)[0].split('_')[0]
            if f.split(ext_str)[0].split('_')[1] == 'res':
                key = 'res'
            elif f.split(ext_str)[0].split('_')[1] == 'goal':
                key = 'goal'
            elif f.split(ext_str)[0].split('_')[1] == 'refine' and f.split(ext_str)[0].split('_')[-1] == 'refine':
                key = 'refine'
            else:
                key = 'refine_goal'

        if angle not in sb_dict.keys():
            sb_dict[angle] = {}
        sb_dict[angle][key] = []

        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
        pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)
        pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
        img = vu.enhance_grayimg(textureimg)
        lines = pcdu.extract_lines_from_pcd(img, pcd, z_range=z_range, line_thresh=line_thresh,
                                            line_size_thresh=line_size_thresh, toggledebug=False)
        for slope, pts in lines:
            pcdu.show_pcd(pts, rgba=pcd_color[key])
            sb_dict[angle][key].append(slope)
        res = _get_angle_from_vecs(sb_dict[angle][key][0], sb_dict[angle][key][1], 15)
        print(res)
        # gm.gen_stick(spos=line.B, epos=line.A + line.B, rgba=pcd_color[clr]).attach_to(base)
        # kpts = get_kpts_gmm(pcd_crop, rgba=kpts_color[clr])

    pickle.dump(sb_dict,
                open(os.path.join(config.ROOT, 'bendplanner/springback', f'{fo.split("/")[1]}_springback.pkl'), 'wb'))
    return sb_dict


def _get_angle_from_vecs(v1, v2, gt):
    angle = np.degrees(rm.angle_between_vectors(v1, v2))
    if abs(angle - gt) > abs((180 - angle) - gt):
        angle = abs(180 - angle)
    return angle


def show_data(input_dict, gt_offset=0):
    X = []
    sb_err = []
    bend_err = []
    for k, v in input_dict.items():
        if int(k) == 0:
            continue
        gt = int(k) + 15 + gt_offset
        res = _get_angle_from_vecs(input_dict[k]['res'][0], input_dict[k]['res'][1], gt)
        goal = _get_angle_from_vecs(input_dict[k]['goal'][0], input_dict[k]['goal'][1], gt)
        print(f'------------{int(k) + 15}------------')
        sb = gt - res
        if sb < 0:
            res = 180 - res
            sb = gt - res
        bend = gt - goal

        print('goal, result', goal, res)
        print('spring back:', sb)

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
    plt.plot(X, bend_err, c='g')
    plt.plot(X, [np.mean(bend_err)] * len(X), c='g', linestyle='dashed')

    next = lasso_pre(X[:2], sb_err[:2], 45)
    print(next)

    # plt.plot(X, np.asarray(bend_err) + np.asarray(sb_err))
    plt.show()


def show_data_w_refine(input_dict):
    X = []
    sb_err = []
    bend_err = []
    refined_err = []
    for k, v in input_dict.items():
        if int(k) == 0:
            continue
        gt = int(k) + 15
        res = _get_angle_from_vecs(input_dict[k]['res'][0], input_dict[k]['res'][1], gt)
        goal = _get_angle_from_vecs(input_dict[k]['goal'][0], input_dict[k]['goal'][1], gt)
        refine = _get_angle_from_vecs(input_dict[k]['refine'][0], input_dict[k]['refine'][1], gt)
        print(f'------------{int(k) + 15}------------')
        sb = goal - res
        if sb < 0:
            res = 180 - res
            sb = goal - res
        bend = gt - goal

        print('goal, result, refined', goal, res, refine)
        print('spring back:', sb)

        sb_err.append(sb)
        bend_err.append(bend)
        refined_err.append(goal - refine)
        X.append(int(k))

    sort_inx = np.argsort(X)
    X = [X[i] for i in sort_inx]
    sb_err = [sb_err[i] for i in sort_inx]
    bend_err = [bend_err[i] for i in sort_inx]
    refined_err = [refined_err[i] for i in sort_inx]

    plt.grid()
    plt.xticks(X)
    plt.plot(X, sb_err, c='gold')

    plt.plot(X, [np.mean(sb_err)] * len(X), c='gold', linestyle='dashed')
    plt.plot(X, bend_err, c='g')
    plt.plot(X, [np.mean(bend_err)] * len(X), c='g', linestyle='dashed')

    plt.plot(X, refined_err, c='b')
    plt.plot(X, [np.mean(refined_err)] * len(X), c='b', linestyle='dashed')

    # next = lasso_pre(X[:2], sb_err[:2], 45)
    # print(next)

    # plt.plot(X, np.asarray(bend_err) + np.asarray(sb_err))
    plt.show()


def lasso_pre(X, y, x_pre):
    model = linear_model.Lasso(alpha=10)
    model.fit([[x] for x in X], y)
    print(model.coef_, model.intercept_)
    y_pre = model.predict([[x] for x in X])
    plt.plot(X, y_pre, c='r')
    return model.predict([[x_pre]])


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0])
    rbt = el.loadYumi(showrbt=True)

    fo = 'springback/steel_refine_lr_1'
    z_range = (.15, .18)
    line_thresh = 0.0024
    line_size_thresh = 300
    # sb_dict = springback_from_img(fo, z_range, line_thresh, line_size_thresh)
    sb_dict = pickle.load(
        open(os.path.join(config.ROOT, 'bendplanner/springback', f'{fo.split("/")[1]}_springback.pkl'), 'rb'))

    if 'refine' in sb_dict['0'].keys():
        show_data_w_refine(sb_dict)
    else:
        show_data(sb_dict)

    base.run()
