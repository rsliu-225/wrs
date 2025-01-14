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
    pcd_color = {'init': (1, 0, 0, 1), 'goal': (0, 1, 0, 1), 'res': (250 / 255, 220 / 255, 55 / 255, 1),
                 'refine': (0, 0, 1, 1), 'refine_goal': (0, 1, 1, 1)}
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
            continue
        else:
            angle = int(f.split(ext_str)[0].split('_')[0]) + 15
            if f.split(ext_str)[0].split('_')[1] == 'res':
                key = 'res'
            elif f.split(ext_str)[0].split('_')[1] == 'goal':
                key = 'goal'
            elif f.split(ext_str)[0].split('_')[1] == 'refine' and f.split(ext_str)[0].split('_')[-1] == 'refine':
                key = 'refine'
            else:
                key = 'refine_goal'
                # continue

        if angle not in sb_dict.keys():
            sb_dict[angle] = {}
        sb_dict[angle][key] = []

        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
        pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
        # pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
        img = vu.enhance_grayimg(textureimg)
        lines = pcdu.extract_lines_from_pcd(img, pcd, z_range=z_range, line_thresh=line_thresh,
                                            line_size_thresh=line_size_thresh, toggledebug=False)
        for slope, pts in lines:
            pcdu.show_pcd(pts, rgba=pcd_color[key])
            sb_dict[angle][key].append(slope)

    pickle.dump(sb_dict,
                open(os.path.join(config.ROOT, 'bendplanner/springback', f'{fo.split("/")[1]}_springback.pkl'), 'wb'))
    return sb_dict


def update_springback_from_img(fo, z_range, line_thresh=.002, line_size_thresh=300):
    sb_dict = {}
    pcd_color = {'init': (1, 0, 0, 1), 'goal': (0, 1, 0, 1), 'res': (250 / 255, 220 / 255, 55 / 255, 1),
                 'refine': (0, 0, 1, 1), 'refine_goal': (0, 1, 1, 1)}
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
            continue
        else:
            angle = int(f.split(ext_str)[0].split('_')[0]) + 15
            if f.split(ext_str)[0].split('_')[1] == 'res':
                key = 'res'
            elif f.split(ext_str)[0].split('_')[1] == 'goal':
                key = 'goal'
            elif f.split(ext_str)[0].split('_')[1] == 'refine' and f.split(ext_str)[0].split('_')[-1] == 'refine':
                key = 'refine'
            else:
                key = 'refine_goal'
                # continue

        if angle not in sb_dict.keys():
            sb_dict[angle] = {}
        sb_dict[angle][key] = []

        textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
        pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
        # pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
        img = vu.enhance_grayimg(textureimg)
        lines = pcdu.extract_lines_from_pcd(img, pcd, z_range=z_range, line_thresh=line_thresh,
                                            line_size_thresh=line_size_thresh, toggledebug=False)
        for slope, pts in lines:
            pcdu.show_pcd(pts, rgba=pcd_color[key])
            sb_dict[angle][key].append(slope)

    pickle.dump(sb_dict,
                open(os.path.join(config.ROOT, 'bendplanner/springback', f'{fo.split("/")[1]}_springback.pkl'), 'wb'))
    return sb_dict


def _get_angle_from_vecs(v1, v2, gt):
    angle = np.degrees(rm.angle_between_vectors(v1, v2))
    if abs(angle - gt) > abs((180 - angle) - gt):
        angle = abs(180 - angle)
    return angle


def show_data(input_dict):
    X = []
    sb_err_list = []
    bend_err_list = []
    refined_err_list = []
    refine_goal_list = []
    # fig = plt.figure()
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.size"] = 24
    # ax = fig.add_subplot(1, 1, 1)
    # grid_on(ax)
    # ax.set_xticks([v for v in range(15, 166, 30)])
    # ax.set_yticks([v for v in range(-2, 12, 2)])

    for k, v in input_dict.items():
        if int(k) == 0:
            continue
        gt = int(k)
        res = _get_angle_from_vecs(input_dict[k]['res'][0], input_dict[k]['res'][1], gt)
        goal = _get_angle_from_vecs(input_dict[k]['goal'][0], input_dict[k]['goal'][1], gt)
        refine = _get_angle_from_vecs(input_dict[k]['refine'][0], input_dict[k]['refine'][1], gt)
        refine_goal = _get_angle_from_vecs(input_dict[k]['refine_goal'][0], input_dict[k]['refine_goal'][1], gt)
        print(f'------------{int(k) + 15}------------')
        sb = goal - res
        if sb < 0:
            res = 180 - res
            # refine = 180 - refine
            sb = goal - res

        print('goal, result, refined', goal, res, refine)
        print('spring back:', sb)

        sb_err_list.append(sb)
        bend_err_list.append(gt - goal)
        if abs(refine - goal) < abs((180 - refine) - goal):
            refined_err_list.append(refine - goal)
        else:
            refined_err_list.append((180 - refine) - goal)

        refine_goal_list.append(refine_goal)
        X.append(goal)
        # if len(X) > 1 and gt <= 150:
        #     pre = lasso_pre(X, sb_err_list, gt + 15)
        #     print('prediction:', pre)
        #     ax.scatter([gt + 15], [pre], c='g')
        # elif gt < 150:
        #     ax.scatter([gt + 15], [np.mean(sb_err_list)], c='g')

    sort_inx = np.argsort(X)
    X = [X[i] for i in sort_inx]
    sb_err_list = [sb_err_list[i] for i in sort_inx]
    bend_err_list = [bend_err_list[i] for i in sort_inx]
    refined_err_list = [refined_err_list[i] for i in sort_inx]

    ax.scatter(X, sb_err_list, marker='x', c='gold')
    # plt.plot(X, [np.mean(sb_err_list)] * len(X), c='gold', linestyle='dashed')
    # print(X, sb_err_list)
    lasso_pre(X, sb_err_list, 0, plot=True)

    # plt.plot(X, bend_err_list, c='g')
    # plt.plot(X, [np.mean(bend_err_list)] * len(X), c='g', linestyle='dashed')

    plt.scatter(X, refined_err_list, marker='x', c='b')
    plt.plot([14, 175], [np.mean(refined_err_list)] * 2, c='b', linestyle='dashed')

    X = [72.0, 72.0, 72.0, 72.0, 90.0, 90.0, 109.63, 13.41, 29.42, 35.25]
    y = [3.88, 3.81, 3.79, 2.23, 4.12, 5.94, 5.71, 3.22, 2.88, 3.09]
    # X = [72.0, 72.0, 72.0, 72.0]
    # y = [4.51, 5.57, 4.06, 4.17]

    X = [72.0, 72.0 + 3.88, 72.0 + 3.86, 72.0 + 3.82, 90.0, 90.0 + 4.12, 109.63, 13.41 + 5.71, 29.42 + 3.77,
         35.25 + 3.57]
    # X = [72.0, 72.0 + 4.51, 72.0 + 5.04, 72.0 + 4.71]
    ax.scatter(X, y, marker='x', c='r')

    # plt.plot(X, np.asarray(bend_err) + np.asarray(sb_err_list))


def show_data_fix(input_dict):
    X = []
    sb_err_list = []
    bend_err_list = []
    refined_err_list = []
    refine_goal_list = []

    for k, v in input_dict.items():
        if int(k) == 0:
            continue
        gt = int(k)
        res = _get_angle_from_vecs(input_dict[k]['res'][0], input_dict[k]['res'][1], gt)
        goal = _get_angle_from_vecs(input_dict[k]['goal'][0], input_dict[k]['goal'][1], gt)
        refine = _get_angle_from_vecs(input_dict[k]['refine'][0], input_dict[k]['refine'][1], gt)
        refine_goal = _get_angle_from_vecs(input_dict[k]['refine_goal'][0], input_dict[k]['refine_goal'][1], gt)
        print(f'------------{int(k) + 15}------------')
        sb = goal - res
        if sb < 0:
            res = 180 - res
            refine = 180 - refine
            sb = goal - res

        print('goal, result, refined', goal, res, refine)
        print('spring back:', sb)
        if abs(sb) > 10:
            continue

        sb_err_list.append(sb)
        bend_err_list.append(gt - goal)
        if abs(refine - goal) < abs((180 - refine) - goal):
            refined_err_list.append(refine - goal)
        else:
            refined_err_list.append((180 - refine) - goal)

        refine_goal_list.append(refine_goal)
        X.append(goal)

    sort_inx = np.argsort(X)
    X = [X[i] for i in sort_inx]
    sb_err_list = [sb_err_list[i] for i in sort_inx]
    bend_err_list = [bend_err_list[i] for i in sort_inx]
    refined_err_list = [refined_err_list[i] for i in sort_inx]

    ax.scatter(X, sb_err_list, s=50, edgecolors='darkorange', facecolor='none')
    # plt.plot(X, [np.mean(sb_err_list)] * len(X), c='gold', linestyle='dashed')

    # plt.plot(X, bend_err_list, c='g')
    # plt.plot(X, [np.mean(bend_err_list)] * len(X), c='g', linestyle='dashed')

    # plt.scatter(X, refined_err_list, marker='o', c='cyan')
    ax.scatter(X, refined_err_list, s=50, edgecolors='cyan', facecolor='none')
    plt.plot([14, 175], [np.mean(refined_err_list)] * 2, c='cyan', linestyle='dashed')


def lasso_pre(X, y, x_pre, plot=False):
    model = linear_model.Lasso(alpha=10)
    model.fit([[x] for x in X], y)
    print('model', model.coef_, model.intercept_)
    y_pre = model.predict([[x] for x in X])
    std = np.std(np.asarray(y) - np.asarray(y_pre))
    mean = np.mean(abs(np.asarray(y) - np.asarray(y_pre)))
    print(std, mean)
    if plot:
        plt.plot(X, y_pre, c='gold', linestyle='dashed')
        plt.fill_between(X, y_pre - std, y_pre + std, alpha=0.2, color='gold')
        # plt.plot(X, [(x + model.intercept_) / (1 - model.coef_[0]) - x for x in X], c='r', linestyle='dashed')
    pre = (x_pre + model.intercept_) / (1 - model.coef_[0]) - x_pre
    # return model.predict([[x_pre]])
    return pre


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle='--', alpha=.2)


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import basis.robot_math as rm

    base = wd.World(cam_pos=[.8, 0, 1.5], lookat_pos=[0, 0, 0])
    rbt = el.loadYumi(showrbt=True)

    mtr = 'steel'
    # fo = f'springback/{mtr}_refine_lr_1'
    fo = f'springback/{mtr}_fix_lr_3'

    fig = plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    ax = fig.add_subplot(1, 1, 1)
    grid_on(ax)
    ax.set_xticks([v for v in range(15, 166, 30)])
    ax.set_yticks([v for v in range(-2, 12, 2)])
    ax.set_ylim([-3.5, 11])
    ax.set_xlim([5, 180])

    z_range = (.15, .18)
    line_thresh = 0.0029
    line_size_thresh = 500

    # sb_dict_fix = springback_from_img(fo, z_range, line_thresh, line_size_thresh)
    # sb_dict_fix = pickle.load(
    #     open(os.path.join(config.ROOT, 'bendplanner/', f'{fo}_springback.pkl'), 'rb'))

    # sb_dict_fix = {}
    # sb_dict_fix_1 = pickle.load(
    #     open(os.path.join(config.ROOT, 'bendplanner/', f'springback/{mtr}_fix_lr_1_springback.pkl'), 'rb'))
    # sb_dict_fix_2 = pickle.load(
    #     open(os.path.join(config.ROOT, 'bendplanner/', f'springback/{mtr}_fix_lr_2_springback.pkl'), 'rb'))
    # for k in sb_dict_fix_1.keys():
    #     if k in [15, 150, 165]:
    #         sb_dict_fix[k] = sb_dict_fix_1[k]
    #     else:
    #         sb_dict_fix[k] = sb_dict_fix_2[k]

    # show_data_fix(sb_dict_fix)

    # X = [72.0, 72.0, 72.0, 72.0]
    # y = [3.88, 3.81, 3.79, 2.23]
    # y = [4.51, 5.57, 4.06, 4.17]
    # X = [90.0, 90.0, 90, 90]
    # y = [4.12, 5.94, None, None]
    X = [109.63, 8.0, 14.18, 14.33, 13.41, 29.42, 35.25, 79.11]
    y = [5.71, None, None, None, 3.22, 2.88, 3.09, None]
    res = ['-']
    for i in range(1, len(X)):
        # print(X[:i], y[:i])
        pre = lasso_pre([x for j, x in enumerate(X[:i]) if y[j] is not None],
                              [v for v in y[:i] if v is not None],
                              X[i], plot=False)
        print(X[i], pre)
        print('--')
        res.append(X[i]+pre)
    print('/'.join([str(v) for v in res]))

    #
    # sb_dict = {}
    # sb_dict_1 = pickle.load(
    #     open(os.path.join(config.ROOT, 'bendplanner/', f'springback/{mtr}_refine_lr_1_springback.pkl'), 'rb'))
    # # sb_dict_2 = pickle.load(
    # #     open(os.path.join(config.ROOT, 'bendplanner/', f'springback/{mtr}_refine_lr_2_springback.pkl'), 'rb'))
    # # sb_dict_3 = pickle.load(
    # #     open(os.path.join(config.ROOT, 'bendplanner/', f'springback/{mtr}_refine_lr_3_springback.pkl'), 'rb'))
    #
    # sb_dict.update(sb_dict_1)
    # # sb_dict.update(sb_dict_2)
    # # sb_dict.update(sb_dict_3)
    # show_data(sb_dict)
    #
    # plt.show()
    #
    # base.run()
