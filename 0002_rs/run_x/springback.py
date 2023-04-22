import os
import pickle
import time

import cv2
import numpy as np

import basis.robot_math as rm
import config
import motorcontrol.Motor as motor
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.vision_utils as vu
from sklearn import linear_model
import matplotlib.pyplot as plt

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def lasso_pre(X, y, x_pre):
    model = linear_model.Lasso(alpha=10)
    model.fit([[x] for x in X], y)
    print(model.coef_, model.intercept_)
    y_pre = model.predict([[x] for x in X])
    plt.grid()
    plt.plot(X, y, c='g')
    plt.plot(X, y_pre, c='r')
    plt.show()
    print(model.predict([[x_pre]]))
    return model.predict([[x_pre]])


def _action(fo, f_name, goal_angle, z_range, ulim=None, rgba=(0, 1, 0, 1)):
    textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img/phoxi/', 'springback', fo, f_name))
    # cv2.imshow("depthimg", depthimg)
    # cv2.waitKey(0)
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
    # textureimg = vu.enhance_grayimg(textureimg)
    lines = pcdu.extract_lines_from_pcd(textureimg, pcd, z_range=z_range, line_thresh=line_thresh,
                                        line_size_thresh=line_size_thresh)
    angle = np.degrees(rm.angle_between_vectors(lines[0][0], lines[1][0]))
    pcdu.show_pcd(lines[0][1], rgba=rgba)
    pcdu.show_pcd(lines[1][1], rgba=rgba)
    if ulim is None:
        if abs(angle - goal_angle) > abs((180 - angle) - goal_angle):
            angle = abs(180 - angle)
    else:
        if abs(angle - goal_angle) > abs((180 - angle) - goal_angle) and abs(180) - angle < ulim:
            angle = abs(180 - angle)
    return angle


def uniform_bend(s_angle, e_angle, interval, fo='steel'):
    _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"init.pkl"))
    cv2.imshow("depthimg", depthimg)
    cv2.waitKey(0)
    for a in range(s_angle, e_angle + 1, interval):
        print(a)
        if a == 0:
            motor.rot_degree(clockwise=0, rot_deg=interval)
            _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
            cv2.imshow("depthimg", depthimg)
            cv2.waitKey(0)

            motor.rot_degree(clockwise=1, rot_deg=interval)
            _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
            cv2.imshow("depthimg", depthimg)
            cv2.waitKey(0)
        else:
            motor.rot_degree(clockwise=0, rot_deg=interval * 2)
            _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
            cv2.imshow("depthimg", depthimg)
            cv2.waitKey(0)
            motor.rot_degree(clockwise=1, rot_deg=interval)
            _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
            cv2.imshow("depthimg", depthimg)
            cv2.waitKey(0)


def uniform_bend_avg(s_angle, e_angle, interval, z_range, line_thresh=.002, line_size_thresh=300, fo='steel'):
    textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"init.pkl"))
    cv2.imshow("depthimg", depthimg)
    cv2.waitKey(0)
    # textureimg = vu.enhance_grayimg(textureimg)
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)
    lines = pcdu.extract_lines_from_pcd(textureimg, pcd, z_range=z_range, line_thresh=line_thresh,
                                        line_size_thresh=line_size_thresh, toggledebug=True)
    sb_list = []
    for a in range(s_angle, e_angle + 1, interval):
        print('=============================')
        print('bending angle:', a)
        print('Avg. spring back:', np.mean(np.asarray(sb_list)))

        if a == 0:
            bend_angle = interval
            reverse_angle = interval
        else:
            bend_angle = interval * 2
            reverse_angle = interval

        motor.rot_degree(clockwise=0, rot_deg=bend_angle)
        goal = _action(goal_angle=a + interval, fo=fo, f_name=f"{str(a)}_goal.pkl", rgba=(0, 1, 0, 1))
        print('***************************************** goal:', goal)

        motor.rot_degree(clockwise=1, rot_deg=reverse_angle)
        res = _action(goal_angle=a + interval, fo=fo, f_name=f"{str(a)}_res.pkl", rgba=(1, 1, 0, 1))
        print('***************************************** result:', res)
        if abs(goal - res) > 10:
            sb_list.append(np.mean(np.asarray(sb_list)))
        else:
            sb_list.append(abs(goal - res))
        print('***************************************** spring back:', sb_list)

        motor.rot_degree(clockwise=0, rot_deg=reverse_angle + np.mean(np.asarray(sb_list)))
        time.sleep(1)
        motor.rot_degree(clockwise=1, rot_deg=reverse_angle + np.mean(np.asarray(sb_list)))
        refined = _action(goal_angle=a + interval, fo=fo, f_name=f"{str(a)}_refine.pkl", rgba=(0, 0, 1, 1))
        print('***************************************** refined:', refined)


def uniform_bend_lr(s_angle, e_angle, interval, z_range, line_thresh=.002, line_size_thresh=300, fo='steel'):
    textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"init.pkl"))
    cv2.imshow("depthimg", depthimg)
    cv2.waitKey(0)
    textureimg = vu.enhance_grayimg(textureimg)
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
    _ = pcdu.extract_lines_from_pcd(textureimg, pcd, z_range=z_range, line_thresh=line_thresh,
                                    line_size_thresh=line_size_thresh, toggledebug=True)
    sb_list = []
    for cnt, a in enumerate(range(s_angle, e_angle, interval)):
        print('=============================')
        print('bending angle:', a)
        print('Avg. spring back:', np.mean(np.asarray(sb_list)))

        if cnt == 0:
            bend_angle = interval + a
            reverse_angle = interval
        else:
            bend_angle = interval * 2
            reverse_angle = interval

        motor.rot_degree(clockwise=0, rot_deg=bend_angle)
        goal = _action(goal_angle=a + interval, z_range=z_range, fo=fo, f_name=f"{str(a)}_goal.pkl", rgba=(0, 1, 0, 1))
        print('***************************************** goal:', goal)
        motor.rot_degree(clockwise=1, rot_deg=reverse_angle)
        res = _action(goal_angle=a + interval, z_range=z_range, ulim=goal, fo=fo, f_name=f"{str(a)}_res.pkl",
                      rgba=(1, 1, 0, 1))
        print('***************************************** release:', res)
        if abs(goal - res) > 10:
            sb_list.append(np.mean(np.asarray(sb_list)))
        else:
            if goal - res < 0:
                res = 180 - res
            sb_list.append(abs(goal - res))
        print('***************************************** spring back:', sb_list)
        if len(sb_list) == 1:
            sb_pre = sb_list[0]
        else:
            sb_pre = lasso_pre(range(s_angle, a, interval), sb_list[:-1], a)

        motor.rot_degree(clockwise=0, rot_deg=reverse_angle + sb_pre)
        refined_goal = _action(goal_angle=a + interval, z_range=z_range, fo=fo, f_name=f"{str(a)}_refine_goal.pkl",
                               rgba=(0, 1, 1, 1))
        print('***************************************** refined goal:', refined_goal)
        motor.rot_degree(clockwise=1, rot_deg=reverse_angle + sb_pre)
        refined = _action(goal_angle=a + interval, z_range=z_range, ulim=refined_goal, fo=fo,
                          f_name=f"{str(a)}_refine.pkl",
                          rgba=(0, 0, 1, 1))
        print('***************************************** refined:', refined)


if __name__ == '__main__':
    import visualization.panda.world as wd

    motor = motor.MotorNema23()
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    base = wd.World(cam_pos=[1.5, 1.5, 1.5], lookat_pos=[0, 0, 0])

    fo = 'steel_refine_lr_2'
    z_range = (.12, .15)
    line_thresh = 0.0018
    line_size_thresh = 150

    uniform_bend_lr(s_angle=5, e_angle=170, interval=15, fo=fo,
                    z_range=z_range, line_thresh=line_thresh, line_size_thresh=line_size_thresh)
    base.run()
