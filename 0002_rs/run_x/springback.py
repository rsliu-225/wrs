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


# def lasso_pre(X, y, x_pre):
#     model = linear_model.Lasso(alpha=10)
#     model.fit([[x] for x in X], y)
#     print(model.coef_, model.intercept_)
#     y_pre = model.predict([[x] for x in X])
#     plt.grid()
#     plt.plot(X, y, c='g')
#     plt.plot(X, y_pre, c='r')
#     plt.show()
#     print(model.predict([[x_pre]]))
#     return model.predict([[x_pre]])

def lasso_pre(X, y, x_pre, plot=False):
    model = linear_model.Lasso(alpha=10)
    model.fit([[x] for x in X], y)
    print('model', model.coef_, model.intercept_)
    y_pre = model.predict([[x] for x in X])
    if plot:
        plt.plot(X, y_pre, c='gold', linestyle='dashed')
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


def uniform_bend_lr_fix(s_angle, e_angle, interval, z_range, line_thresh=.002, line_size_thresh=300, fo='steel'):
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

        # X = [15.92373831323462, 21.37597014362774, 24.357836513027678, 29.47014813049081, 34.78468890617915,
        #      39.60658144363412, 45.06994420279963, 50.68232674484642, 54.746310535386584, 60.302537763068266,
        #      65.50825800441525, 68.29504544018977, 74.67257408799554, 80.33885467891466, 84.68041017656921,
        #      89.93573058394696, 95.0679131510559, 99.306531641558, 104.67575119020732, 110.56710855013736,
        #      113.57555146356385, 120.78105109433913, 125.8074331815408, 128.51079043010927, 134.8858936927587,
        #      141.70941400924494, 144.29981686874018, 151.24959917117653, 156.68331555834217, 158.70889541881522,
        #      165.14472307027782, 171.54848296512858, 172.44826274618714]
        # y = [3.4337664165497284, 2.842386063820232, 3.455781868600212, 2.145564892041854, 1.7676972935744644,
        #      1.8856400849715484, 3.0211594283408942, 2.939357030494847, 3.4629779675103975, 4.3773096085985586,
        #      3.328512398673311, 2.589828233001711, 3.6493859750563473, 2.681192606537522, 4.5753072352061395,
        #      4.242515704860111, 2.9175522703430374, 4.116013793500628, 3.059611533042144, 2.3151094109490344,
        #      4.436480748890446, 5.163854991321003, 3.83171922078553, 3.5299850719865162, 4.230159151058729,
        #      3.9804301567524476, 3.1727646802011407, 5.40038577291898, 4.701437283043987, 4.225439117952618,
        #      4.775031257836673, 5.3749910744971885, 4.237757025074217]

        X = [12.634125288405954, 19.592982663099633, 26.76186822194174, 28.201343143379717, 33.28632506548249,
             42.52995608240193, 43.36187024567479, 48.64118119643959, 57.33350331891519, 58.823049385477475,
             63.841642856663626, 72.40771806864932, 73.566153026606, 78.9981664786894, 86.60576537376394,
             88.33597417356685, 93.14916806494222, 100.8311384280346, 104.1811835349101, 108.60731753027753,
             117.57279146104945, 119.29277839501682, 124.32445613183096, 132.75261247547803, 133.16401491822188,
             138.56036146060143, 146.7412281982213, 148.13806603276765, 154.0380125546232, 161.95608134842143,
             163.2642782183215, 169.8339656159621, 177.72696791495696]
        y = [4.895714169065798, 5.819841235473765, 5.3380329853433786, 4.410828408944582, 5.249549824222811,
             6.77790300440882, 5.225539960372451, 6.04737145252988, 6.549671106598922, 5.633799069060132,
             7.738766361920511, 6.330343270789655, 5.085951348186597, 6.753044438718462, 5.8927811955058615,
             5.329695291371877, 8.046845518471073, 6.03113346447401, 6.8826233739358855, 8.00380865594424,
             8.814146187927832, 7.914782304450469, 9.843600067775014, 9.397582871971991, 7.2092942954097765,
             9.556607580538298, 9.285874828133615, 8.398195269144395, 9.960099204478126, 9.999565938477986,
             8.702457052566075, 10.546171081051568, 10.538236239591868]
        sb_pre = lasso_pre(X, y, a)

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

    fo = 'steel_fix_lr_1'
    z_range = (.15, .18)
    line_thresh = 0.0016
    line_size_thresh = 300

    uniform_bend_lr_fix(s_angle=0, e_angle=165, interval=15, fo=fo,
                        z_range=z_range, line_thresh=line_thresh, line_size_thresh=line_size_thresh)
    base.run()
