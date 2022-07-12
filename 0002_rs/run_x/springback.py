import motorcontrol.Motor as motor
import time
import utils.phoxi as phoxi
import config
import os
import cv2

motor = motor.MotorNema23()
phxi = phoxi.Phoxi(host=config.PHOXI_HOST)


# motor.rot_degree(clockwise=1, rot_deg=1)
# motor.rot_degree(clockwise=0, rot_deg=1)


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


def uniform_bend_fb(s_angle, e_angle, interval, fo='steel'):
    _, depthimg, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"init.pkl"))
    cv2.imshow("depthimg", depthimg)
    cv2.waitKey(0)
    for a in range(s_angle, e_angle + 1, interval):
        print(a)
        if a == 0:
            motor.rot_degree(clockwise=0, rot_deg=interval)
            _, depthimg_goal, pcd_goal = \
                phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
            cv2.imshow("depthimg", depthimg_goal)
            cv2.waitKey(0)

            motor.rot_degree(clockwise=1, rot_deg=interval)
            _, depthimg_res, pcd_res = \
                phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
            cv2.imshow("depthimg", depthimg_res)
            cv2.waitKey(0)
        else:
            motor.rot_degree(clockwise=0, rot_deg=interval * 2)
            _, depthimg_goal, pcd_goal = phxi.dumpalldata(
                f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
            cv2.imshow("depthimg", depthimg_goal)
            cv2.waitKey(0)
            motor.rot_degree(clockwise=1, rot_deg=interval)
            _, depthimg_res, pcd_res = \
                phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
            cv2.imshow("depthimg", depthimg_res)
            cv2.waitKey(0)


if __name__ == '__main__':
    uniform_bend(s_angle=0, e_angle=150, interval=15, fo='steel')
