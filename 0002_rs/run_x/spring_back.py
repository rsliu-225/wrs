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
fo = 'steel'
_, depth_img, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"init.pkl"))
cv2.imshow("depth_img", depth_img)
cv2.waitKey(0)

for a in range(0, 151, 15):
    print(a)
    if a == 0:
        motor.rot_degree(clockwise=0, rot_deg=15)
        _, depth_img, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
        cv2.imshow("depth_img", depth_img)
        cv2.waitKey(0)

        motor.rot_degree(clockwise=1, rot_deg=15)
        _, depth_img, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
        cv2.imshow("depth_img", depth_img)
        cv2.waitKey(0)
    else:
        motor.rot_degree(clockwise=0, rot_deg=30)
        _, depth_img, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_goal.pkl"))
        cv2.imshow("depth_img", depth_img)
        cv2.waitKey(0)
        motor.rot_degree(clockwise=1, rot_deg=15)
        _, depth_img, _ = phxi.dumpalldata(f_name=os.path.join('img/phoxi/springback', fo, f"{str(a)}_res.pkl"))
        cv2.imshow("depth_img", depth_img)
        cv2.waitKey(0)
