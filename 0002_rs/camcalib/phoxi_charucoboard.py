import robotcon.rpc.phoxi.phoxi_client as pclt
import pandaplotutils.pandactrl as pc
import cv2
import time
from vision.camcalib.calibrate import calibcharucoboard
from cv2 import aruco as aruco
import numpy as np
import yaml
import robotcon.ur3edual as ur3ex
import robotsim.ur3edual.ur3edual as ur3es
import robotsim.ur3edual.ur3edualmesh as ur3esm
import robotcon.ur3edual as ru
import manipulation.grip.robotiqhe.robotiqhe as rtqhe
import manipulation.grip.robotiqhe.robotiqhe_bigfinger as rtqhe_bf

import utiltools.robotmath as rm
import pickle
import wrssettingfree as wf

#   Take the pic for yaml file
if __name__ == '__main__':

    pxc = pclt.PhxClient(host="10.2.0.60:18300")

    for i in range(0, 100):
        pxc.triggerframe()
        img = pxc.gettextureimg()
        cv2.imwrite("phoxi/charuco/" + str(i) + ".png", img)
        cv2.imshow("tst", img)
        time.sleep(0.5)
        print(i)

    print("----------finished----------------")

    #   Get the yaml file
    calibcharucoboard(7, 5, arucomarkerdict=aruco.DICT_6X6_250, squaremarkersize=40, imgspath='./charuco/',
                      savename='phoxi_1.yaml')

    #   Initialize
    base = pc.World(camp=[5000, -2000, 2000], lookatp=[500, 0, 780], w=1024, h=768)
    pxc = pclt.PhxClient(host="10.2.0.60:18300")
