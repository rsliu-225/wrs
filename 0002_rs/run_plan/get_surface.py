import pickle
import config
import cv2
import utils.vision_utils as vu
import numpy as np
import utils.pcd_utils as pcdu
import utils.phoxi_locator as pl

if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    x_range = (.2, .6)
    y_range = (-.5, .5)
    z_range = (.05, .2)
    phxi_info = pickle.load(open(config.ROOT + '\img\phoxi\skull.pkl', 'rb'))
    grayimg = phxi_info[0]
    depthimg = phxi_info[1]
    pcd = pcdu.trans_pcd(phxi_info[2] / 1000,
                         transmat=pickle.load(open(config.ROOT + '/camcalib/data/phoxi_calibmat_yumi.pkl', 'rb')))
    mask = vu.extract_clr(grayimg, (10, 255), toggledebug=False)
    cv2.imshow('grayimg', grayimg)
    cv2.waitKey(0)
    grayimg = grayimg * mask
    cv2.imshow('grayimg2', grayimg)
    cv2.waitKey(0)
    pcd = vu.map_gray2pcd(grayimg, pcd)

    # pcd = pcdu.crop_pcd(pcd, x_range, y_range, z_range)
    pcdu.show_pcd(pcd)
    base.run()
