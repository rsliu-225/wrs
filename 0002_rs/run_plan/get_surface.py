import pickle
import config
import cv2
import utils.vision_utils as vu
import numpy as np

if __name__ == '__main__':
    phxi_info = pickle.load(open(config.ROOT + '\img\phoxi\skull.pkl', 'rb'))
    grayimg = phxi_info[0]
    depthimg = phxi_info[1]
    vu.extract_clr(grayimg, (50, 50), toggledebug=True)
