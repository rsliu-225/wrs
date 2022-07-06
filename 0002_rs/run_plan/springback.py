import pickle
import os
import cv2
import config

fo = 'springback/alu'
f_name = '0.pkl'
textureimg, depthimg, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f_name), 'rb'))
cv2.imshow('', textureimg)
cv2.waitKey(0)
