import os
import pickle
import cv2
import utils.pcd_utils as pcdu
import numpy as np
import config

transmat = np.asarray([[0.00320929, -1.00401041, 0.00128358, 0.31255359],
                       [-0.98255047, -0.00797362, 0.19879522, -0.16003892],
                       [-0.20321256, 0.00352566, -0.96782627, 0.95215224],
                       [0.0, 0.0, 0.0, 1.0]])


def loadalldata(folder_name, show=True):
    files = os.listdir(folder_name)
    alldata = []
    if not os.path.exists(os.path.join(folder_name, "grayimg/")):
        os.mkdir(os.path.join(folder_name, "grayimg/"))
    for file in files:
        if not os.path.isdir(file) and file[-4:] == ".pkl" and file[2:] != "bg":
            print(file)
            data = pickle.load(open(folder_name + file, "rb"))
            alldata.append(data)
            grayimg = data[0]
            depthimg = data[1]
            cv2.imwrite(os.path.join(folder_name, "grayimg/", f"{file[:-4]}.jpg"), grayimg)
            cv2.imwrite(os.path.join(folder_name, "depthimg/", f"{file[:-4]}.jpg"), depthimg)
            if show:
                cv2.imshow(str(file), depthimg)
                cv2.waitKey(0)
    return [img[0] for img in alldata], [img[1] for img in alldata], [img[2] for img in alldata]


def show_pcd(folder_name):
    _, _, pcdseq = loadalldata(folder_name, show=False)
    pcd = np.asarray(pcdseq[0]) / 1000
    pcd = pcdu.trans_pcd(pcd, transmat)
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    pcdu.show_pcd(pcd)
    base.run()


if __name__ == '__main__':
    import visualization.panda.world as wd

    loadalldata('./seq/plate/')
    # show_pcd('./')
