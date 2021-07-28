import os
import pickle
import cv2


def loadalldata(folder_name):
    files = os.listdir(folder_name)
    alldata = []
    for file in files:
        if not os.path.isdir(file) and file[-4:] == ".pkl" and file[2:] != "bg":
            data = pickle.load(open(folder_name + file, "rb"))
            greyimg = data[0]
            depthimg = data[1]
            # cv2.imwrite("greyimg/" + file[:-4] + ".jpg", greyimg)
            # cv2.imwrite("depthimg/" + file[:-4] + ".jpg", depthimg)
            cv2.imshow(str(file), greyimg)
            cv2.waitKey(0)
    return [img[0] for img in alldata], [img[1] for img in alldata], [img[2] for img in alldata]


if __name__ == '__main__':
    loadalldata('./')
