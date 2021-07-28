import os
import pickle

import cv2

from localenv import envloader as el
import utils.phoxi as phoxi
import utils.vision_utils as vu
import utils.pcd_utils as pcdu
import utils.phoxi_locator as pl


def loadalldata(folder_name):
    files = os.listdir(folder_name)
    alldata = []
    f_name_list = []
    for file in files:
        if not os.path.isdir(file) and file[-4:] == ".pkl" and file[:2] != "bg":
            alldata.append(pickle.load(open(folder_name + file, "rb")))
            f_name_list.append(file[:-4])

    return [img[0] for img in alldata], [img[1] for img in alldata], [img[2] for img in alldata], f_name_list


def locate_hand_by_bouding_rect(grey_image):
    x, y, w, h = cv2.boundingRect(grey_image)
    cropped_grey_image = grey_image[y: y + h, x: x + w, :]
    return cropped_grey_image


def greyimg_resize(img, toggledebug=False):
    img_cut = locate_hand_by_bouding_rect(img)

    if toggledebug:
        cv2.imshow('img', img_cut)
        cv2.waitKey()

    return img_cut


if __name__ == '__main__':
    base, env = el.loadEnv_wrs()
    """
    读取图片 -> 灰度图片findhand -> 灰度图片把手切出来
    """
    greyimg_list, depthnparray_float32_list, pcd_list, f_name_list = loadalldata(el.root + "/img/hand/")

    phoxi_host = "10.2.0.199:18300"
    phxi = phoxi.Phoxi(host=phoxi_host)
    phxilocator = pl.PhxiLocator(phxi, amat_f_name="phoxi_calibmat_0117.pkl")
    for img_id in range(len(f_name_list)):
        if f_name_list[img_id][:1] == "a":
            workingarea_uint8 = phxilocator.remove_depth_bg(depthnparray_float32_list[img_id],
                                                            bg_f_name="bg_0.pkl", toggledebug=True)
            hand_depth = phxilocator.find_hand_icp(workingarea_uint8, pcd_list[img_id], toggledebug=False,
                                                   show_icp=False)

            if hand_depth is not None:
                hand_grey = vu.map_depth2gray(hand_depth, greyimg_list[img_id])
                hand_pcd = pcdu.trans_pcd(vu.map_depth2pcd(hand_depth, pcd_list[img_id]), phxilocator.amat)

                pcdu.show_pcd(hand_pcd)
                base.run()

                cv2.imwrite(el.root + "/dataset/raw/" + f_name_list[img_id] + ".jpg", hand_grey)
                pickle.dump(hand_pcd, open(el.root + "/dataset/pcd/" + f_name_list[img_id] + ".pkl", "wb"))
                print(f_name_list[img_id] + ".jpg", "saved!")
                print(f_name_list[img_id] + ".pkl", "saved!")

    '''
    get sample hand
    '''
    # for img_id in range(len(f_name_list)):
    #     sample_name_list = ["a_lft_0","a_rgt_0"]
    #     if f_name_list[img_id] in sample_name_list:
    #         workingarea_uint8 = phxilocator.find_workingarea(depthnparray_float32_list[img_id])
    #         hand_depth = phxilocator.find_hand(workingarea_uint8, toggledebug=False)
    #         if hand_depth is not None:
    #             hand_pcd = vu.get_realpcd(vu.depth2pcd(hand_depth, pcd_list[img_id]), phxilocator.amat)
    #             hand_grey = vu.depth2greyimg(hand_depth, greyimg_list[img_id])
    #             pickle.dump(hand_pcd,open(el.root + "/dataset/sample_handpcd/"+f_name_list[img_id]+"_pcd.pkl","wb"))
    #             print(f_name_list[img_id]+"_pcd.pkl", "saved!")
