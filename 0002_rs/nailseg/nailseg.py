import copy
import os
import pickle

import cv2
import numpy as np
import sklearn.cluster as skc

from localenv import envloader as el
import utils.pcd_utils as pcdu
import utils.phoxi as pu
import utils.vision_utils as vu
from nailseg.model_unet import FingerNailUNet

image_size = (784, 1040, 1)  # 图像长宽必须为16的倍数
org_image_size = (772, 1032, 1)


def padding(img, shape):
    pad_add = np.pad(img, [(int((shape[0] - img.shape[0]) / 2), int((shape[0] - img.shape[0]) / 2)),
                           (int((shape[1] - img.shape[1]) / 2), int((shape[1] - img.shape[1]) / 2))], 'constant')
    if len(pad_add.shape) != 3:
        pad_add = np.expand_dims(pad_add, 3)
    return pad_add


def remove_padding(img, shape):
    img = copy.deepcopy(img)
    org_shape = img.shape
    col_pad = int((org_shape[0] - shape[0]) / 2)
    row_pad = int((org_shape[1] - shape[1]) / 2)
    pad_removed = img[col_pad:-col_pad, row_pad:-row_pad]
    # pad_removed = img[:shape[0],:shape[1]]
    return pad_removed


def threshold_grey_scale_image(img, threshold=150, max_val=1):
    _, result = cv2.threshold(img, threshold, max_val, cv2.THRESH_BINARY)
    result = np.expand_dims(result, 3)
    return result


def save_result(img, result, img_name):
    img *= 255
    result *= 255
    img = np.concatenate((img, img, img), axis=-1).astype(np.uint8)
    result = np.concatenate((result, result, result), axis=-1).astype(np.uint8)

    for row in range(len(result)):
        for col in range(len(result[row])):
            if result[row][col][0] > 175:
                img[row][col][0] -= 0
                img[row][col][1] -= 150
                img[row][col][2] -= 150
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    cv2.imwrite(el.root + "/dataset/tst_result/" + str(img_name) + ".jpg", img)
    print(str(img_name), "image saved!")


def get_train_data(train_hand="", train_name="", mask_img_dir='/dataset/mask/', raw_img_dir='/dataset/raw/',
                   image_size=image_size):
    # load train set
    if train_hand != "":
        # sample file name: "a_lft_0_mask.jpg", id: "a_lft_0"
        train_ids = [x.split('_mask')[0] for x in os.listdir(el.root + mask_img_dir) if x.split('_')[1] == train_hand]
    elif train_name != "":
        train_ids = [x.split('_mask')[0] for x in os.listdir(el.root + mask_img_dir) if x.split('_')[0] == train_name]
    else:
        train_ids = [x.split('_mask')[0] for x in os.listdir(el.root + mask_img_dir)]

    X_train = [cv2.imread(el.root + raw_img_dir + img_id + '.jpg', cv2.IMREAD_GRAYSCALE) for img_id in train_ids]
    Y_train = [cv2.imread(el.root + mask_img_dir + img_id + '_mask.jpg', cv2.IMREAD_GRAYSCALE) for img_id in train_ids]

    # preprocess train set
    # padding images to target shape
    X_train = [padding(image, image_size) for image in X_train]
    Y_train = [padding(image, image_size) for image in Y_train]

    # threshold Y train to highlight only nails
    Y_train = [threshold_grey_scale_image(image) for image in Y_train]

    # concatenate train set
    X_train = np.array(X_train).astype('float32')
    Y_train = np.array(Y_train).astype('float32')

    # normalize X train to meet binary cross entropy requirement
    X_train /= 255

    return X_train, Y_train, train_ids


def get_test_data(train_ids=None, train_for_test=False, raw_img_dir='/dataset/raw/', test_pcd_dir='/dataset/pcd/',
                  image_size=image_size):
    ### Prepare train set and test set
    # load test set
    if train_ids is None:
        X_test_f_list = [f[:-4] for f in os.listdir(el.root + raw_img_dir) if f[-4:] == ".jpg"]
    else:
        if train_for_test:
            X_test_f_list = [f[:-4] for f in os.listdir(el.root + raw_img_dir) if
                             f.split('.')[0] in train_ids and f[-4:] == ".jpg"]
        else:
            X_test_f_list = [f[:-4] for f in os.listdir(el.root + raw_img_dir) if
                             f.split('.')[0] not in train_ids and f[-4:] == ".jpg"]

    X_test = [cv2.imread(el.root + raw_img_dir + f + ".jpg", cv2.IMREAD_GRAYSCALE) for f in X_test_f_list]

    # padding images to target shape
    X_test = [padding(image, image_size) for image in X_test]

    # normalize X test to meet binary cross entropy requirement
    X_test = np.array(X_test).astype('float32')
    X_test /= 255

    X_test_pcd = [pickle.load(open(el.root + test_pcd_dir + f + ".pkl", "rb")) for f in X_test_f_list]

    return X_test, X_test_pcd


def remove_noise(predict_list, x_list):
    result_list = []
    if len(predict_list) != len(x_list):
        print("Length of the input list is different!")
        return None
    for i in range(len(predict_list)):
        predict = predict_list[i]
        x = x_list[i]
        print(predict.shape)
        for row in range(predict.shape[0]):
            for col in range(predict.shape[1]):
                if x[row][col] == 0.0:
                    predict[row][col] = 0.0
        result_list.append(predict)

    return np.array(result_list)


def nailseg_pre_sgl(model_path, X_test_sgl, X_test_pcd_sgl, amat, org_image_size=org_image_size, image_size=image_size,
                    toggledebbug=False):
    model = FingerNailUNet(image_size)
    model.load_model(el.root + model_path)
    result_mask = model.predict(np.array([X_test_sgl]))[0]

    result_mask_nopadding = remove_padding(result_mask, org_image_size)
    X_test_sgl_nopadding = remove_padding(X_test_sgl, org_image_size)

    result_pcd = vu.map_gray2pcd(result_mask_nopadding, X_test_pcd_sgl)
    result_gray = vu.mask2gray(result_mask_nopadding, X_test_sgl_nopadding)
    save_result(X_test_sgl, result_mask, img_name="sgl")

    if toggledebbug:
        result_pcd = pcdu.trans_pcd(result_pcd, amat)
        pcdu.show_pcd(result_pcd)
        base.run()
    return result_gray, result_pcd


def nailseg_pre_all(model_path, X_test, X_test_pcd, org_image_size=org_image_size, image_size=image_size):
    model = FingerNailUNet(image_size)
    model.load_model(model_path)

    result_gray_list = model.predict(X_test)
    # result_gray_list = remove_noise(result_gray_list, X_test[:1])
    result_pcd_list = []
    for i in range(result_gray_list.shape[0]):
        result = remove_padding(result_gray_list[i], org_image_size)
        result_pcd = vu.map_gray2pcd(result, X_test_pcd[i])
        result_pcd_list.append(result_pcd)
        save_result(remove_padding(X_test[i], org_image_size), result, img_name=i)
    return result_gray_list, result_pcd_list


def nailseg_fit(X_train, Y_train, image_size=image_size):
    model = FingerNailUNet(image_size)
    model.fit(X_train, Y_train, batch_size=2, epochs=100)
    return model


def get_nail_list(pcd):
    db = skc.DBSCAN(eps=8, min_samples=50).fit(pcd)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    unique_labels = set(labels)
    nppcdlist = []
    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            temppartialpcd = pcd[class_member_mask & core_samples_mask]
            print(len(temppartialpcd))
            nppcdlist.append(temppartialpcd)
    return nppcdlist


if __name__ == '__main__':
    base, env = el.loadEnv_wrs()
    phxi_host = "10.2.0.60:18300"
    phxi = pu.Phoxi(host=phxi_host)
    amat = phxi.load_phoxicalibmat(f_name="phoxi_calibmat_0217.pkl")

    # Initialize model
    # image_size = (784, 1040, 1)  # 图像长宽必须为16的倍数
    # org_image_size = (772, 1032, 1)
    # model = FingerNailUNet(image_size)

    X_train, Y_train, train_ids = get_train_data(train_name="a")
    X_test, X_test_pcd = get_test_data(train_ids=train_ids, train_for_test=True)

    # nailseg_fit(X_train, Y_train)
    model_path = '/model/unet_a.h5'
    # result_gray_list, result_pcd_list = nailseg_pre_all(model_path, X_test, X_test_pcd)
    result_gray, result_pcd = nailseg_pre_sgl(model_path, X_test[0], X_test_pcd[0], amat)
    result_pcd = pcdu.trans_pcd(pcdu.remove_pcd_zeros(result_pcd), amat)
    nail_pcd_list = get_nail_list(result_pcd)
    color_list = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1)]
    for i in range(len(nail_pcd_list)):
        pcdu.show_pcd(nail_pcd_list[i], rgba=color_list[i])
    base.run()
