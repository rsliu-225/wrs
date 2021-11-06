import chainer.functions as F
import cv2
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import sknw
import pickle
from shapely.geometry import Polygon
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import open3d as o3d

from localenv import envloader as el
import visualization.panda.world as wd
import utils.drawpath_utils as du
import utils.pcd_utils as pcdu
import local_vis.realsense.realsense as rs
import utils.vision_utils as vu
import basis.robot_math as rm
import basis.o3dhelper as o3dhelper
from detectron.predictor import DetectronPredictor
import config_LfD as config
import shutil
from tqdm import tqdm

from sklearn.svm import OneClassSVM
from collections import Counter


def oneclasssvm_pcd(pcd_train, pcd_test):
    train = list(pcd_train.colors)
    test = list(pcd_test.colors)
    # clf = OneClassSVM(gamma='auto', nu=.6).fit(train)
    # print(len(train), len(test))
    # labels = clf.predict(test)
    db = DBSCAN(eps=3, min_samples=2).fit(train)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # score = clf.score_samples(train)
    clusters = []
    for label in set(labels):
        print(label)
        mask = (labels == label)
        clusters.append(np.asarray(pcd_train.points)[mask])
    plot_pts(clusters)
    print(Counter(labels))
    # pre = clf.predict(eval)
    # print(Counter(pre))


def load_frame_seq(folder_name=None, root_path=os.path.join(config.DATA_PATH, 'raw_img/rs/seq/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    depthimg_list = []
    rgbimg_list = []
    pcd_list = []
    for f in sorted(os.listdir(path)):
        if f[-3:] != 'pkl':
            continue
        tmp = pickle.load(open(os.path.join(path, f), 'rb'))
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            rgbimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            rgbimg_list.append(tmp[1])
        if len(tmp) == 3:
            pcd_list.append(tmp[2])
    return [depthimg_list, rgbimg_list, pcd_list]


def load_frame(f_name, root_path=os.path.join(config.DATA_PATH, 'rs/sgl/')):
    return pickle.load(open(os.path.join(root_path, f_name), 'rb'))


def load_o3dpcd_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'seg_pcd/')):
    res_list = []
    for f in sorted(os.listdir(os.path.join(root_path, folder_name))):
        if f[-3:] != 'pcd':
            continue
        res_list.append(o3d.io.read_point_cloud(os.path.join(root_path, folder_name, f)))
    return res_list


def load_mask(folder_name, mask_type, root_path=os.path.join(config.DATA_PATH, 'mask')):
    mask_list = []
    f_name_list = []
    path = os.path.join(root_path, folder_name, mask_type)
    for f in sorted(os.listdir(path)):
        if f[-3:] == 'png':
            f_name_list.append(int(f.split('.png')[0]))
            mask = cv2.imread(os.path.join(path, f))
            mask = mask[:, :, 0].reshape(mask.shape[0], mask.shape[1])
            mask[mask == 255] = 1
            mask_list.append(mask.astype(bool))
    return mask_list, f_name_list


def show_mask(mask, win_name=''):
    cv2.imshow(win_name, mask.astype(np.uint8) * 255)
    cv2.waitKey(0)


def get_list_inersection_by_id(ls, ids):
    id_intersec = ids[0]
    for i in range(len(ids) - 1):
        id_intersec = set(id_intersec).intersection(ids[i + 1])
    id_intersec = sorted(id_intersec)
    ls_new = []
    for i in range(len(ids)):
        l_new = []
        for id in id_intersec:
            # print(i, id, ids[i].index(id))
            l_new.append(ls[i][ids[i].index(id)])
        ls_new.append(l_new)
    return ls_new, list(id_intersec)


def get_depth_diff(depth_img_bg, depth_img, threshold=2):
    diff = depth_img.astype(int) - depth_img_bg.astype(int)
    mask = np.abs(diff) > threshold
    return depth_img * mask


def get_tgt_maskseq(rgbimg_list, folder_name=None, label=0, start_id=30, toggledebug=False, dilation=False):
    if folder_name is not None:
        create_path(f'{config.DATA_PATH}/inf_result/{folder_name}/')
        create_path(f'{config.DATA_PATH}/mask/{folder_name}/hand/')
    mask_list = []
    predictor = DetectronPredictor()

    for i, rgbimg in tqdm(enumerate(rgbimg_list)):
        mask = np.zeros((rgbimg.shape[:2]))
        if i < start_id:
            mask_list.append(mask)
            continue

        hnd_prediction = predictor.predict(rgbimg, label)
        hnd_mask_list = hnd_prediction.get('pred_masks').numpy()
        if len(hnd_mask_list) == 0:
            mask_list.append(mask)
            print('no target mask!')
            continue

        visualized_pred = predictor.visualize_prediction(rgbimg, hnd_prediction)
        for m in hnd_mask_list:
            if dilation:
                kernel = np.ones((3, 3), np.uint8)
                if toggledebug:
                    cv2.imshow('mask_org', m.astype(np.uint8) * 255)
                    cv2.waitKey(0)
                m = cv2.dilate(np.float32(m), kernel, iterations=5)
            mask = np.logical_or(mask, m)
        mask_list.append(mask)

        if folder_name is not None:
            cv2.imwrite(f'{config.DATA_PATH}/inf_result/{folder_name}/{str(i).zfill(4)}.png', visualized_pred)
            cv2.imwrite(f'{config.DATA_PATH}/mask/{folder_name}/hand/{str(i).zfill(4)}.png',
                        mask.astype(np.uint8) * 255)

    return mask_list


def get_bg_maskseq(depthimg_list, threshold=10, bg_inxs=range(20), folder_name=None, toggledebug=False, find_same=True):
    bg_list = np.asarray(depthimg_list)[bg_inxs]
    mask_list = []
    if folder_name is not None:
        create_path(f'{config.DATA_PATH}/bg/{folder_name}/')
        create_path(f'{config.DATA_PATH}/mask/{folder_name}/bg/')

    for inx, depthimg in enumerate(depthimg_list):
        if inx in bg_inxs:
            mask_list.append(np.ones((depthimg.shape)))
            continue
        mask = np.zeros((depthimg.shape))
        for bg in bg_list:
            diff = depthimg.astype(int) - bg.astype(int)
            if find_same:
                mask_tmp = np.abs(diff) < threshold
            else:
                mask_tmp = np.abs(diff) > threshold
            mask = np.logical_or(mask, mask_tmp)
        mask_list.append(mask)
        cv2.imwrite(f"{config.DATA_PATH}/mask/{folder_name}/bg/{str(inx).zfill(4)}.png", mask.astype(np.uint8) * 255)

    if toggledebug:
        for i, mask in enumerate(mask_list):
            cv2.imshow("", mask.astype(np.uint8) * 255)
            cv2.waitKey(0)

    return mask_list


def filter_by_maskseq(maskseq, depthimg_list, rgbimg_list, corr_id_list=None, exclude=True, folder_name=None,
                      toggledebug=False):
    print('mask seq length', len(maskseq))
    print('img seq length', len(depthimg_list))
    depthimg_list_res, rgbimg_list_res = [], []
    if folder_name is not None:
        create_path(f'{config.DATA_PATH}/filter_result/{folder_name}/')
    for i in range(len(maskseq)):
        mask = maskseq[i].astype(np.bool)
        if not exclude:
            mask = np.logical_not(mask)
        if corr_id_list is None:
            depthimg = copy.deepcopy(depthimg_list[i])
            rgbimg = copy.deepcopy(rgbimg_list[i])
        else:
            depthimg = copy.deepcopy(depthimg_list[corr_id_list[i]])
            rgbimg = copy.deepcopy(rgbimg_list[corr_id_list[i]])
        depthimg[mask] = 0
        rgbimg = rgbimg * np.repeat(~mask[:, :, np.newaxis], 3, axis=2)
        depthimg_list_res.append(depthimg)
        rgbimg_list_res.append(rgbimg)
        if toggledebug:
            cv2.imshow('mask', vu.gray23channel(np.asarray(mask).astype(np.uint8) * 255))
            cv2.imshow('depth', scale_depth_img(depthimg))
            cv2.imshow('rgb', rgbimg)
            cv2.waitKey(0)

        if folder_name is not None:
            pickle.dump([depthimg, rgbimg],
                        open(f'{config.DATA_PATH}/filter_result/{folder_name}/{str(i).zfill(4)}.pkl', 'wb'))

    return depthimg_list_res, rgbimg_list_res


def get_dyn_maskseq(rgbimg_list, toggledebug=False):
    mask_list = []

    frame1 = rgbimg_list[0]
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    for rgbimg in rgbimg_list:
        mask = np.zeros((depthimg.shape))
        next = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', rgbimg)
            cv2.imwrite('opticalhsv.png', bgr)
        prvs = next
        mask_list.append(mask)

    if toggledebug:
        for i, mask in enumerate(mask_list):
            print(i)
            cv2.imshow("", mask.astype(np.uint8) * 255)
            # cv2.imwrite(f"./bg_{inx}.png", mask.astype(np.uint8)*255)
            cv2.waitKey(0)

    return mask_list


def show_rgb_diff_list(depthimg_list, rgbimg_list, threshold=20, toggledebug=False):
    stroke_mask = get_strokes_mask(rgbimg_list[0], rgbimg_list[-1], threshold=threshold, toggledebug=toggledebug)
    rgbimg_bg = rgbimg_list[0]
    grayimg_bg = vu.rgb2gray(rgbimg_bg)
    depth_mask_list = get_bg_maskseq(depthimg_list)
    for inx, rgb_img in enumerate(rgbimg_list[1:]):
        diff_gray = np.abs(vu.rgb2gray(rgb_img).astype(int) - grayimg_bg.astype(int)).astype(np.uint8)
        current_mask = diff_gray > threshold
        mask = np.logical_and(stroke_mask, current_mask)

        depth_mask = depth_mask_list[inx]
        mask = np.logical_and(depth_mask, mask)

        gray = diff_gray * mask * 255
        show_img_hstack(rgb_img, vu.gray23channel(gray))


def unpooling_2d(x, ksize):
    kh, kw = ksize

    for i in range(kh - 1):
        x = x.repeat(2, axis=0)
    for j in range(kw - 1):
        x = x.repeat(2, axis=1)
    return x


def max_pooling_2d(depthimg_list, rgbimg_list):
    depth_bg = depthimg_list[0].astype(float)

    for inx, depth_img in enumerate(depthimg_list[1:]):
        depth_img = depth_img.astype(float)
        depth_bg_pooled = F.max_pooling_2d(depth_bg[np.newaxis, np.newaxis, :, :], ksize=(4, 4))
        depth_img_pooled = F.max_pooling_2d(depth_img[np.newaxis, np.newaxis, :, :], ksize=(4, 4))
        diff_pooled = depth_img_pooled.data - depth_bg_pooled.data
        print(diff_pooled.shape)

        diff = F.unpooling_2d(diff_pooled, ksize=(3, 3))
        print(diff.shape)
        mask = np.abs(diff) < 20
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        rgb_img = rgbimg_list[inx + 1] * mask
        cv2.imshow("", rgb_img)
        cv2.waitKey(0)


def scale_depth_img(depthimg, scale_max=255.0, range=(1, 1000)):
    mask = np.logical_or(depthimg < range[0], depthimg > range[1])
    # print(depth_img.min(), depthimg.max())
    scaled_depthimg = copy.deepcopy(depthimg)
    scaled_depthimg[mask] = 0
    scaled_depthimg[~mask] = np.round(scale_max * (depthimg[~mask] - range[0]) / (range[1] - range[0] - 1.0))
    scaled_depthimg = scaled_depthimg.astype(np.uint8)
    return scaled_depthimg


def show_img_hstack(img1, img2):
    numpy_horizontal = np.hstack((img1, img2))
    cv2.imshow('', numpy_horizontal)
    cv2.waitKey(0)


def skeleton2graph(binaryimg):
    graph = sknw.build_sknw(binaryimg, multi=True)
    # draw image
    plt.imshow(binaryimg, cmap='gray')
    # draw edges by pts
    for (s, e) in graph.edges():
        for cnt in range(10):
            try:
                ps = graph[s][e][cnt]['pts']
                plt.plot(ps[:, 1], ps[:, 0], 'green')
                for i in range(len(ps)):
                    if i % 10 == 0:
                        plt.plot([ps[i, 1]], [ps[i, 0]], 'b.')
            except:
                break

    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])
    plt.plot(ps[:, 1], ps[:, 0], 'r.')
    plt.title('Build Graph')
    plt.show()
    return graph


def graph2strokelist(graph, interval=5):
    exit_node = list(set([s for s, e in graph.edges()] + [e for s, e in graph.edges()]))
    nodes = graph.nodes()
    stroke_list = [[nodes[i]['o'][::-1]] for i in nodes if i not in exit_node]
    print(graph.edges())

    for (s, e) in graph.edges():
        for cnt in range(10):
            stroke = []
            try:
                ps = graph[s][e][cnt]['pts']
                for i in range(len(ps)):
                    if i % interval == 0:
                        stroke.append([ps[i, 1], ps[i, 0]])
                # stroke.append([ps[-1, 1], ps[-1, 0]])
                stroke_list.append(stroke)
            except:
                break
    return stroke_list


def get_strokes_mask(rgbimg1, rgbimg2, threshold=20, shape=(480, 640), toggledebug=False):
    gray1 = cv2.cvtColor(rgbimg1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rgbimg2, cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray1.astype(int) - gray2.astype(int)).astype(np.uint8)
    mask = diff > threshold

    crop_mask = np.zeros(shape)
    crop_mask[100:380, 240:500] = 1
    mask = mask * crop_mask

    diff_p_narray = mask2narray(mask)
    stroke_narray = np.array(get_max_cluster(diff_p_narray))
    stroke_mask = narray2mask(stroke_narray, shape)

    skeleton = skeletonize(stroke_mask)
    graph = skeleton2graph(skeleton)
    stroke_list = du.fit_drawpath_in_center_ms(graph2strokelist(graph))
    for stroke in stroke_list:
        du.draw_by_plist(stroke)

    if toggledebug:
        cv2.imshow("", (mask * 255).astype(np.uint8))
        cv2.waitKey(0)
        plt.scatter(diff_p_narray[:, 0], diff_p_narray[:, 1], c='green')
        plt.scatter(stroke_narray[:, 0], stroke_narray[:, 1])
        plt.show()

        # display results
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(stroke_mask)
        ax[0].axis('off')
        ax[0].set_title('original', fontsize=20)

        ax[1].imshow(skeleton)
        ax[1].axis('off')
        ax[1].set_title('skeleton', fontsize=20)

        fig.tight_layout()
        plt.show()

    return stroke_mask


def narray2mask(narray, shape):
    mask = np.zeros(shape)
    for p in narray:
        mask[p[0], p[1]] = 1
    return mask


def mask2narray(mask):
    p_list = []
    for i, row in enumerate(mask):
        for j, val in enumerate(row):
            if val > 0:
                p_list.append((i, j))
    return np.asarray(p_list)


def get_max_cluster(pts, eps=.003):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=2).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []
    max_len = 0

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            if len(cluster) > max_len:
                max_len = len(cluster)
                res = cluster
                mask = [class_member_mask & core_samples_mask]

    return res, mask


def get_closest_cluster(pts, seed, eps=.003, min_pts=50, min_dist=.2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=2).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            cluster_center = np.mean(cluster, axis=0)
            dist = np.linalg.norm(cluster_center - seed)
            if dist < min_dist and len(cluster) > min_pts:
                min_dist = dist
                res = cluster
                mask = [class_member_mask & core_samples_mask]
    print('dist:', min_dist)

    return res, mask


def plot_pts(pts_narray_list):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    for pts_narray in pts_narray_list:
        ax.scatter(pts_narray[:, 0], pts_narray[:, 1], pts_narray[:, 2], s=1)
    plt.show()


def cluster_dbscan(pts, eps=0.003, toggledebug=False):
    pts_narray = np.array(pts)

    db = DBSCAN(eps=eps, min_samples=2).fit(pts_narray)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    cluster_list = []

    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        cluster = pts_narray[class_member_mask & core_samples_mask]
        cluster_list.append(cluster)

    if toggledebug:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x", size=14)
        ax.set_ylabel("y", size=14)
        ax.set_zlabel("z", size=14)
        for cluster in cluster_list:
            ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=1)
        plt.show()

    return cluster_list


def cluster_meanshift(pts, bandwidth=.02, toggledebug=False):
    pts_narray = np.array(pts)
    ms = MeanShift(bandwidth=bandwidth).fit(pts_narray)
    cluster_centers = ms.cluster_centers_
    labels = ms.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    cluster_list = []
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x", size=14)
    ax.set_ylabel("y", size=14)
    ax.set_zlabel("z", size=14)
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        cluster = pts_narray[class_member_mask]
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=1)
        cluster_list.append(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2])
    if toggledebug:
        plt.show()

    return cluster_list, cluster_centers


def get_pose_icp(pcd, objcm, use_rmse=False, downsampling_voxelsize=1, toggledebug=False):
    source = pcdu.get_objpcd(objcm)
    # pos = pcdu.get_pcd_center(pcd)
    # inithomomat = rm.homobuild(pos, np.eye(3))
    # _, _, objmat4 = o3dhelper.registration_ptpt(source, pcd, downsampling_voxelsize=10, toggledebug=True)

    best_rmse = np.inf
    best_fitness = 0
    objmat4 = None
    for angle in range(0, 181, 60):
        rot = rm.rotmat_from_axangle((0, 0, 1), angle)
        inithomomat = rm.homomat_from_posrot((0, 0, 0), rot)
        # rmse, fitness, tempmat4 = o3dhelper.registration_icp_ptpt(source, pcd, inithomomat=inithomomat,
        #                                                           toggledebug=True)
        source = pcdu.trans_pcd(source, inithomomat)
        rmse, fitness, tempmat4 = \
            o3dhelper.registration_ptpt(source, pcd, downsampling_voxelsize, toggledebug=toggledebug)
        print(angle, rmse, fitness)
        if use_rmse:
            if rmse < best_rmse:
                objmat4 = np.dot(tempmat4, inithomomat)
                best_fitness = fitness
                best_rmse = rmse
        else:
            if fitness > best_fitness:
                objmat4 = np.dot(tempmat4, inithomomat)
                best_fitness = fitness
                best_rmse = rmse

    objcm.set_homomat(objmat4)
    objcm.set_rgba((1, 1, 0, 0.5))
    objcm.attach_to(base.render)

    return objmat4


def get_strokes_from_img(f_name):
    img = cv2.imread(config.ROOT + f_name)
    binaryimg = vu.rgb2binary(img)

    skeleton = skeletonize(binaryimg.astype(bool))
    pts = vu.binary2pts(skeleton)
    cluster_list = cluster_dbscan(pts)
    stroke_list = []

    for cluster in cluster_list:
        skeleton = narray2mask(np.array(cluster), shape=(480, 640))
        # plt.imshow(skeleton, cmap='gray')
        # plt.title('Skeleton')
        # plt.show()
        try:
            graph = skeleton2graph(skeleton)
            stroke_list.extend(graph2strokelist(graph))
        except:
            contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i, c in enumerate(contours):
                c = [p[0] for p in c]
                poly = geojson.dumps(Polygon(c).simplify(0.5))
                print(eval(poly))
                stroke_list.append(eval(poly)['coordinates'][0])
            # pts = vu.binary2pts(skeleton)
    stroke_list = du.fit_drawpath_in_center_ms(stroke_list)

    return stroke_list


def get_dp_components(depthimg, epdelta=5, minsize=100, toggledebug=False):
    """
    finds the next connected components whose area is larger than minsize
    region grow using the given seed
    returns one nparray_float32

    :param depthimg:
    :param epdelta:
    :param minsize:
    :return:
    """

    def __region_growing(depthimg, seed, epdelta):
        list = []
        outimg = np.zeros_like(depthimg).astype(dtype=np.uint8)
        list.append((seed[0], seed[1]))
        __processed = np.zeros_like(depthimg).astype(dtype=np.uint8)
        while len(list) > 0:
            pix = list[0]
            outimg[pix[0], pix[1]] = 255
            for coord in __get8n(pix[0], pix[1], depthimg.shape, __processed):
                newvalue = int(depthimg[coord[0], coord[1]][0])
                cmpvalue = int(depthimg[pix[0], pix[1]][0])
                if depthimg[coord[0], coord[1]] != 0 and abs(newvalue - cmpvalue) < epdelta:
                    outimg[coord[0], coord[1]] = 255
                    if __processed[coord[0], coord[1]] == 0:
                        list.append(coord)
                    __processed[coord[0], coord[1]] = 1
            list.pop(0)
            # cv2.imshow("progress", outimg)
            # cv2.waitKey(0)
        return outimg

    def __get8n(x, y, shape, processed):
        out = []
        maxx = shape[1] - 1
        maxy = shape[0] - 1
        # top left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # top center
        outx = x
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # top right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # left
        outx = min(max(x - 1, 0), maxx)
        outy = y
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # right
        outx = min(max(x + 1, 0), maxx)
        outy = y
        out.append((outx, outy))
        # bottom left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # bottom center
        outx = x
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # bottom right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))

        return out

    depthimg = np.array(depthimg, dtype=np.float32).reshape((depthimg.shape[0], depthimg.shape[1], 1))
    depthimg_cp = copy.deepcopy(depthimg)
    if toggledebug:
        cv2.imshow("input", depthimg)
        cv2.waitKey(0)
    components_list = []
    while True:
        tmpidx = np.nonzero(depthimg_cp)
        if len(tmpidx[0]) == 0:
            break
        seed = [tmpidx[0][0], tmpidx[1][0]]
        tmpcomponent = __region_growing(depthimg_cp, seed, epdelta)
        depthimg_cp[tmpcomponent != 0] = 0
        if np.count_nonzero(tmpcomponent) > minsize:
            components_list.append(tmpcomponent)
        else:
            continue
        if toggledebug:
            cv2.imshow("component", components_list[-1])
            cv2.waitKey(0)
            cv2.imshow("remain", depthimg_cp)
            cv2.waitKey(0)
    return components_list


def find_closest_dpcomponent(components_list, seed, dist_threshold=50):
    min_dist = np.inf
    result = None
    center = (0, 0)

    for component in components_list:
        cnts = cv2.findContours(component.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts[0]:
            (x, y), _ = cv2.minEnclosingCircle(c)
            tmp_center = (int(x), int(y))
            dist = np.linalg.norm(np.asarray(seed) - np.asarray(tmp_center))
            if dist < min_dist:
                min_dist = dist
                result = component
                center = tmp_center
    result = result.reshape(result.shape[:2])
    print(seed, min_dist)
    if min_dist > dist_threshold:
        print('skip')
        return None, seed
    return result, center


def find_largest_dpcomponent(components_list):
    max_radius = 0
    result = None
    center = (0, 0)

    for component in components_list:
        cnts = cv2.findContours(component.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts[0]:
            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)
            tmp_center = (int(x), int(y))
            if radius > max_radius:
                max_radius = radius
                result = component
                center = tmp_center
    result = result.reshape(result.shape[:2])
    return result, center


# def meanshift(frameseq):
#     # setup initial location of window
#     x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
#     track_window = (x, y, w, h)
#     # set up the ROI for tracking
#     roi = frameseq[0][y:y + h, x:x + w]
#     hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
#     roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
#     cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
#     # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#     term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
#     for frame in frameseq:
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
#         # apply meanshift to get the new location
#         ret, track_window = cv2.meanShift(dst, track_window, term_crit)
#         # Draw it on image
#         x, y, w, h = track_window
#         img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
#         cv2.imshow('img2', img2)
#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


if __name__ == '__main__':
    import geojson
    # from shapely.geometry.polygon import LinearRing
    import config
    from tqdm import tqdm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])

    realsense = rs.RealSense()

    depthimglist, rgbimg_list = load_frame_seq("osaka")
    depthimg_bg = depthimglist[5]
    pcd_bg = vu.convert_depth2pcd_o3d(depthimg_bg)
    pcdu.show_pcd(pcd_bg, rgba=(1, 1, 1, 1))
    normal, d = pcdu.get_plane(pcd_bg, dist_threshold=2, toggledebug=True)
    # print(normal, d)
    # gm.gen_box()
    # base.run()
    # depthimg = get_depth_diff(depthimg_list[0], depthimg_list[5])

    predictor = DetectronPredictor()
    label = 0
    stroke = []
    for i, im in tqdm(enumerate(rgbimg_list)):
        print(f"---------------image {i}---------------")
        mask = predictor.predict(im, label)
        mask = mask.get("pred_masks").numpy()
        if len(mask) == 0:
            print("no target mask!")
            continue
        depthimg = np.asarray(mask[0]) * depthimglist[i]
        pcd = vu.convert_depth2pcd_o3d(depthimg, toggledebug=False)
        if len(pcd) == 0:
            print("no pcd generated!")
            continue
        objmat4 = get_pose_icp(pcd, objcm=el.loadObj("pentip.stl"), toggledebug=False, use_rmse=False)
        stroke.append(objmat4[:3, 3])

    print(stroke)
    base.run()

    # get_depth_diff_list(depth_img_list, rgb_img_list, toggledebug=True)
    # show_rgb_diff_list(depth_img_list, rgb_img_list, toggledebug=True)

    stroke_list = get_strokes_from_img(config.ROOT + "/drawpath/img/pig.png")
    stroke_list = du.fit_drawpath_in_center_ms(stroke_list)
    print(stroke_list)
    for stroke in stroke_list:
        du.draw_by_plist(stroke)

    """
    meanshift
    """
    cap = cv2.VideoCapture('./slow_traffic_small.mp4')
    # take first frame of the video
    ret, frame = cap.read()
    frameseq = []
    while True:
        ret, frame = cap.read()
        if ret:
            frameseq.append(frame)
        else:
            break
    print(len(frameseq))
    meanshift(frameseq)
