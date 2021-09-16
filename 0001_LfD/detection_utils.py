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
import open3d as o3d

from localenv import envloader as el
import visualization.panda.world as wd
import utils.drawpath_utils as du
import utils.pcd_utils as pcdu
import local_vis.realsense.realsense as rs
import utils.vision_utils as vu
import basis.robot_math as rm
import basis.o3dhelper as o3dhelper
from mask_rcnn_seg.inference import MaskRcnnPredictor
import config_LfD as config


def load_frame_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'raw_img/rs/seq/')):
    depthimg_list = []
    rgbimg_list = []
    for f in sorted(os.listdir(os.path.join(root_path, folder_name))):
        if f[-3:] != 'pkl':
            continue
        tmp = pickle.load(open(os.path.join(root_path, folder_name, f), 'rb'))
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            rgbimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            rgbimg_list.append(tmp[1])
    return [depthimg_list, rgbimg_list]


def load_frame(f_name, root_path=os.path.join(config.DATA_PATH, 'rs/sgl/')):
    return pickle.load(open(os.path.join(root_path, f_name), 'rb'))


def load_o3dpcd_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'seg_pcd/')):
    res_list = []
    for f in sorted(os.listdir(os.path.join(root_path, folder_name))):
        if f[-3:] != 'pcd':
            continue
        res_list.append(o3d.io.read_point_cloud(os.path.join(root_path, folder_name, f)))
    return res_list


def get_depth_diff(depth_img_bg, depth_img, threshold=2):
    diff = depth_img.astype(int) - depth_img_bg.astype(int)
    mask = np.abs(diff) > threshold
    return depth_img * mask


def flist_remove_bg(depthimg_list, rgb_img_list, threshold=10, bg_index=0, toggledebug=False, find_same=True):
    # depth_bg_scaled = scale_depth_img(depthimg_list[bg_index])
    depth_bg = depthimg_list[bg_index]
    mask_list = []
    if toggledebug:
        realsense = rs.RealSense()
        depthimg_bg = depthimg_list[bg_index]
        pcd_bg = realsense.depth2pcd(depthimg_bg)
        pcdu.show_pcd(pcd_bg, rgba=(1, 1, 1, .1))
        base.run()

    for inx, depthimg in enumerate(depthimg_list):
        if inx == bg_index:
            continue
        # depthimg_scaled = scale_depth_img(depthimg)
        diff = depthimg.astype(int) - depth_bg.astype(int)
        if find_same:
            mask = np.abs(diff) < threshold
        else:
            mask = np.abs(diff) > threshold
        mask_list.append(mask)
        if toggledebug:
            # cv2.imshow("", mask.astype(np.uint8)*255)
            # cv2.waitKey(0)
            mask_show = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            rgb_img = rgb_img_list[inx + 1] * mask_show
            show_img_hstack(rgb_img, vu.gray23channel(depthimg))

    return mask_list


def show_rgb_diff_list(depth_img_list, rgb_img_list, threshold=20, toggledebug=False):
    stroke_mask = get_strokes_mask(rgb_img_list[0], rgb_img_list[-1], threshold=threshold, toggledebug=toggledebug)
    rgbimg_bg = rgb_img_list[0]
    grayimg_bg = vu.rgb2gray(rgbimg_bg)
    depth_mask_list = flist_remove_bg(depth_img_list, rgb_img_list)
    for inx, rgb_img in enumerate(rgb_img_list[1:]):
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


def max_pooling_2d(depth_img_list, rgb_img_list):
    depth_bg = depth_img_list[0].astype(float)

    for inx, depth_img in enumerate(depth_img_list[1:]):
        depth_img = depth_img.astype(float)
        depth_bg_pooled = F.max_pooling_2d(depth_bg[np.newaxis, np.newaxis, :, :], ksize=(4, 4))
        depth_img_pooled = F.max_pooling_2d(depth_img[np.newaxis, np.newaxis, :, :], ksize=(4, 4))
        diff_pooled = depth_img_pooled.data - depth_bg_pooled.data
        print(diff_pooled.shape)

        diff = F.unpooling_2d(diff_pooled, ksize=(3, 3))
        print(diff.shape)
        mask = np.abs(diff) < 20
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        rgb_img = rgb_img_list[inx + 1] * mask
        cv2.imshow("", rgb_img)
        cv2.waitKey(0)


def scale_depth_img(depthimg, scale_max=255.0, range=(1, 1000)):
    mask = np.logical_or(depthimg < range[0], depthimg > range[1])
    # print(depth_img.min(), depthimg.max())
    scaled_depthimg = depthimg
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


def get_closest_cluster(pts, seed, eps=.003, min_pts=50):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=2).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []
    min_dist = .2

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
    print('dist:',min_dist)

    return res, mask


def cluster_dbscan(pts, eps=0.003):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=2).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
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
        cluster = pts_narray[class_member_mask & core_samples_mask]
        cluster_list.append(cluster)
        ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], s=1)
    plt.show()

    return cluster_list


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


def get_component_by_seed(depthnparray_float32, seed=[0, 0], epdelta=.005, minsize=300):
    """
    finds the next connected components whose area is larger than minsize
    region grow using the given seed
    returns one nparray_float32

    :param depthnparray_float32:
    :param epdelta:
    :param minsize:
    :return:
    """

    depthnparray_float32_cp = copy.deepcopy(depthnparray_float32)
    while True:
        tmpidx = np.nonzero(depthnparray_float32_cp)
        if len(tmpidx[0]) == 0:
            return np.zeros_like(depthnparray_float32)
        tmpcomponent = __region_growing(depthnparray_float32_cp, seed, epdelta)
        depthnparray_float32_cp[tmpcomponent != 0] = 0
        if np.count_nonzero(tmpcomponent) > minsize:
            return tmpcomponent
        else:
            continue


def __region_growing(depthimg, seed, epdelta):
    print(depthimg)
    cv2.imshow("", depthimg)
    cv2.waitKey(0)
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
                # if not coord in processed:
                #     list.append(coord)
                # processed.append(coord)
        list.pop(0)
        cv2.imshow("progress", outimg)
        cv2.waitKey(0)
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


if __name__ == '__main__':
    import geojson
    # from shapely.geometry.polygon import LinearRing
    import config
    from tqdm import tqdm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])

    realsense = rs.RealSense()

    depthimglist, rgbimg_list = realsense.load_frame_seq("osaka")
    depthimg_bg = depthimglist[5]
    pcd_bg = vu.convert_depth2pcd_o3d(depthimg_bg)
    pcdu.show_pcd(pcd_bg, rgba=(1, 1, 1, 1))
    normal, d = pcdu.get_plane(pcd_bg, dist_threshold=2, toggledebug=True)
    # print(normal, d)
    # gm.gen_box()
    # base.run()
    # depthimg = get_depth_diff(depthimg_list[0], depthimg_list[5])

    predictor = MaskRcnnPredictor()
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
