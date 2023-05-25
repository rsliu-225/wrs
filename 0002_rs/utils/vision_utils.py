import os
import pickle

import cv2
import numpy as np
import open3d as o3d

import config
from sklearn.cluster import DBSCAN
# from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import basis.robot_math as rm


# import sknw


def load_kntcalibmat(amat_path=os.path.join(config.ROOT, "./camcalib/data/"), f_name="knt_calibmat.pkl"):
    amat = pickle.load(open(amat_path + f_name, "rb"))
    return amat


def map_depth2pcd(depthnarray, pcd):
    pcdnarray = np.array(pcd)
    depthnarray = depthnarray.flatten().reshape((depthnarray.shape[0] * depthnarray.shape[1], 1))

    pcd_result = []
    for i in range(len(depthnarray)):
        if depthnarray[i] != 0:
            pcd_result.append(pcdnarray[i])
        else:
            pcd_result.append(np.array([0, 0, 0]))

    pcdnarray = np.array(pcd_result)

    return pcdnarray


def convert_depth2pcd(depthnarray):
    h, w = depthnarray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_, y_)
    z_ = depthnarray.flatten()
    pcd = np.zeros((np.size(mesh_x), 3))
    pcd[:, 0] = np.reshape(mesh_x, -1)
    pcd[:, 1] = np.reshape(mesh_y, -1)
    pcd[:, 2] = np.reshape(z_, -1)
    return np.delete(pcd, np.where(pcd[:, 2] == 0)[0], axis=0) / 1000


def convert_depth2pcd_o3d(depthnarray, intr_f_name="realsense_intr.pkl", toggledebug=False):
    intr = pickle.load(open(os.path.join(config.ROOT, "camcalib/data", intr_f_name), "rb"))
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"],
                                                                 intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    depthnarray = np.array(depthnarray, dtype=np.uint16)
    depthimg = o3d.geometry.Image(depthnarray)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic,
                                                          np.array(np.eye(4), dtype=np.float64))

    if toggledebug:
        o3d.visualization.draw_geometries([pcd])
        print(np.asarray(pcd.points))
    return np.asarray(pcd.points)


def map_gray2pcd(grayimg, pcd, zeros=False):
    pcdnarray = np.array(pcd)
    grayimg = grayimg.flatten().reshape((grayimg.shape[0] * grayimg.shape[1], 1))

    pcd_result = []
    for i in range(len(grayimg)):
        if grayimg[i] != 0:
            pcd_result.append(pcdnarray[i])
        else:
            if zeros:
                pcd_result.append(np.array([0, 0, 0]))

    return np.asarray(pcd_result)


def mask2gray(mask, grayimg):
    grayimg[mask == 0] = 0
    return grayimg


def pcd2gray(pcd, grayimg):
    return NotImplemented


def map_depth2gray(depthnarray, greyimg):
    greyimg[depthnarray == 0] = 0
    return greyimg


def map_grayp2pcdp(grayp, grayimg, pcd):
    return np.array([pcd[int(grayp[1] * grayimg.shape[1] + grayp[0])]])


def map_pcdpinx2graypinx(pcdpinx, grayimg):
    a, b = divmod(pcdpinx, grayimg.shape[1])
    return b, a + 1


def gray23channel(grayimg):
    return np.stack((grayimg,) * 3, axis=-1)


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb2binary(img, threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)[1]


def binary2pts(binary):
    shape = binary.shape
    p_list = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if binary[i, j]:
                p_list.append([i, j])
    return p_list


def mask2pts(mask):
    p_list = []
    for i, row in enumerate(mask):
        for j, val in enumerate(row):
            if val > 0:
                p_list.append((i, j))
    return np.asarray(p_list)


def pts2mask(pts, shape):
    pts = np.asarray(pts).astype(int)
    mask = np.zeros(shape)
    for p in pts:
        mask[p[0], p[1]] = 1
    return mask


def get_max_cluster(pts, eps=6, min_samples=20):
    pts_narray = np.asarray(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    res = []
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
    return res


def extract_clr_gray(grayimg, clr, shape=(772, 1032, 1), crop_xy=((250, 400), (400, 650)), toggledebug=False,
                     erode=False):
    mask = np.where((grayimg > clr[0]) & (grayimg < clr[1]), 1, 0)
    if crop_xy is not None:
        crop_mask = np.zeros(shape)
        crop_mask[crop_xy[0][0]:crop_xy[0][1], crop_xy[0][0]:crop_xy[0][1]] = 1
        mask = mask * crop_mask
    diff_p_narray = mask2pts(mask)
    stroke_pts = np.array(get_max_cluster(diff_p_narray, eps=3, min_samples=20))
    clr_mask = pts2mask(stroke_pts, shape)
    if erode:
        kernel = np.ones((2, 2), np.uint8)
        clr_mask = cv2.erode(clr_mask, kernel, iterations=1)
    # print(clr_mask)

    if toggledebug:
        # cv2.imshow('diff', diff)
        # cv2.waitKey(0)
        cv2.imshow('cropped mask', mask)
        cv2.waitKey(0)
        cv2.imshow('clustered mask', clr_mask)
        cv2.waitKey(0)

    return clr_mask


def crop(crop_xy, img):
    img = np.asarray(img)
    crop_mask = np.zeros(img.shape)
    print(img.shape)
    crop_mask[crop_xy[0][0]:crop_xy[0][1], crop_xy[1][0]:crop_xy[1][1]] = 1
    return img * crop_mask


def extract_label_rgb(rgbimg, shape=(772, 1032, 1), crop_xy=None, toggledebug=False, erode=False):
    rgbimg_std = np.std(rgbimg, axis=2)
    mask = np.where((rgbimg_std > 50), 1, 0)
    mask = mask.reshape(shape)
    if crop_xy is not None:
        crop_mask = np.zeros(shape)
        crop_mask[crop_xy[0][0]:crop_xy[0][1], crop_xy[1][0]:crop_xy[1][1]] = 1
        mask = mask * crop_mask
    if erode:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.astype(np.float)

    if toggledebug:
        cv2.imshow('mask', mask)
        cv2.waitKey(0)

    return mask


def mask2skmask(mask, inp=5, shape=(772, 1032, 1), toggledebug=False):
    skeleton = skeletonize(mask)
    graph = sknw.build_sknw(skeleton, multi=True)
    pts_all = []

    # draw edges by pts
    for (s, e) in graph.edges():
        for cnt in range(10):
            try:
                pts = graph[s][e][cnt]['pts']
                plt.plot(pts[:, 1], pts[:, 0], 'g')
                for i in range(len(pts)):
                    if i % inp == 0:
                        pts_all.append(pts[i])
                        plt.plot([pts[i, 1]], [pts[i, 0]], 'b.')
            except:
                break

    nodes = graph.nodes()
    pts = np.array([nodes[i]['o'] for i in nodes])
    pts_all.extend(pts)
    if toggledebug:
        plt.imshow(skeleton, cmap='gray')
        plt.plot(pts[:, 1], pts[:, 0], 'r.')
        plt.title('Build Graph')
        plt.show()
    return pts2mask(pts_all, shape=shape)


def enhance_grayimg(grayimg):
    if len(grayimg.shape) == 2 or grayimg.shape[2] == 1:
        grayimg = grayimg.reshape(grayimg.shape[:2])
    return cv2.equalizeHist(grayimg)


def get_corners_aruco(img):
    import cv2.aruco as aruco
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if ids is None:
        return corners, ids
    ids = [v[0] for v in ids]
    # cv2.imshow('', img)
    # cv2.waitKey(0)
    return corners, ids


# def get_axis_aruco(img, pcd, id=1):
#     corners, ids = get_corners_aruco(img)
#     pts = []
#     for i, corner in enumerate(corners[ids.index(id)][0]):
#         p = np.asarray(map_grayp2pcdp(corner, img, pcd))[0]
#         if all(np.equal(p, np.asarray([0, 0, 0]))):
#             break
#         pts.append(p)
#     pts = np.asarray(pts)
#     if len(pts) < 4:
#         return None
#     x = pts[0] - pts[1]
#     y = pts[2] - pts[1]
#     z = np.cross(x, y)
#     rot = np.asarray([rm.unit_vector(x), rm.unit_vector(y), rm.unit_vector(z)])
#     return rm.homomat_from_posrot(np.mean(pts, axis=0), rot)


def get_axis_aruco(img, pcd):
    import modeling.geometric_model as gm
    corners, ids = get_corners_aruco(img)
    pts_dict = {}

    for i, corners in enumerate(corners):
        id = ids[i]
        pts_tmp = []
        for corner in corners[0]:
            p = np.asarray(map_grayp2pcdp(corner, img, pcd))[0]
            if all(np.equal(p, np.asarray([0, 0, 0]))):
                break
            pts_tmp.append(p)
            gm.gen_sphere(p, radius=.002).attach_to(base)
        pts_dict[id] = np.asarray(pts_tmp)
    pts = []
    for v in pts_dict.values():
        pts.extend(v)
    pts = np.asarray(pts)
    if len(pts) < 4 * 6:
        return None

    pcv, pcaxmat = rm.compute_pca(pts)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x_v = pcaxmat[:, inx[2]]
    y_v = pcaxmat[:, inx[1]]
    z_v = pcaxmat[:, inx[0]]
    x = np.mean(pts_dict[3], axis=0) - np.mean(pts_dict[1], axis=0)
    if rm.angle_between_vectors(x, x_v) > np.pi / 2:
        x_v = -x_v
    y = np.mean(pts_dict[2], axis=0) - np.mean(pts_dict[1], axis=0)
    if rm.angle_between_vectors(y, y_v) > np.pi / 2:
        y_v = -y_v
    if rm.angle_between_vectors(np.asarray([0, 0, 1]), z_v) > np.pi / 2:
        z_v = -z_v
    pcaxmat = np.asarray([x_v, y_v, z_v]).T
    return rm.homomat_from_posrot(np.mean(pts, axis=0), pcaxmat)
