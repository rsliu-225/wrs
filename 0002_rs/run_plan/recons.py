import os
import pickle

import numpy as np
import open3d as o3d
from cv2 import aruco as aruco
from sklearn.cluster import DBSCAN

import basis.o3dhelper as o3dh
import basis.o3dhelper as o3h
import basis.robot_math as rm
import config
import modeling.geometric_model as gm
import utils.pcd_utils as pcdu
import utils.vision_utils as vu
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import sknw


def load_frame_seq(folder_name=None, root_path=os.path.join(config.ROOT, 'img/phoxi/seq/'), path=None):
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


def load_frame_seq_withf(folder_name=None, root_path=os.path.join(config.ROOT, 'img/phoxi/seq/'), path=None):
    if path is None:
        path = os.path.join(root_path, folder_name)
    depthimg_list = []
    rgbimg_list = []
    pcd_list = []
    fname_list = []
    for f in sorted(os.listdir(path)):
        if f[-3:] != 'pkl':
            continue
        fname_list.append(f[:-4])
        tmp = pickle.load(open(os.path.join(path, f), 'rb'))
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            rgbimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            rgbimg_list.append(tmp[1])
        if len(tmp) == 3:
            pcd_list.append(tmp[2])
    return [fname_list, depthimg_list, rgbimg_list, pcd_list]


def get_max_cluster(pts, eps=.003, min_samples=2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
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

    return np.asarray(res), mask


def get_nearest_cluster(pts, seed=(0, 0, 0), eps=.003, min_samples=2):
    pts_narray = np.array(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    # print("cluster:", unique_labels)
    res = []
    mask = []
    max_len = 0
    min_dist = 100
    gm.gen_sphere(seed, radius=.001).attach_to(base)

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            center = np.mean(cluster, axis=0)
            dist = np.linalg.norm(np.asarray(seed) - center)
            # if len(cluster) > max_len:
            #     max_len = len(cluster)
            #     res = cluster
            #     mask = [class_member_mask & core_samples_mask]
            if dist < min_dist:
                min_dist = dist
                res = cluster
                mask = [class_member_mask & core_samples_mask]

    return np.asarray(res), mask


def get_center_frame(corners, id, img, colors=None):
    ps = []
    if id == 1:
        seq = [1, 0, 0, 3]
        relpos = np.asarray([0, -.025, -.05124])
        relrot = np.eye(3)
    elif id == 2:
        seq = [1, 0, 0, 3]
        relpos = np.asarray([0, .025, -.05124])
        relrot = np.eye(3)
    elif id == 3:
        seq = [2, 3, 0, 3]
        relpos = np.asarray([0, -.025, -.03776])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi)
    elif id == 4:
        seq = [2, 3, 0, 3]
        relpos = np.asarray([0, .025, -.03776])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi)
    elif id == 5:
        seq = [0, 3, 3, 2]
        relpos = np.asarray([0, 0, -.072])
        relrot = rm.rotmat_from_axangle((1, 0, 0), -np.pi / 2)
    else:
        seq = [1, 2, 3, 2]
        relpos = np.asarray([0, 0, -.072])
        relrot = rm.rotmat_from_axangle((1, 0, 0), np.pi / 2)
    for i, corner in enumerate(corners[0]):
        p = np.asarray(vu.map_grayp2pcdp(corner, img, pcd))[0]
        if all(np.equal(p, np.asarray([0, 0, 0]))):
            break
        ps.append(p)
        # gm.gen_sphere(p, radius=.005, rgba=(1, 0, 0, i * .25)).attach_to(base)
    if len(ps) == 4:
        center = np.mean(np.asarray(ps), axis=0)
        x = rm.unit_vector(ps[seq[0]] - ps[seq[1]])
        y = rm.unit_vector(ps[seq[2]] - ps[seq[3]])
        z = rm.unit_vector(np.cross(x, y))
        rotmat = np.asarray([x, y, z]).T
        marker_mat4 = rm.homomat_from_posrot(center, rotmat)
        relmat4 = rm.homomat_from_posrot(relpos, relrot)
        origin_mat4 = np.dot(marker_mat4, relmat4)

        gm.gen_frame(np.linalg.inv(relmat4)[:3, 3], np.linalg.inv(relmat4)[:3, :3], thickness=.005,
                     length=.05, rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        if colors is not None:
            gm.gen_sphere(np.linalg.inv(relmat4)[:3, 3], rgba=colors[id], radius=.007).attach_to(base)
        # gm.gen_frame(origin_mat4[:3, 3], origin_mat4[:3, :3]).attach_to(base)
        # gm.gen_frame(marker_mat4[:3, 3], marker_mat4[:3, :3]).attach_to(base)
        return origin_mat4
    else:
        return None


def crop_maker(img, pcd):
    # tgt_id = 1
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if ids is None:
        return None, None
    ids = [v[0] for v in ids]
    print(ids)
    # if tgt_id not in ids:
    #     return None, None
    # pcdu.show_pcd(pcd, rgba=(1, 1, 1, 1))
    # gripperframe = get_center_frame(corners[ids.index(tgt_id)], tgt_id, img)
    gripperframe = get_center_frame(corners[0], ids[0], img)
    if gripperframe is None:
        return None, None
    # gm.gen_frame(pos=center, rotmat=rotmat).attach_to(base)
    pcd_trans = pcdu.trans_pcd(pcd, np.linalg.inv(gripperframe))
    pcdu.show_pcd(pcd_trans, rgba=(1, 1, 1, .1))
    gm.gen_frame().attach_to(base)
    # base.run()
    return ids[0], pcdu.crop_pcd(pcd_trans, x_range=(.05, .215), y_range=(-.4, .4), z_range=(-.2, -.0155))
    # return ids[0], pcdu.crop_pcd(pcd_trans, x_range=(.08, .215), y_range=(-.4, .4), z_range=(.05, .3))


def make_3dax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


def skeleton(pcd):
    pcd = o3dh.nparray2o3dpcd(np.asarray(pcd))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=.001)
    voxel_bin = np.zeros([100, 100, 100])
    for v in voxel_grid.get_voxels():
        voxel_bin[v.grid_index[0]][v.grid_index[1]][v.grid_index[2]] = 1
    ax = make_3dax(True)
    ax.voxels(voxel_bin, shade=False)
    skeleton = skeletonize(voxel_bin)
    graph = sknw.build_sknw(skeleton, multi=True)

    exit_node = list(set([s for s, e in graph.edges()] + [e for s, e in graph.edges()]))
    nodes = graph.nodes()
    stroke_list = [[nodes[i]['o'][::-1]] for i in nodes if i not in exit_node]

    for (s, e) in graph.edges():
        for cnt in range(10):
            stroke = []
            try:
                ps = graph[s][e][cnt]['pts']
                for i in range(len(ps)):
                    if i % 3 == 0:
                        stroke.append([ps[i][0], ps[i][1], ps[i][2]])
                # stroke.append([ps[-1, 1], ps[-1, 0]])
                stroke_list.append(stroke)
                print(stroke)
            except:
                break

    for stroke in np.asarray(stroke_list):
        stroke = np.asarray(stroke)
        ax.scatter(stroke[:, 0], stroke[:, 1], stroke[:, 2])
    plt.show()


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    icp = False
    folder_name = 'plate_a_cubic_2'
    if not os.path.exists(os.path.join(config.ROOT, 'recons_data', folder_name)):
        os.mkdir(os.path.join(config.ROOT, 'recons_data', folder_name))

    fnlist, grayimg_list, depthimg_list, pcd_list = load_frame_seq_withf(folder_name=folder_name)
    pcd_cropped_list = []
    inx_list = []
    trans = np.eye(4)
    colors = [(1, 0, 0, 1), (1, 1, 0, 1), (1, 0, 1, 1),
              (0, 1, 0, 1), (0, 1, 1, 1), (0, 0, 1, 1)]

    seed = (.116, 0, -.1)
    center = (.116, 0, -.0155)
    gm.gen_frame(center, np.eye(3)).attach_to(base)
    for i in range(len(grayimg_list)):
        pcd = np.asarray(pcd_list[i]) / 1000
        inx, pcd_cropped = crop_maker(grayimg_list[i], pcd)
        if pcd_cropped is not None:
            pcd_cropped, _ = get_nearest_cluster(pcd_cropped, seed=seed, eps=.01, min_samples=200)
            seed = np.mean(pcd_cropped, axis=0)
            print(len(pcd_cropped))
            # skeleton(pcd_cropped)
            if len(pcd_cropped) > 0:
                pcd_cropped = pcd_cropped - np.asarray(center)
                o3dpcd = o3dh.nparray2o3dpcd(pcd_cropped)
                # cl, ind = o3dpcd.remove_radius_outlier(nb_points=16, radius=0.005)
                # display_inlier_outlier(o3dpcd, ind)
                o3d.io.write_point_cloud(os.path.join(config.ROOT, 'recons_data', folder_name, f'{fnlist[i]}' + '.pcd'),
                                         o3dpcd)
                pcd_cropped_list.append(pcd_cropped)
                inx_list.append(inx)

    for i in range(1, len(pcd_cropped_list)):
        print(len(pcd_cropped_list[i - 1]))
        if icp:
            _, _, trans_tmp = o3h.registration_ptpt(pcd_cropped_list[i], pcd_cropped_list[i - 1],
                                                    downsampling_voxelsize=.005,
                                                    toggledebug=True)
            trans = trans_tmp.dot(trans)
            print(trans)
            pcdu.show_pcd(pcdu.trans_pcd(pcd_cropped_list[i], trans), rgba=colors[inx_list[i]])

        else:
            pcdu.show_pcd(pcd_cropped_list[i - 1], rgba=colors[inx_list[i] - 1])

    base.run()
