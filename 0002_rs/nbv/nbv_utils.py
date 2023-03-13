import json
import os
import random
import h5py

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from geomdl import BSpline
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree

import basis.robot_math as rm
import basis.o3dhelper as o3dh
import datagenerator.data_utils as du
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import bendplanner.bend_utils as bu
import pcn.inference as inference
import modeling.geometric_model as gm

COLOR = np.asarray(
    [[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14], [148, 103, 189], [23, 190, 207]]) / 255


def transpose(data, len=6):
    mat = [[] for _ in range(len)]
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            # if len(mat) > j:
            mat[j].append(v)
            # else:
            #     mat.append([v])
    return mat


def load_cov(path, cat_list, fo, cat_cnt_list, max_times=5, prefix='pcn'):
    cov_list = []
    max_list = []
    cnt_list = [0] * max_times
    for inx, cat in enumerate(cat_list):
        for f in os.listdir(os.path.join(path, cat, 'mesh')):
            print(f'-----------{f}------------')
            if int(f.split(".ply")[0]) >= cat_cnt_list[inx]:
                continue
            try:
                res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
            except:
                break
            # o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(res_dict['0']['input']))
            # o3d.visualization.draw_geometries([o3dpcd_i])
            cov_list_tmp = [res_dict['init_coverage']]
            max_tmp = [res_dict['init_coverage']]
            max = 0
            max_cnt = 0
            if res_dict['init_coverage'] > .94:
                print('remove')
            print(prefix, 'init', res_dict['init_coverage'])

            for i in range(max_times):
                if str(i) in res_dict.keys():
                    # try:
                    print(prefix, i + 1, res_dict[str(i)]['coverage'])
                    cov_list_tmp.append(res_dict[str(i)]['coverage'])
                    max = res_dict[str(i)]['coverage']
                    # except:
                    #     max_tmp.append(max)
                    #     continue
                    max_cnt = i
                max_tmp.append(max)
            cnt_list[max_cnt] += 1
            max_list.append(max_tmp)
            cov_list.append(cov_list_tmp)
    return transpose(cov_list, max_times + 1), transpose(max_list, max_times + 1), [cnt_list[0]] + cnt_list


def load_cov_w_fail(path, cat_list, fo, cat_cnt_list, max_times=5, prefix='pcn'):
    cov_list = []
    max_list = []
    cnt_list = [0] * max_times
    plan_fail_cnt = 0
    for inx, cat in enumerate(cat_list):
        for f in os.listdir(os.path.join(path, cat, 'mesh')):
            print(f'-----------{cat} {f}------------')
            if int(f.split(".ply")[0]) >= cat_cnt_list[inx]:
                continue
            try:
                res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
            except:
                break
            # o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(res_dict['0']['input']))
            # o3d.visualization.draw_geometries([o3dpcd_i])
            cov_list_tmp = [res_dict['init_coverage']]
            max_tmp = [res_dict['init_coverage']]
            max = 0
            max_cnt = 0
            if res_dict['init_coverage'] > .94:
                print('remove')
            print(prefix, 'init', res_dict['init_coverage'])

            for i in range(max_times):
                if str(i) in res_dict.keys():
                    if 'coverage' in res_dict[str(i)].keys():
                        print(prefix, i + 1, res_dict[str(i)]['coverage'])
                        cov_list_tmp.append(res_dict[str(i)]['coverage'])
                        max = res_dict[str(i)]['coverage']
                    else:
                        plan_fail_cnt += 1
                        print(prefix, i + 1, 'Planning failed')
                        max_tmp.append(max)
                        max_cnt = -1
                        continue
                    max_cnt = i
                max_tmp.append(max)
            if max_cnt != -1:
                cnt_list[max_cnt] += 1
            max_list.append(max_tmp)
            cov_list.append(cov_list_tmp)
    return transpose(cov_list, max_times + 1), transpose(max_list, max_times + 1), \
           [cnt_list[0]] + cnt_list, plan_fail_cnt


def fit(path, cat, fo, cross_sec, prefix='pcn', toggledebug=False):
    cd_list = []
    hd_list = []

    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        if int(f.split(".ply")[0]) > 10:
            continue
        print(f'-----------{f}------------')
        try:
            res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
            objcm_gt = du.o3dmesh2cm(o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f)))
        except:
            break
        cd_tmp = []
        hd_tmp = []
        stop = False
        for i in range(6):
            if str(i) in res_dict.keys():
                pts = np.asarray(res_dict[str(i)]['input'])
            elif not stop:
                pts = np.asarray(res_dict['final'])
                stop = True
            else:
                cd_tmp.append(cd_tmp[-1])
                hd_tmp.append(hd_tmp[-1])
                continue

            kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts, rgba=(1, 1, 0, 1), n_components=16)
            inp_pseq = nurbs_inp(kpts)
            inp_rotseq = pcdu.get_rots_wkpts(pts, inp_pseq, show=True, rgba=(1, 0, 0, 1))
            kpts = np.asarray(kpts)

            if toggledebug:
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='gray', s=.01, alpha=.5)
                ax.plot(kpts[:, 0], kpts[:, 1], kpts[:, 2])
                ax.plot(inp_pseq[:, 0], inp_pseq[:, 1], inp_pseq[:, 2])
                plt.show()

            objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008)
            cd = chamfer_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2', direction='bi')
            hd = hausdorff_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2')
            cd_tmp.append(cd * 1000)
            hd_tmp.append(hd * 1000)
        cd_list.append(cd_tmp)
        hd_list.append(hd_tmp)
        print(prefix, cd_list[-1], hd_list[-1])

    return transpose(cd_list), transpose(hd_list)


def fit_dist_cov(path, cat, fo, cross_sec, prefix='pcn', kpts_num=16, toggledebug=False):
    cd_list = []
    hd_list = []
    cov_list = []

    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        # if int(f.split(".ply")[0]) > 2:
        #     continue
        print(f'-----------{f}------------')
        try:
            res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
            objcm_gt = du.o3dmesh2cm(o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f)))
        except:
            break

        stop = False
        for i in range(6):
            if str(i) in res_dict.keys() and not stop:
                pts = np.asarray(res_dict[str(i)]['input'])
                if i == 0:
                    cov_list.append(res_dict['init_coverage'])
                else:
                    cov_list.append(res_dict[str(i - 1)]['coverage'])
            elif not stop:
                pts = np.asarray(res_dict['final'])
                cov_list.append(res_dict[str(i - 1)]['coverage'])
                stop = True
            else:
                break

            pts_o = inference.inference_sgl(np.asarray(pts))
            o3dpcd_o = o3dh.nparray2o3dpcd(pts_o)
            o3dpcd_o.estimate_normals()
            o3dpcd = o3dh.nparray2o3dpcd(pts)
            o3dpcd.estimate_normals()
            o3dpcd_o.paint_uniform_color(COLOR[2])
            o3dpcd.paint_uniform_color(COLOR[0])

            from sklearn.svm import OneClassSVM

            kpts_o, kpts_rotseq_o = pcdu.get_kpts_gmm(pts_o, rgba=(1, 1, 0, 1), n_components=kpts_num)
            # o3dpcd_final = None
            # for p in kpts_o:
            #     o3dpcd_kdt_o = o3d.geometry.KDTreeFlann(o3dpcd_o)
            #     _, idx, _ = o3dpcd_kdt_o.search_radius_vector_3d(p, 0.02)
            #     o3dpcd_tmp_o = o3dpcd_o.select_by_index(idx)
            #
            #     o3dpcd_kdt = o3d.geometry.KDTreeFlann(o3dpcd_o)
            #     _, idx, _ = o3dpcd_kdt.search_radius_vector_3d(p, 0.02)
            #     o3dpcd_tmp = o3dpcd.select_by_index(idx)
            #     o3d.visualization.draw_geometries([o3dpcd_tmp_o, o3dpcd_tmp])
            #
            #     clf = OneClassSVM(gamma='auto'). \
            #         fit(X=np.hstack((np.asarray(o3dpcd_tmp_o.points), np.asarray(o3dpcd_tmp_o.normals))),
            #             y=np.ones(len(np.asarray(o3dpcd_tmp_o.points))))
            #     labels = clf.predict(np.hstack((np.asarray(o3dpcd_tmp.points), np.asarray(o3dpcd_tmp.normals))))
            #
            #     o3dpcd_tmp = o3dpcd_tmp.select_by_index(np.where(labels == 1)[0])
            #     o3d.visualization.draw_geometries([o3dpcd_tmp_o, o3dpcd_tmp])
            #     if o3dpcd_final is None:
            #         o3dpcd_final = o3dpcd_tmp
            #     else:
            #         o3dpcd_final += o3dpcd_tmp
            #
            # o3dpcd_final.paint_uniform_color(COLOR[3])
            # o3d.visualization.draw_geometries([o3dpcd, o3dpcd_final])

            # kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts, rgba=(1, 1, 0, 1), means_init=kpts_o, n_components=kpts_num)
            kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts, rgba=(1, 1, 0, 1), means_init=None, n_components=kpts_num)

            inp_pseq = nurbs_inp(kpts)
            # inp_pseq = du.spl_inp(kpts)
            inp_rotseq = pcdu.get_rots_wkpts(pts, inp_pseq, show=True, rgba=(1, 0, 0, 1))
            kpts = np.asarray(kpts)

            if toggledebug:
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='gray', s=.02, alpha=.5)
                ax.plot(kpts[:, 0], kpts[:, 1], kpts[:, 2])
                ax.plot(kpts_o[:, 0], kpts_o[:, 1], kpts_o[:, 2])
                ax.plot(inp_pseq[:, 0], inp_pseq[:, 1], inp_pseq[:, 2])
                plt.show()

            objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008)
            cd = chamfer_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2', direction='bi')
            hd = hausdorff_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2')
            cd_list.append(cd * 1000)
            hd_list.append(hd * 1000)
        print(prefix, cd_list[-1], hd_list[-1], cov_list[-1])

    return cd_list, hd_list, cov_list


def plot_box(ax, data, clr, positions, showfliers=False):
    box = ax.boxplot(data, positions=positions, notch=True, patch_artist=True, showfliers=showfliers)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box[item], color=clr)
    # plt.setp(box["boxes"], facecolor=clr, alpha=.2)
    plt.setp(box["fliers"], markeredgecolor=clr, marker='.')


def nurbs_inp(kpts):
    curve = BSpline.Curve()
    degree = 3
    curve.degree = degree
    curve.ctrlpts = kpts.tolist()
    curve.knotvector = [0] * degree + np.linspace(0, 1, len(kpts) - degree + 1).tolist() + [1] * degree
    curve.delta = 0.01
    inp_pseq = np.asarray(curve.evalpts)

    return inp_pseq


def cal_avg(cnt_list):
    sum_v = 0
    for i in range(len(cnt_list)):
        sum_v += (i + 2) * cnt_list[i]
    return sum_v / sum(cnt_list)


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def hausdorff_distance(x, y, metric='l2'):
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    hausdorff_distance = max([np.max(min_y_to_x), np.max(min_x_to_y)])

    return hausdorff_distance


def gen_partial_o3dpcd(o3dmesh, rot=np.eye(3), trans=np.zeros(3), rot_center=(0, 0, 0), cam_mat4=np.eye(4),
                       vis_threshold=np.radians(75), fov=False, othermesh=[], w_otherpcd=False, toggledebug=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window('win', left=0, top=0)
    o3dmesh = o3dmesh.filter_smooth_taubin(number_of_iterations=10)
    # o3dmesh.rotate(rot, center=rot_center)
    # o3dmesh.translate(trans)
    o3dmesh.transform(rm.homomat_from_posrot(trans, rot))
    o3dmesh.transform(np.linalg.inv(cam_mat4))

    for mesh in othermesh:
        mesh.transform(np.linalg.inv(cam_mat4))
        vis.add_geometry(mesh)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    tmp_f_name = str(random.randint(0, 100))
    vis.capture_depth_point_cloud(f'./{tmp_f_name}.pcd', do_render=False, convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(f'./{tmp_f_name}.pcd')
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    o3dpcd_nrml = np.asarray(o3dpcd.normals)

    vis_idx = np.argwhere(np.arccos(abs(o3dpcd_nrml.dot(cam_mat4[:3, 2]))) < vis_threshold).flatten()
    o3dpcd = o3dpcd.select_by_index(vis_idx)
    o3dpcd.paint_uniform_color(COLOR[0])

    if toggledebug:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        o3d.visualization.draw_geometries([coord, o3dpcd, o3dmesh], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([o3dpcd], mesh_show_back_face=True)

    o3dpcd = o3dpcd.voxel_down_sample(voxel_size=.001)

    if not w_otherpcd:
        pcd_kdt = KDTree(np.asarray(o3dpcd.points))
        mesh_kdt = KDTree(np.asarray(o3dmesh.vertices))
        idxs = list(pcd_kdt.query_ball_tree(mesh_kdt, r=0.006))
        selected_idx = np.asarray([i for i in range(len(idxs)) if len(idxs[i]) > 0])
        o3dpcd = o3dpcd.select_by_index(selected_idx)

    if fov:
        o3dpcd = filer_pcd_by_cam(o3dpcd, cam_mat4[:3, 3], cam_mat4[:3, 2], dist=1.5, angle=np.pi / 6)

    # o3dpcd.translate(-trans)
    # o3dpcd.rotate(np.linalg.inv(rot), center=rot_center)

    o3dpcd.transform(cam_mat4)
    o3dpcd.transform(np.linalg.inv(rm.homomat_from_posrot(trans, rot)))

    if toggledebug:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1)
        o3d.visualization.draw_geometries([coord, o3dpcd], mesh_show_back_face=True)
        # o3d.visualization.draw_geometries([o3dpcd], mesh_show_back_face=True)

    vis.destroy_window()
    os.remove(f'./{tmp_f_name}.pcd')
    return o3dpcd


def gen_partial_o3dpcd_occ(path, f, rot, rot_center, trans=np.zeros(3), resolusion=(1280, 720), cam_pos=(0, 0, 0),
                           rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6), vis_threshold=np.radians(75),
                           occ_vt_ratio=1.0, noise_vt_ratio=1.0,
                           add_noise_vt=False, add_occ_nrml=False, add_occ_vt=False, add_occ_rnd=True,
                           add_noise_pts=True, noise_cnt=random.randint(0, 5),
                           othermesh=[], fov=False, w_otherpcd=False, toggledebug=False):
    o3dmesh = o3d.io.read_triangle_mesh(os.path.join(path, 'mesh', f + '.ply'))

    vis = o3d.visualization.Visualizer()
    vis.create_window('win', width=resolusion[0], height=resolusion[1], left=0, top=0)
    o3dmesh.rotate(rot, center=rot_center)
    o3dmesh.translate(trans)
    for mesh in othermesh:
        vis.add_geometry(mesh)
    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(os.path.join(path, f'{f}_tmp.pcd'), do_render=False, convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))

    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    o3dpcd_nrml = np.asarray(o3dpcd.normals)
    vis_idx = np.argwhere(np.arccos(abs(o3dpcd_nrml.dot(np.asarray([0, 0, 1])))) < vis_threshold).flatten()

    if add_occ_rnd:
        o3dpcd = du.add_random_occ(o3dpcd, occ_ratio_rng=rnd_occ_ratio_rng)
    if add_occ_nrml:
        o3dpcd = du.add_random_occ_by_nrml(o3dpcd, occ_ratio_rng=nrml_occ_ratio_rng)
    if add_occ_vt:
        o3dpcd = du.add_random_occ_by_vt(o3dpcd, np.asarray(o3dmesh.vertices),
                                         edg_radius=5e-4, edg_sigma=5e-4, ratio=occ_vt_ratio)
    if add_noise_vt:
        o3dmesh.compute_vertex_normals()
        o3dpcd = du.add_guassian_noise_by_vt(o3dpcd, np.asarray(o3dmesh.vertices), np.asarray(o3dmesh.vertex_normals),
                                             noise_mean=1e-3, noise_sigma=2e-4, ratio=noise_vt_ratio)
    if add_noise_pts:
        o3dpcd = du.add_noise_pts_by_vt(o3dpcd, noise_cnt=noise_cnt, size=.03)
    o3dpcd = o3dpcd.select_by_index(vis_idx)

    # o3dpcd = du.resample(o3dpcd, smp_num=2048)
    o3dpcd, _ = o3dpcd.remove_radius_outlier(nb_points=10, radius=0.05)
    o3dpcd = o3dpcd.voxel_down_sample(voxel_size=.001)

    if not w_otherpcd:
        pcd_kdt = KDTree(np.asarray(o3dpcd.points))
        mesh_kdt = KDTree(np.asarray(o3dmesh.vertices))
        idxs = list(pcd_kdt.query_ball_tree(mesh_kdt, r=0.006))
        selected_idx = np.asarray([i for i in range(len(idxs)) if len(idxs[i]) > 0])
        o3dpcd = o3dpcd.select_by_index(selected_idx)
    if fov:
        o3dpcd = filer_pcd_by_cam(o3dpcd, cam_pos, dist=1.5, angle=np.pi / 6)

    if toggledebug:
        o3dpcd_org = o3d.io.read_point_cloud(os.path.join(path, f'{f}_tmp.pcd'))
        o3dpcd_org.paint_uniform_color([0, 0.7, 1])
        o3dpcd.paint_uniform_color(COLOR[0])
        o3d.visualization.draw_geometries([o3dpcd] + othermesh)
        # o3d.visualization.draw_geometries([o3dpcd_org])

    o3dpcd.translate(-trans)
    o3dpcd.rotate(np.linalg.inv(rot), center=rot_center)

    vis.destroy_window()
    os.remove(os.path.join(path, f'{f}_tmp.pcd'))

    return o3dpcd


def filer_pcd_by_cam(o3dpcd, cam_pos, cam_ls=None, dist=.8, angle=np.pi / 9):
    if len(np.asarray(o3dpcd.points)) == 0:
        return o3dpcd
    o3dpcd_kdt = o3d.geometry.KDTreeFlann(o3dpcd)
    _, radius_vis_idx, _ = o3dpcd_kdt.search_radius_vector_3d(cam_pos, dist)
    o3dpcd = o3dpcd.select_by_index(radius_vis_idx)
    pcd_fov = np.asarray(o3dpcd.points)
    # pcd_fov = np.asarray([p for p in np.asarray(o3dpcd.points)
    #                       if rm.angle_between_vectors(cam_mat4[:3, 3] - p, cam_mat4[:3, 3]) < angle])
    if cam_ls is not None:
        pcd_fov = np.asarray([p for p in pcd_fov if rm.angle_between_vectors(cam_pos - p, cam_ls) < angle])

    if len(pcd_fov) == 0:
        return o3d.geometry.PointCloud()
    o3dpcd = o3dh.nparray2o3dpcd(pcd_fov)
    return o3dpcd.voxel_down_sample(voxel_size=0.001)


def read_pcn_res_pytorch(result_path, test_path, id, toggledebug=False):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    if toggledebug:
        o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][id]))
        o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][id]))
        o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][id]))
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
        o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])
    return np.asarray(test_f['complete_pcds'][id]), \
           np.asarray(test_f['incomplete_pcds'][id]), \
           np.asarray(res_f['results'][id])


def is_complete(pts_pcn, pts, radius=.01, threshold=50):
    def _normalize(l):
        return [(v - min(l)) / (max(l) - min(l)) for v in l]

    if len(pts_pcn) == 0:
        return False
    _, _, trans = o3dh.registration_icp_ptpt(pts_pcn, pts, maxcorrdist=.02, toggledebug=False)
    pts_pcn = pcdu.trans_pcd(pts_pcn, trans)
    # pcdu.show_pcd(pts_pcn, rgba=(.7, 0, 0, .5))
    # pcdu.show_pcd(pts, rgba=(.7, .7, .7, .5))
    o3d_pcn = du.nparray2o3dpcd(pts_pcn)
    # kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts_pcn, n_components=16, show=False)
    o3d_kpts = o3d_pcn.voxel_down_sample(voxel_size=.002)
    kpts = np.asarray(o3d_kpts.points)
    confs = pcdu.cal_distribution(pts, kpts, radius=radius)
    o3dpcd = o3dh.nparray2o3dpcd(pts)
    print(np.asarray(confs).max(), np.asarray(confs).mean(), np.asarray(confs).min(), np.asarray(confs).std())
    # o3d_kpts.paint_uniform_color(COLOR[2])
    # o3dpcd.paint_uniform_color(COLOR[0])
    # o3d.visualization.draw_geometries([o3dpcd, o3d_kpts])
    if min(confs) > threshold:
        return True
    return False


'''
open3d related
'''


def gen_o3d_arrow(spos, epos, rgb=[0, 0, 1]):
    vec_len = np.linalg.norm(np.array(epos) - np.array(spos))
    o3d_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06 * vec_len,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04 * vec_len
    )
    o3d_arrow.paint_uniform_color(rgb)
    o3d_arrow.compute_vertex_normals()

    rot_mat = rm.rotmat_between_vectors((0, 0, 1), np.array(epos) - np.array(spos))
    o3d_arrow.rotate(rot_mat, center=(0, 0, 0))
    o3d_arrow.translate(np.array(spos))

    return o3d_arrow


def gen_o3d_sphere(pos, radius=.01, rgb=[0, 0, 1]):
    o3d_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    o3d_sphere.paint_uniform_color(rgb)
    o3d_sphere.translate(np.array(pos))

    return o3d_sphere


def rbt2o3dmesh(rbt, link_num=10, show_nrml=True):
    import modeling.collision_model as cm
    import basis.trimesh as trm
    rbtcm_list = rbt.gen_meshmodel().cm_list[link_num:]
    vertices = []
    vertex_normals = []
    faces = []
    for tmp_cm in rbtcm_list:
        tmp_vertices, tmp_vertex_normals, tmp_faces = tmp_cm.extract_rotated_vvnf()
        tmp_faces = np.asarray(tmp_faces) + len(vertices)
        vertices.extend(tmp_vertices)
        vertex_normals.extend(tmp_vertex_normals)
        faces.extend(list(tmp_faces))
    rbt_cm = cm.CollisionModel(initor=trm.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces),
                                                  vertex_normals=np.asarray(vertex_normals)),
                               btwosided=True)
    rbt_o3d = o3dh.cm2o3dmesh(rbt_cm)
    if show_nrml:
        rbt_o3d.compute_vertex_normals()
    return rbt_o3d


def show_nbv_o3d(pts_nbv, nrmls_nbv, confs_nbv, o3dpcd, coord=None, o3dpcd_o=None):
    nbv_mesh_list = []
    for i in range(len(pts_nbv)):
        nbv_mesh_list.append(gen_o3d_arrow(pts_nbv[i], pts_nbv[i] + rm.unit_vector(nrmls_nbv[i]) * .02,
                                           rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]]))
    circle_mesh = gen_o3d_sphere(pts_nbv[0], radius=.002, rgb=[0, 0, 1])
    o3dpcd.paint_uniform_color(COLOR[0])
    if o3dpcd_o is not None:
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o, circle_mesh, coord] + nbv_mesh_list)
    else:
        o3d.visualization.draw_geometries([o3dpcd, circle_mesh, coord] + nbv_mesh_list)


def show_pcn_res_pytorch(result_path, test_path):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')
    for i in range(10, len(test_f['complete_pcds'])):
        o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][i]))
        o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][i]))
        o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][i]))
        o3dpcd_gt.paint_uniform_color(COLOR[1])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
        o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])


'''
panda3d related
'''


def attach_nbv_gm(pts, nrml, conf, cam_pos, arrow_len, thickness=.002):
    for i in range(len(pts)):
        gm.gen_arrow(pts[i], pts[i] + nrml[i] * arrow_len / np.linalg.norm(nrml[i]),
                     rgba=(conf[i], 0, 1 - conf[i], 1), thickness=thickness).attach_to(base)
    # gm.gen_dashstick(cam_pos, pts[0], rgba=(.7, .7, 0, .5), thickness=.002).attach_to(base)


def attach_nbv_conf_gm(pts, nrml, conf, cam_pos, arrow_len, thickness=.002):
    for i in range(len(pts)):
        if conf[i] > .2:
            continue
        gm.gen_arrow(pts[i], pts[i] + nrml[i] * arrow_len / np.linalg.norm(nrml[i]),
                     rgba=(conf[i], 0, 1 - conf[i], 1), thickness=thickness).attach_to(base)
        # gm.gen_sphere(pts[i], radius=.01, rgba=(conf[i], 0, 1 - conf[i], .2)).attach_to(base)
        gm.gen_dashstick(cam_pos, pts[i], rgba=(.7, .7, 0, .5), thickness=.002).attach_to(base)


if __name__ == '__main__':
    from sympy import Plane, Line3D  # Point3D it's not needed here

    # plane Points
    a1 = [1, 0, 0]
    a2 = [0, 1, 0]
    a3 = [0, 0, 1]
    # line Points
    p0 = [0, 0, 0]  # point in line
    v0 = [1, 1, 1]  # line direction as vector

    # create plane and line
    plane = Plane(a1, a2, a3)
    line = Line3D(p0, direction_ratio=a2)
    line2 = Line3D(p0, direction_ratio=v0)

    print(f"plane equation: {plane.equation()}")
    print(f"line equation: {line.equation()}")

    # find intersection:

    intr = line2.intersection(line)
    print(intr)
