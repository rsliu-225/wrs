import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from geomdl import BSpline
from sklearn.neighbors import NearestNeighbors
import basis.robot_math as rm
import datagenerator.data_utils as du
# import localenv.envloader as el
# import motionplanner.motion_planner as mp
import utils.pcd_utils as pcdu
import bendplanner.bend_utils as bu


def transpose(data):
    mat = [[]]
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            if len(mat) > j:
                mat[j].append(v)
            else:
                mat.append([v])
    return mat


def load_cov(path, cat, fo, prefix='pcn'):
    cov_list = []
    max_list = []
    cnt_list = [0] * 5
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        print(f'-----------{f}------------')
        try:
            res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
        except:
            break
        pcd_gt = res_dict['gt']
        cov_list_tmp = [res_dict['init_coverage']]
        max_tmp = [res_dict['init_coverage']]
        max = 0
        for i in range(5):
            if str(i) in res_dict.keys():
                print(prefix, i, res_dict[str(i)]['coverage'])
                cov_list_tmp.append(res_dict[str(i)]['coverage'])
                max = res_dict[str(i)]['coverage']
                cnt_list[i] += 1
            max_tmp.append(max)
        max_list.append(max_tmp)
        cov_list.append(cov_list_tmp)
    return cov_list, max_list, [cnt_list[0]] + cnt_list


def plot_box(ax, data, clr, positions):
    box = ax.boxplot(data, positions=positions)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box[item], color=clr)
    # plt.setp(box["boxes"], facecolor=clr)
    plt.setp(box["fliers"], markeredgecolor=clr)


def cal_avg(cnt_list):
    sum_v = 0
    for i in range(len(cnt_list)):
        try:
            sum_v += i * (cnt_list[i] - cnt_list[i + 1])
        except:
            sum_v += i * cnt_list[i]
    return sum_v / cnt_list[0]


def kpts2bspl(kpts):
    curve = BSpline.Curve()
    degree = 3
    curve.degree = degree
    curve.ctrlpts = kpts.tolist()
    curve.knotvector = [0] * degree + np.linspace(0, 1, len(kpts) - degree + 1).tolist() + [1] * degree
    curve.delta = 0.01
    inp_pseq = np.asarray(curve.evalpts)

    return inp_pseq


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


def load_pts(path, cat, fo, cross_sec, prefix='pcn', toggledebug=False):
    cd_list = []
    hd_list = []

    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        # if int(f.split(".ply")[0]) > 30:
        #     continue
        print(f'-----------{f}------------')
        try:
            res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{f.split(".ply")[0]}.json'), 'rb'))
            objcm_gt = du.o3dmesh2cm(o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f)))
        except:
            break
        cd_tmp = []
        hd_tmp = []
        stop = False
        for i in range(5):
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
            inp_pseq = kpts2bspl(kpts)
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

    return cd_list, hd_list


def gen_partial_o3dpcd(o3dmesh, rot=np.eye(3), trans=np.zeros(3), rot_center=(0, 0, 0)):
    vis = o3d.visualization.Visualizer()
    vis.create_window('win', left=0, top=0)
    o3dmesh = o3dmesh.filter_smooth_taubin(number_of_iterations=10)
    o3dmesh.rotate(rot, center=rot_center)
    o3dmesh.translate(trans)

    vis.add_geometry(o3dmesh)
    vis.poll_events()
    vis.capture_depth_point_cloud(f'./tmp.pcd', do_render=False, convert_to_world_coordinate=True)
    o3dpcd = o3d.io.read_point_cloud(f'./tmp.pcd')
    o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=10))
    o3dpcd_nrml = np.asarray(o3dpcd.normals)

    vis_idx = np.argwhere(np.arccos(abs(o3dpcd_nrml.dot(np.asarray([0, 0, 1])))) < np.radians(75)).flatten()
    o3dpcd = o3dpcd.select_by_index(vis_idx)

    o3dpcd.translate(-trans)
    o3dpcd.rotate(np.linalg.inv(rot), center=rot_center)

    return o3dpcd


def gen_o3d_arrow(spos, epos):
    vec_len = np.linalg.norm(np.array(epos) - np.array(spos))
    o3d_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06 * vec_len,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04 * vec_len
    )
    o3d_arrow.paint_uniform_color([1, 0, 1])
    o3d_arrow.compute_vertex_normals()

    rot_mat = rm.rotmat_between_vectors((0, 0, 1), np.array(epos) - np.array(spos))
    o3d_arrow.rotate(rot_mat, center=(0, 0, 0))
    o3d_arrow.translate(np.array(spos))

    return o3d_arrow
