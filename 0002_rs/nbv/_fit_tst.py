import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import bendplanner.bend_utils as bu
import datagenerator.data_utils as du
import utils.pcd_utils as pcdu
import visualization.panda.world as wd
from geomdl import BSpline
from sklearn.neighbors import NearestNeighbors
import nbv_utils as nbv_utl

if __name__ == '__main__':
    cam_pos = [0, 0, .5]
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'D:/nbv_mesh'
    cat = 'bspl'
    fo = 'res_75'
    prefix = 'pcn'
    obj_id = '0001'

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    res_dict = json.load(open(os.path.join(path, cat, fo, f'{prefix}_{obj_id}.json'), 'rb'))
    objcm_gt = du.o3dmesh2cm(o3d.io.read_triangle_mesh(os.path.join(path, cat, 'mesh', f'{obj_id}.ply')))
    pts = res_dict['1']['pcn_output']
    # pts = res_dict['final']
    pts = pcdu.remove_outliers(pts, toggledebug=False)

    # o3dpts = du.nparray2o3dpcd(pts)
    # o3dpts = o3dpts.voxel_down_sample(voxel_size=.02)
    # kpts = np.asarray(o3dpts.points)
    # kpts = pcdu.sort_kpts(kpts, seed=np.asarray([0, 0, 0]))
    # kpts_rotseq = pcdu.get_rots_wkpts(pts, kpts, show=True, rgba=(1, 0, 0, 1))

    kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts, rgba=(1, 1, 0, 1), n_components=16)
    pcdu.show_pcd(pts)

    o3dpcd = du.nparray2o3dpcd(pts)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3dpcd, .005)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # inp_pseq = np.asarray(
    #     interpolate.splev(np.linspace(0, 1, 100), interpolate.splprep(kpts.transpose(), k=5)[0], der=0)
    # ).transpose()

    inp_pseq = nbv_utl.nurbs_inp(kpts)
    # inp_pseq, inp_rotseq = du.get_rotseq_by_pseq(inp_pseq)
    inp_rotseq = pcdu.get_rots_wkpts(pts, inp_pseq, show=True, rgba=(1, 0, 0, 1))
    kpts = np.asarray(kpts)

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='gray', s=.01, alpha=.5)
    ax.plot(kpts[:, 0], kpts[:, 1], kpts[:, 2])
    ax.plot(inp_pseq[:, 0], inp_pseq[:, 1], inp_pseq[:, 2])
    ax_min = min([min(pts[:, 0]), min(pts[:, 1]), min(pts[:, 2])])
    ax_max = max([max(pts[:, 0]), max(pts[:, 1]), max(pts[:, 2])])
    ax.set_xlim([ax_min, ax_max])
    ax.set_ylim([ax_min, ax_max])
    ax.set_zlim([ax_min, ax_max])
    plt.show()

    # for i, rot in enumerate(inp_rotseq):
    #     gm.gen_frame(inp_pseq[i], inp_rotseq[i], thickness=.001, length=.03).attach_to(base)
    objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008)
    objcm_kpts = bu.gen_swap(kpts, kpts_rotseq, cross_sec, extend=.008)

    cd = nbv_utl.chamfer_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2', direction='bi')
    hd = nbv_utl.hausdorff_distance(objcm.objtrm.vertices, objcm_gt.objtrm.vertices, metric='l2')
    print(round(cd * 1000, 2), round(hd * 1000, 2))

    objcm.set_rgba((1, 1, 1, .5))
    objcm.attach_to(base)

    objcm_kpts.set_rgba((1, 1, 0, .5))
    objcm_kpts.attach_to(base)

    objcm_gt.set_rgba((0, 1, 0, .5))
    objcm_gt.attach_to(base)
    base.run()
