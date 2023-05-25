import h5py
import random
import numpy as np
import basis.o3dhelper as o3dh
import nbv_utils as nu
import open3d as o3d
import utils.pcd_utils as pcdu
import modeling.geometric_model as gm
import bendplanner.bend_utils as bu

width = .008
thickness = .0015
cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]


def show_res(result_path, test_path, label=1):
    res_f = h5py.File(result_path, 'r')
    test_f = h5py.File(test_path, 'r')

    print(len(test_f['complete_pcds']))
    cam_pos = np.asarray([0, 0, .1])

    while True:
        i = random.randint(0, len(test_f['complete_pcds']))
        # for i in range(30000):
        if test_f['labels'][i] == label:
            o3dpcd_gt = o3dh.nparray2o3dpcd(np.asarray(test_f['complete_pcds'][i]))
            o3dpcd_i = o3dh.nparray2o3dpcd(np.asarray(test_f['incomplete_pcds'][i]))
            o3dpcd_o = o3dh.nparray2o3dpcd(np.asarray(res_f['results'][i]))
            pcdu.show_pcd(np.asarray(o3dpcd_i.points), list(nu.COLOR[0]) + [1])
            pcdu.show_pcd(np.asarray(o3dpcd_o.points), list(nu.COLOR[2]) + [1])

            kpts, kpts_rotseq = pcdu.get_kpts_gmm(np.asarray(o3dpcd_o.points), rgba=(1, 1, 0, 1), n_components=15)
            inp_pseq = nu.nurbs_inp(kpts)
            inp_rotseq = pcdu.get_rots_wkpts(np.asarray(o3dpcd_o.points), inp_pseq, k=250, show=True, rgba=(1, 0, 0, 1))
            objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec, extend=.008)
            o3dmesh = o3dh.cm2o3dmesh(objcm)

            pts_nbv, nrmls_nbv, confs_nbv = \
                pcdu.cal_nbv_pcn(np.asarray(o3dpcd_i.points), np.asarray(o3dpcd_o.points), campos=cam_pos, theta=None,
                                 toggledebug=True)
            base.run()

            # nbv_mesh_list = []
            # for i in range(len(pts_nbv)):
            #     nbv_mesh_list.append(nu.gen_o3d_arrow(pts_nbv[i],
            #                                           pts_nbv[i] + nrmls_nbv[i] * .02 / np.linalg.norm(nrmls_nbv[i]),
            #                                           rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]]))
            # nbv_mesh_list.append(nu.gen_o3d_sphere(pts_nbv[0], radius=.005,
            #                                        rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]]))
            #
            # # o3dpcd_o.estimate_normals()
            # o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
            # o3dpcd_i.paint_uniform_color(nu.COLOR[0])
            # o3dpcd_o.paint_uniform_color(nu.COLOR[2])
            # o3d.visualization.draw_geometries(nbv_mesh_list + [o3dpcd_i, o3dpcd_o])
            # o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])
            # o3d.visualization.draw_geometries([o3dpcd_gt, o3dpcd_o])
            # # draw_geometry_with_rotation([o3dpcd_i, o3dpcd_o])
            # # draw_geometry_with_rotation([o3dpcd_o, o3dpcd_gt])


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    cam_pos = [0, 0, .4]

    result_path = 'D:/liu/MVP_Benchmark/completion/log/pcn_emd_rlen/results.h5'
    test_path = 'E:/liu/h5_data/data_conf/test.h5'
    show_res(result_path, test_path, label=2)
