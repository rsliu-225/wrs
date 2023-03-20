import basis.robot_math as rm
import bendplanner.bend_utils as bu
import modeling.geometric_model as gm
import nbv.nbv_utils as nu
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd
import numpy as np
import pcn.inference as inf
from sklearn.neighbors import KDTree
import basis.o3dhelper as o3dh
import open3d as o3d


def get_meterial(rgba):
    material = o3d.visualization.rendering.Material()
    material.shader = 'defaultLitTransparency'
    material.base_color = rgba

    return material


if __name__ == '__main__':
    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    base = wd.World(cam_pos=[.4, -.4, -.4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, .4], lookat_pos=[0, 0, 0])

    fo = 'nbc_pcn/extrude_1'
    # fo = 'nbc_pcn/extrude_1'
    # gm.gen_frame().attach_to(base)

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]][::-1]

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, -.02)

    x_range = (.1, .25)
    y_range = (-.15, .02)
    # y_range = (-.02, .15)
    z_range = (-.15, -.02)
    gm.gen_frame(pos=(-.005, 0, 0), rotmat=rm.rotmat_from_axangle((1, 0, 0), np.pi),
                 length=.03, thickness=.002).attach_to(base)
    # gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=False, icp=True)
    pcd_all = []
    pcd_cropped_list[1] = pcdu.trans_pcd(pcd_cropped_list[1],
                                         rm.homomat_from_posrot((-.003, 0, 0),
                                                                rm.rotmat_from_euler(0, 0, 0)))
    # pcd_cropped_list[2] = pcdu.trans_pcd(pcd_cropped_list[2],
    #                                      rm.homomat_from_posrot((0, 0, -.002),
    #                                                             rm.rotmat_from_euler(0, 0, 0)))

    # pcdu.show_pcd(pcd_cropped_list[0], rgba=list(nu.COLOR[0]) + [1])
    # pcdu.show_pcd(pcd_cropped_list[1], rgba=list(nu.COLOR[0]) + [1])
    # pcdu.show_pcd(pcd_cropped_list[2], rgba=(.8, .8, 0, 1))
    # base.run()
    for pcd in pcd_cropped_list[:1]:
        pcd_all.extend(pcd)

    '''
    remove noise
    '''
    pcd_o = inf.inference_sgl(np.asarray(pcd_all))
    kdt_o = KDTree(pcd_o, leaf_size=2)
    dist, ind = kdt_o.query(pcd_all, k=1)
    inds = [i for i, d in enumerate(dist) if d[0] < .005]
    print(len(pcd_all), len(inds))
    pcd_all = np.asarray(pcd_all)[inds]

    pcd_o = inf.inference_sgl(np.asarray(pcd_all))
    # pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv_pcn(pcd_all, pcd_o, icp=False, toggledebug=False)
    pts, nrmls, confs = pcdu.cal_conf(pcd_all, voxel_size=.005, theta=None)
    pts_nbv, nrmls_nbv, confs_nbv = pcdu.cal_nbv(pts, nrmls, confs)

    kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_all, show=False, rgba=(1, 1, 0, 1), n_components=15)
    inp_pseq = nu.nurbs_inp(kpts)
    # inp_pseq, inp_rotseq = du.get_rotseq_by_pseq(inp_pseq)
    inp_rotseq = pcdu.get_rots_wkpts(pcd_all, inp_pseq, show=False, rgba=(1, 0, 0, 1))

    nbv_mesh_list = []
    for i in range(len(pts_nbv)):
        # if confs_nbv[i] > .2:
        #     continue
        o3d_arrow = nu.gen_o3d_arrow(pts_nbv[i], pts_nbv[i] + rm.unit_vector(nrmls_nbv[i]) * .02,
                                     rgb=[confs_nbv[i], 0, 1 - confs_nbv[i]])
        # nbv_mesh_list.append({'name': f'arrow_{i}', 'geometry': o3d_arrow,
        #                       'material': o3d.visualization.rendering.Material()})
        nbv_mesh_list.append(o3d_arrow)
        # nbv_mesh_list.append({'name': f'circle_{i}', 'geometry': o3d_circle,
        #                       'material': get_meterial([confs_nbv[i], 0, 1 - confs_nbv[i], .5])})
    o3d_circle = nu.gen_o3d_sphere(pts_nbv[0], radius=.001, rgb=[0, 1, 0])
    nbv_mesh_list.append(o3d_circle)

    o3dpcd = o3dh.nparray2o3dpcd(pcd_all)
    o3dpcd_1 = o3dh.nparray2o3dpcd(pcd_cropped_list[0])
    o3dpcd_2 = o3dh.nparray2o3dpcd(pcd_cropped_list[1])
    o3dpcd_o = o3dh.nparray2o3dpcd(pcd_o)
    o3dpcd.paint_uniform_color(nu.COLOR[0])
    o3dpcd_1.paint_uniform_color(nu.COLOR[0])
    o3dpcd_2.paint_uniform_color(nu.COLOR[-1])
    o3dpcd_o.paint_uniform_color(nu.COLOR[2])
    # nbv_mesh_list.append(o3dpcd)

    objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec)
    o3dmesh = o3dh.cm2o3dmesh(objcm)
    o3dmesh.compute_vertex_normals()

    # o3d.visualization.draw(nbv_mesh_list)
    # o3dh.custom_draw_geometry_with_rotation(nbv_mesh_list)
    o3dh.custom_draw_geometry_with_rotation([o3dpcd_1, o3dpcd_2])
    # o3dh.custom_draw_geometry_with_rotation([o3dmesh])
