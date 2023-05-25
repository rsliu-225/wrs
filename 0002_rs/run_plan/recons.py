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

if __name__ == '__main__':

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    base = wd.World(cam_pos=[.4, -.4, -.4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, .4], lookat_pos=[0, 0, 0])

    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    # fo = 'nbc_opt/extrude_1'
    fo = 'nbc_pcn/extrude_1'
    # fo = 'nbc/plate_a_cubic'
    # fo = 'opti/plate_a_cubic'
    # fo = 'seq/plate_a_quadratic'
    # gm.gen_frame().attach_to(base)

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]][::-1]

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, -.02)
    # center = (0, 0, 0)

    # x_range = (.07, .2)
    # y_range = (-.15, .15)
    # z_range = (.0165, .2)
    x_range = (.1, .25)
    y_range = (-.15, .02)
    # y_range = (-.02, .15)
    z_range = (-.15, -.02)
    gm.gen_frame(pos=(-.005, 0, 0), rotmat=rm.rotmat_from_axangle((1, 0, 0), np.pi),
                 length=.03, thickness=.002).attach_to(base)
    gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=False, icp=True)
    pcd_all = []
    pcd_cropped_list[1] = pcdu.trans_pcd(pcd_cropped_list[1],
                                         rm.homomat_from_posrot((-.003, 0, 0),
                                                                rm.rotmat_from_euler(0, 0, 0)))
    # pcd_cropped_list[2] = pcdu.trans_pcd(pcd_cropped_list[2],
    #                                      rm.homomat_from_posrot((0, 0, -.002),
    #                                                             rm.rotmat_from_euler(0, 0, 0)))

    pcdu.show_pcd(pcd_cropped_list[0], rgba=list(nu.COLOR[0]) + [1])
    pcdu.show_pcd(pcd_cropped_list[1], rgba=list(nu.COLOR[0]) + [1])
    # pcdu.show_pcd(pcd_cropped_list[2], rgba=(.8, .8, 0, 1))
    base.run()
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
    # pcdu.show_pcd(pcd_o, rgba=list(nu.COLOR[2]) + [1])
    # pcdu.show_pcd(pcd_all, rgba=list(nu.COLOR[0]) + [1])
    # pcdu.cal_nbv_pcn(pcd_all, pcd_o, icp=False, toggledebug=True)
    # base.run()

    kpts, kpts_rotseq = pcdu.get_kpts_gmm(pcd_all, show=False, rgba=(1, 1, 0, 1), n_components=15)
    inp_pseq = nu.nurbs_inp(kpts)
    # inp_pseq, inp_rotseq = du.get_rotseq_by_pseq(inp_pseq)
    inp_rotseq = pcdu.get_rots_wkpts(pcd_all, inp_pseq, show=False, rgba=(1, 0, 0, 1))
    # for i, rot in enumerate(kpts_rotseq):
    #     gm.gen_frame(kpts[i], kpts_rotseq[i], thickness=.001, length=.01).attach_to(base)

    objcm = bu.gen_swap(inp_pseq, inp_rotseq, cross_sec)
    objcm.set_rgba((1, 1, 1, 1))
    objcm.attach_to(base)

    # surface = b_surface.BiBSpline(pts[:, :2], pts[:, 2])
    # surface = rbf_surface.RBFSurface(pts[:, :2], pts[:, 2])
    # surface_gm = surface.get_gometricmodel([[min(pts[:, 0]), max(pts[:, 0])], [min(pts[:, 1]), max(pts[:, 1])]],
    #                                        rgba=[.5, .7, 1, .5])
    # surface_gm.attach_to(base)

    pcdu.show_pcd(pcd_all, rgba=list(nu.COLOR[0]) + [.5])

    # rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1]]
    # for i, pcd in enumerate(pcd_cropped_list):
    #     pcdu.show_pcd(pcd, rgba=rgba_list[i])

    base.run()
