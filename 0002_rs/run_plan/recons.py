import modeling.geometric_model as gm
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
import visualization.panda.world as wd

if __name__ == '__main__':
    import bendplanner.bend_utils as bu

    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    fo = 'nbc_pcn/extrude_1'
    # fo = 'nbc/plate_a_cubic'
    # fo = 'opti/plate_a_cubic'
    # fo = 'seq/plate_a_quadratic'
    gm.gen_frame().attach_to(base)

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]][::-1]

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, .0155)
    # center = (0, 0, 0)

    # x_range = (.07, .2)
    # y_range = (-.15, .15)
    # z_range = (.0165, .2)
    x_range = (.1, .2)
    y_range = (-.15, .02)
    z_range = (-.1, -.03)
    # gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=False, icp=True)
    pts = []
    for pcd in pcd_cropped_list:
        pts.extend(pcd)
    # pts = pcdu.remove_outliers(pts, nb_points=1000, radius=0.01, toggledebug=True)
    kpts, kpts_rotseq = pcdu.get_kpts_gmm(pts, rgba=(1, 1, 0, 1), n_components=20)

    # kpts = bu.linear_inp3d_by_step(kpts)
    # kpts, kpts_rotseq = bu.inp_rotp_by_step(kpts, kpts_rotseq)

    for i, rot in enumerate(kpts_rotseq):
        gm.gen_frame(kpts[i], kpts_rotseq[i], thickness=.001, length=.03).attach_to(base)
    objcm = bu.gen_swap(kpts, kpts_rotseq, cross_sec)
    objcm.set_rgba((1, 1, 1, .5))
    objcm.attach_to(base)

    # surface = b_surface.BiBSpline(pts[:, :2], pts[:, 2])
    # surface = rbf_surface.RBFSurface(pts[:, :2], pts[:, 2])
    # surface_gm = surface.get_gometricmodel([[min(pts[:, 0]), max(pts[:, 0])], [min(pts[:, 1]), max(pts[:, 1])]],
    #                                        rgba=[.5, .7, 1, .5])
    # surface_gm.attach_to(base)

    pcdu.show_pcd(pts, rgba=(1, 1, 0, 1))

    # rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1]]
    # for i, pcd in enumerate(pcd_cropped_list):
    #     pcdu.show_pcd(pcd, rgba=rgba_list[i])

    # pcdu.show_pcd(pts, rgba_list[0])
    # for fo in sorted(os.listdir(os.path.join(config.ROOT, 'recons_data'))):
    #     if fo[:2] == 'pl':
    #         print(fo)
    #         pcd_cropped_list = reg_plate(fo, seed, center)

    # skeleton(pcd_cropped)
    # pcdu.cal_conf(pcd_cropped, voxel_size=0.005, radius=.005)

    base.run()
