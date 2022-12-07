from nbv_sim import *

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255
RES_FO_NAME = 'res_75'

if __name__ == '__main__':
    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv/best_cd_p_network.pth'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    cam_pos = [0, 0, .5]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    path = 'E:/liu/nbv_mesh/'
    cat = 'bspl'
    coverage_tor = .001
    goal = .95
    visible_threshold = np.radians(75)

    if not os.path.exists(os.path.join(path, cat, RES_FO_NAME)):
        os.makedirs(os.path.join(path, cat, RES_FO_NAME))
    for f in os.listdir(os.path.join(path, cat, 'mesh'))[1:]:
        print(f'-----------{f}------------')
        # if os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'pcn_{f.split(".ply")[0]}.json')) and \
        #         os.path.exists(os.path.join(path, cat, RES_FO_NAME, f'org_{f.split(".ply")[0]}.json')):
        #     continue

        o3dpcd_init = gen_partial_view(os.path.join(path, cat), f.split('.ply')[0], np.eye(3), [0, 0, 0],
                                       rnd_occ_ratio_rng=(.2, .5), nrml_occ_ratio_rng=(.2, .6),
                                       visible_threshold=visible_threshold,
                                       occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
                                       add_vt_occ=False, add_noise=False, add_rnd_occ=False, add_noise_pts=True,
                                       toggledebug=False)
        # o3dpcd_init, ind = o3dpcd_init.remove_radius_outlier(nb_points=50, radius=0.005)

        o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
        o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')

        run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, model_name, load_model,
                goal=goal, coverage_tor=coverage_tor, visible_threshold=visible_threshold, toggledebug=True)
        # run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
        #         visible_threshold=visible_threshold, toggledebug=False)
        # run_random(path, cat, f, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
        #            visible_threshold=visible_threshold, toggledebug=False)
        # run_pcn_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, goal=goal, coverage_tor=coverage_tor,
        #             visible_threshold=visible_threshold, toggledebug=False)
