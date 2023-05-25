import numpy as np

from nbc_sim import *


def show_pcn_opt(path, cat, fo, f, relmat4, cam_pos, show_o3d=False, show_p3d=True):
    res_dict = json.load(open(os.path.join(path, cat, fo, f'pcn_opt_{f.split(".ply")[0]}.json'), 'rb'))

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)

    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)

    print(f'-----------pcn_opt------------')
    pcd_gt = np.asarray(res_dict['gt'])
    init_coverage = res_dict['init_coverage']
    seedjntagls = np.asarray(res_dict['init_jnts'])
    print('init coverage:', init_coverage)
    o3dpcd_gt = o3dh.nparray2o3dpcd(pcd_gt)
    o3dpcd_gt.paint_uniform_color(nu.COLOR[1])
    pcdu.show_cam(rm.homomat_from_posrot(cam_pos, rot=config.CAM_ROT))

    for i in range(10):
        k = str(i)
        rbt.fk('arm', seedjntagls)
        init_eepos, init_eerot = rbt.get_gl_tcp()
        init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)

        pcd_i = np.asarray(res_dict[k]['input'])
        pcd_o = np.asarray(res_dict[k]['pcn_output'])

        pcd_nxt = np.asarray(res_dict[k]['add'])
        o3dpcd_nxt = o3dh.nparray2o3dpcd(pcd_nxt)
        jnts = np.asarray(res_dict[k]['jnts'])
        time_cost = res_dict[k]['time_cost']
        coverage = res_dict[k]['coverage']
        print(i, 'coverage:', coverage)

        pcd_inhnd = pcdu.trans_pcd(pcd_i, init_eemat4)
        pcd_o_inhnd = pcdu.trans_pcd(pcd_o, init_eemat4)

        o3dpcd = o3dh.nparray2o3dpcd(pcd_i)
        o3dpcd_inhnd = o3dh.nparray2o3dpcd(pcd_inhnd)
        o3dpcd_o_inhnd = o3dh.nparray2o3dpcd(pcd_o_inhnd)
        o3dpcd.paint_uniform_color(nu.COLOR[0])
        o3dpcd_inhnd.paint_uniform_color(nu.COLOR[0])
        o3dpcd_o_inhnd.paint_uniform_color(nu.COLOR[2])

        # pts_nbv_inhnd = res_dict[k]['pts_nbv']
        # nrmls_nbv_inhnd = res_dict[k]['nrmls_nbv']
        rbt_o3dmesh = nu.rbt2o3dmesh(rbt, link_num=10, show_nrml=show_o3d)
        pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv = pcdu.cal_nbv_pcn(pcd_inhnd, pcd_o_inhnd)

        if show_o3d:
            coord_inhnd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
            coord_inhnd.transform(init_eemat4)
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord_inhnd, o3dpcd_inhnd])
            nu.show_nbv_o3d(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, o3dpcd_inhnd, coord_inhnd, o3dpcd_o_inhnd)

        if show_p3d:
            gm.gen_frame().attach_to(base)
            pcdu.show_pcd(pcd_inhnd)
            pcdu.show_pcd(pcd_o_inhnd, rgba=(nu.COLOR[2][0], nu.COLOR[2][1], nu.COLOR[2][2], 1))
            rbt.gen_meshmodel().attach_to(base)
            rbt.gen_meshmodel(rgba=(1, 1, 0, .4)).attach_to(base)
            # nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .05)
            nu.attach_nbv_conf_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .05)

        rbt.fk('arm', jnts)
        eepos, eerot = rbt.get_gl_tcp()
        eemat4 = rm.homomat_from_posrot(eepos, eerot).dot(relmat4)
        transmat4 = np.dot(eemat4, np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt = nu.rbt2o3dmesh(rbt, link_num=10, show_nrml=show_o3d)

        if show_p3d:
            pts_nbv_inhnd = pcdu.trans_pcd(pts_nbv_inhnd, transmat4)
            nrmls_nbv_inhnd = pcdu.trans_pcd(nrmls_nbv_inhnd, transmat4)
            pcdu.show_pcd(pcdu.trans_pcd(pcd_i, eemat4), rgba=(0, 1, 0, 1))
            pcdu.show_pcd(pcdu.trans_pcd(pcd_o, eemat4), rgba=(nu.COLOR[2][0], nu.COLOR[2][1], nu.COLOR[2][2], 1))
            rbt.gen_meshmodel(rgba=(0, 1, 0, .5)).attach_to(base)
            # nu.attach_nbv_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .05)
            nu.attach_nbv_conf_gm(pts_nbv_inhnd, nrmls_nbv_inhnd, confs_nbv, cam_pos, .05)
            base.run()

        '''
        get new pcd
        '''
        rbt_o3dmesh.transform(np.linalg.inv(init_eemat4))
        rbt_o3dmesh_nxt.transform(np.linalg.inv(init_eemat4))
        transmat4_origin = np.linalg.inv(init_eemat4).dot(eemat4)
        coord_nxt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.05)
        coord_nxt.transform(transmat4_origin)
        if show_o3d:
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, o3dpcd])
            o3d.visualization.draw_geometries([rbt_o3dmesh, coord, rbt_o3dmesh_nxt, coord_nxt, o3dpcd])

        if show_o3d:
            o3dpcd_nxt.paint_uniform_color(nu.COLOR[5])
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_nxt, coord])

        if show_o3d:
            o3d.visualization.draw_geometries([o3dpcd, o3dpcd_gt, coord])

        seedjntagls = jnts


if __name__ == '__main__':
    import localenv.envloader as el

    model_name = 'pcn'
    load_model = 'pcn_emd_rlen/best_emd_network.pth'
    RES_FO_NAME = 'res_75_rbt'

    path = 'D:/nbv_mesh/'
    if not os.path.exists(path):
        path = 'E:/liu/nbv_mesh/'

    cat = 'bspl_4'
    cam_pos = [0, 0, .4]
    cov_tor = .001
    goal = .95
    vis_threshold = np.radians(75)
    relmat4 = rm.homomat_from_posrot([.02, 0, 0], np.eye(3))

    rbt = el.loadXarm(showrbt=False)
    rbt.jaw_to('hnd', 0)
    init_eepos, init_eerot = rbt.get_gl_tcp()
    init_eemat4 = rm.homomat_from_posrot(init_eepos, init_eerot).dot(relmat4)
    cam_pos = pcdu.trans_pcd([cam_pos], init_eemat4)[0]

    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 1])
    for f in os.listdir(os.path.join(path, cat, 'mesh')):
        # print(f'-----------{f}------------')
        # o3dpcd_init = \
        #     nu.gen_partial_o3dpcd_occ(os.path.join(path, cat), f.split('.ply')[0], np.eye(3), [0, 0, 0],
        #                               rnd_occ_ratio_rng=(.2, .4), nrml_occ_ratio_rng=(.2, .6),
        #                               vis_threshold=vis_threshold, toggledebug=False,
        #                               occ_vt_ratio=random.uniform(.08, .1), noise_vt_ratio=random.uniform(.2, .5),
        #                               noise_cnt=random.randint(1, 5),
        #                               add_occ_vt=True, add_noise_vt=False, add_occ_rnd=False, add_noise_pts=True)
        # o3d.io.write_point_cloud('./tmp/nbc_vis/init.pcd', o3dpcd_init)
        o3dmesh_gt = o3d.io.read_triangle_mesh(os.path.join(path, cat, 'prim', f))
        o3dpcd_gt = du.get_objpcd_full_sample_o3d(o3dmesh_gt, smp_num=2048, method='possion')
        # o3d.io.write_point_cloud('./tmp/nbc_vis/gt.pcd', o3dpcd_gt)

        o3dpcd_init = o3d.io.read_point_cloud('./tmp/nbc_vis/init.pcd')
        # o3dpcd_gt = o3d.io.read_point_cloud('./tmp/nbc_vis/gt.pcd')

        # run_nbv(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, goal=goal, cov_tor=cov_tor,
        #         vis_threshold=vis_threshold, toggledebug=True, toggledebug_p3d=False)
        run_random(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, goal=goal, cov_tor=cov_tor,
                   vis_threshold=vis_threshold, toggledebug=True, toggledebug_p3d=False)
        # run_pcn(path, cat, f, cam_pos, o3dpcd_init, o3dpcd_gt, relmat4, model_name, load_model, goal=goal,
        #         cov_tor=cov_tor, vis_threshold=vis_threshold, toggledebug=False, toggledebug_p3d=True)
        # show_pcn_opt(path, cat, RES_FO_NAME, f, relmat4, cam_pos, show_o3d=False, show_p3d=True)
