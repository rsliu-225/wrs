import pickle
import config
import cv2
import utils.vision_utils as vu
import numpy as np
import utils.pcd_utils as pcdu
import bendplanner.bend_utils as bu
import utils.phoxi_locator as pl
import basis.robot_math as rm
import basis.o3dhelper as o3dh
import open3d as o3d
import robot_sim.robots.yumi.yumi as ym
import modeling.geometric_model as gm


def move_to_init(pseq, rotseq):
    init_homomat = rm.homomat_from_posrot([0, 0, 0], np.eye(3))
    goal_homomat = rm.homomat_from_posrot(pseq[0], rotseq[0])
    transmat4 = np.dot(init_homomat, np.linalg.inv(goal_homomat))
    pseq = rm.homomat_transform_points(transmat4, pseq).tolist()
    rotseq = np.asarray([transmat4[:3, :3].dot(r) for r in rotseq])
    return pseq, rotseq


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])
    f_name = 'skull2'
    robot_s = ym.Yumi()  # simulation rbt_s
    robot_s.gen_meshmodel().attach_to(base)

    labelimg = cv2.imread(config.ROOT + f'\img\phoxi\labelimg\{f_name}.jpg')
    phxi_info = pickle.load(open(config.ROOT + '\img\phoxi\skull.pkl', 'rb'))
    grayimg = phxi_info[0]
    depthimg = phxi_info[1]
    pcd = pcdu.trans_pcd(phxi_info[2] / 1000,
                         transmat=pickle.load(open(config.ROOT + '/camcalib/data/phoxi_calibmat_yumi.pkl', 'rb')))
    mask = vu.extract_label_rgb(labelimg, toggledebug=False)
    # mask = vu.extract_clr_gray(grayimg, clr=(10, 50), toggledebug=True)
    pcd_ext = vu.map_gray2pcd(grayimg * mask, pcd)
    mask_sk = vu.mask2skmask(mask, inp=1, toggledebug=False)
    pcd_sk = vu.map_gray2pcd(grayimg * mask_sk, pcd)

    x_range = (.2, .6)
    y_range = (-.4, .4)
    z_range = (.02, .4)
    pcd = pcdu.crop_pcd(pcd, x_range, y_range, z_range)
    pcd = np.append(pcd, pcd_ext, axis=0)
    kdt, _ = pcdu.get_kdt(pcd)

    # res_pseq = []
    # res_rotseq = []
    # for i, p in enumerate(pcd_sk[:-1]):
    #     rot = pcdu.get_frame_pca(pcdu.get_knn(p, kdt, k=100))
    #     v = pcd_sk[i + 1] - pcd_sk[i]
    #     # gm.gen_arrow(spos=p, epos=p + v, thickness=.001, rgba=(1, 0, 0, 1)).attach_to(base)
    #     # gm.gen_frame(pos=p, rotmat=rot, thickness=.001, length=.01, alpha=.5).attach_to(base)
    #     v = rm.unit_vector(v - rot[:, 2].dot(v))
    #     # gm.gen_dasharrow(spos=p, epos=p + v, thickness=.001, rgba=(1, 1, 0, 1)).attach_to(base)
    #
    #     rot = np.asarray([rot[:, 2], v, -np.cross(v, rot[:, 2])]).T
    #     res_pseq.append(p)
    #     res_rotseq.append(rot)
    #     # gm.gen_frame(pos=p, rotmat=rot, thickness=.001, length=.02, alpha=.5,
    #     #              rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
    #
    # res_pseq, res_rotseq = bu.inp_rotp_by_step(res_pseq, res_rotseq, step=.0001)
    res_pseq, res_rotseq = pcdu.surface_interp(pcd_sk[0], pcd_sk[-1] - pcd_sk[0], kdt)
    for i in range(len(res_pseq)):
        gm.gen_frame(pos=res_pseq[i], rotmat=res_rotseq[i], thickness=.001, length=.01).attach_to(base)

    res_pseq, res_rotseq = move_to_init(res_pseq, res_rotseq)
    for i in range(len(res_pseq)):
        gm.gen_frame(pos=res_pseq[i], rotmat=res_rotseq[i], thickness=.001, length=.01).attach_to(base)

    pickle.dump([res_pseq, res_rotseq], open(config.ROOT + f'/data/bend/rotpseq/{f_name}.pkl', 'wb'))
    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
    pcdu.show_pcd(pcd_ext, rgba=(1, 0, 0, 1))
    base.run()
