import pickle
import config
import cv2
import utils.vision_utils as vu
import numpy as np
import utils.pcd_utils as pcdu
import utils.phoxi_locator as pl

import basis.o3dhelper as o3dh
import open3d as o3d
import robot_sim.robots.yumi.yumi as ym
import modeling.geometric_model as gm

if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    robot_s = ym.Yumi()  # simulation rbt_s
    robot_s.gen_meshmodel().attach_to(base)

    labelimg = cv2.imread(config.ROOT + '\img\phoxi\labelimg\skull.jpg')
    phxi_info = pickle.load(open(config.ROOT + '\img\phoxi\skull.pkl', 'rb'))
    grayimg = phxi_info[0]
    depthimg = phxi_info[1]
    pcd = pcdu.trans_pcd(phxi_info[2] / 1000,
                         transmat=pickle.load(open(config.ROOT + '/camcalib/data/phoxi_calibmat_yumi.pkl', 'rb')))
    mask = vu.extract_label_rgb(labelimg, toggledebug=False)
    # mask = vu.extract_clr_gray(grayimg, clr=(10, 50), toggledebug=True)
    grayimg = grayimg * mask
    pcd_ext = vu.map_gray2pcd(grayimg, pcd)
    _, pts = vu.mask2skeleton(mask)

    mask_sk = vu.pts2mask(pts, shape=(772, 1032, 1))
    pcd_sk = vu.map_gray2pcd(grayimg * mask_sk, pcd)

    x_range = (.2, .6)
    y_range = (-.4, .4)
    z_range = (.02, .4)
    pcd = pcdu.crop_pcd(pcd, x_range, y_range, z_range)
    kdt, _ = pcdu.get_kdt(pcd)
    # for p in pcd_ext:
    #     nrml = pcdu.get_nrml_pca(pcdu.get_knn(p, kdt, k=100))
    #     if nrml[2] < 0:
    #         nrml = -nrml
    #     gm.gen_arrow(spos=p, epos=p + nrml * .01, thickness=.001).attach_to(base)

    for p in pcd_sk:
        nrml = pcdu.get_nrml_pca(pcdu.get_knn(p, kdt, k=100))
        if nrml[2] < 0:
            nrml = -nrml
        gm.gen_arrow(spos=p, epos=p + nrml * .01, thickness=.001, rgba=(1, 1, 0, 1)).attach_to(base)
    pseq, nseq = pcdu.surface_interp(pcd_sk[0], pcd_sk[-1] - pcd_sk, kdt)
    for i in range(len(nseq)):
        gm.gen_arrow(spos=pseq[i], epos=pseq[i] + nseq[i] * .01, thickness=.001, rgba=(1, 1, 0, 1)).attach_to(base)

    pcdu.show_pcd(pcd, rgba=(1, 1, 1, .1))
    pcdu.show_pcd(pcd_ext, rgba=(1, 0, 0, 1))
    base.run()
