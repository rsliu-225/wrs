import cv2
from sklearn.cluster import DBSCAN
import random
import motionplanner.motion_planner as m_planner
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.vision_utils as vu
import utils.prj_utils as pu
import utiltools.thirdparty.o3dhelper as o3dh
from utils.run_script_utils import *
import matplotlib.pyplot as plt
import utils.run_script_utils as rsu


def mask2pts(mask):
    p_list = []
    for i, row in enumerate(mask):
        for j, val in enumerate(row):
            if val > 0:
                p_list.append((i, j))
    return np.asarray(p_list)


def get_max_cluster(pts, eps=6, min_samples=20):
    pts_narray = np.asarray(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    res = []
    max_len = 0

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            cluster = pts_narray[class_member_mask & core_samples_mask]
            if len(cluster) > max_len:
                max_len = len(cluster)
                res = cluster
    return res


def get_clusters(pts, eps=6, min_samples=20):
    pts_narray = np.asarray(pts)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    unique_labels = set(labels)
    print("cluster:", unique_labels)
    res = []
    clusters = []

    for k in unique_labels:
        if k == -1:
            continue
        else:
            class_member_mask = (labels == k)
            clusters.append(pts_narray[class_member_mask & core_samples_mask])
    return clusters


def pts2mask(narray, shape):
    mask = np.zeros(shape)
    for p in narray:
        mask[p[0], p[1]] = 1
    return mask


def extract_stroke(gray1, gray2, threshold=20, toggledebug=False, erode=False):
    shape = (772, 1032, 1)
    diff = np.abs(gray1.astype(int) - gray2.astype(int)).astype(np.uint8)

    mask = diff > threshold
    crop_mask = np.zeros(shape)
    crop_mask[400:580, 540:720] = 1
    # crop_mask[480:550, 520:620] = 1
    # crop_mask[480:550, 520:680] = 1  # force
    # crop_mask[400:460, 650:720] = 1  # bunny
    # crop_mask[500:600, 550:650] = 1  # helmet

    mask = mask * crop_mask

    diff_p_narray = mask2pts(mask)
    stroke_pts = np.array(get_max_cluster(diff_p_narray, eps=3, min_samples=20))
    stroke_mask = pts2mask(stroke_pts, shape)
    if erode:
        kernel = np.ones((2, 2), np.uint8)
        stroke_mask = cv2.erode(stroke_mask, kernel, iterations=1)
    # print(stroke_mask)

    if toggledebug:
        # cv2.imshow('diff', diff)
        # cv2.waitKey(0)
        cv2.imshow('cropped mask', mask)
        cv2.waitKey(0)
        cv2.imshow('clustered mask', stroke_mask)
        cv2.waitKey(0)

    return stroke_mask


def show_thickness(phoxi_f_name1, phoxi_f_name2, ):
    grayimg_org, _, pcd_org = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name1, load=True)
    grayimg_res, _, pcd_res = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name2, load=True)
    stroke_mask = extract_stroke(grayimg_org, grayimg_res, toggledebug=False, threshold=20)
    edges = cv2.Canny(np.uint8(stroke_mask * 255), 100, 200)
    cv2.imshow('edge', edges)
    cv2.waitKey(0)
    clusters = get_clusters(mask2pts(edges), eps=3, min_samples=1)
    pcd_stroke_list = []
    for cluster in clusters:
        rgba = (random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), 1)
        mask = pts2mask(cluster, shape=stroke_mask.shape)
        pcd_stroke = vu.map_gray2pcd(mask, pcdu.trans_pcd(pcd_org, phxilocator.amat))
        for p in pcd_stroke:
            base.pggen.plotSphere(base.render, p, rgba=rgba)
        pcd_stroke_list.append(pcd_stroke)
    kdt, _ = pu.get_kdt(pcd_stroke_list[0])
    dist_list = []
    for p in pcd_stroke_list[1]:
        pt = pu.get_knn(p, kdt)[0]
        dist_list.append(np.linalg.norm(pt - p))
    print(dist_list)
    plt.plot(dist_list)
    plt.show()


def compare(phoxi_f_name1, phoxi_f_name2, penpos_list, rgba=(1, 1, 1, 1)):
    print(phoxi_f_name1, phoxi_f_name2)
    grayimg_org, _, pcd_org = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name1, load=True)
    grayimg_res, _, pcd_res = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name2, load=True)
    stroke_mask = extract_stroke(grayimg_org, grayimg_res, toggledebug=True, threshold=30)

    pcd_stroke = vu.map_gray2pcd(stroke_mask, pcdu.trans_pcd(pcd_org, phxilocator.amat))
    # pcdu.show_pcd(pcd_stroke)

    rmse, fitness, transmat = o3dh.registration_ptpt(np.asarray(pcd_stroke), np.asarray(penpos_list), toggledebug=False)
    # pcdu.show_pcd(pcdu.trans_pcd(pcd_stroke, transmat), rgba=rgba)
    for p in pcdu.trans_pcd(pcd_stroke, transmat):
        base.pggen.plotSphere(base.render, p, rgba=(1, 1, 0, .5))
    print(f'rmse: {rmse}')
    print(f'fitness: {fitness}')

    return rmse, fitness


def show_in_grayimg(phoxi_f_name, penpos_list):
    print(phoxi_f_name)
    grayimg, _, pcd = ru.load_phxiinfo(phoxi_f_name=phoxi_f_name, load=True)
    img = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2RGB)
    pcd = pcdu.trans_pcd(pcd, phxilocator.amat)
    # kdt, _ = pu.get_kdt(pcd)
    for p_pcd in penpos_list:
        # inx_pcd = pu.get_knn_indices(p_pcd, kdt, k=1)[0]
        inx_pcd, result_point = pcdu.get_pcdidx_by_pos(pcd, p_pcd)
        p_img = vu.map_pcdpinx2graypinx(inx_pcd, grayimg)
        print(p_pcd, inx_pcd, p_img)
        cv2.circle(img, p_img, 1, (0, 0, 255))
    cv2.imshow('', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')

    # exp_name = 'cylinder_cad'
    # exp_name = 'force'
    # exp_name = 'helmet'
    # exp_name = 'bunny'
    exp_name = 'raft'
    folder_path = os.path.join(config.MOTIONSCRIPT_REL_PATH + '/exp_' + exp_name + '/')
    phoxi_f_path = 'phoxi_tempdata_' + exp_name + '.pkl'

    draw_path_name = 'draw_circle'
    # draw_path_name = 'draw_star'

    folder_name = "exp_" + exp_name + "/"
    phoxi_f_name = "phoxi_tempdata_" + exp_name + ".pkl"
    phoxi_f_name_result = f"phoxi_tempdata_{exp_name}_{draw_path_name.split('_')[1]}.pkl"
    # phoxi_f_name_result = f"phoxi_tempdata_{exp_name}_refined_{draw_path_name.split('_')[1]}.pkl"
    # phoxi_f_name_result = f"phoxi_tempdata_{exp_name}_t.pkl"
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    # setting_real(phxilocator, phoxi_f_name, config.PEN_STL_F_NAME, None)
    id_list = config.ID_DICT[exp_name]

    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")

    objrelpos, objrelrot, path_draw = load_motion_sgl(draw_path_name, folder_name, id_list)
    penpos_list = []
    for armjnts in path_draw:
        penmat4 = mp_lft.get_world_objmat4(objrelpos, objrelrot, armjnts=armjnts)
        penpos_list.append(penmat4[:3, 3])

    # show_in_grayimg(phoxi_f_name_result, penpos_list)
    # show_in_grayimg(phoxi_f_name_result_refined, penpos_list)
    compare(phoxi_f_name, phoxi_f_name_result, penpos_list, rgba=(1, 1, 0, 1))
    # compare(phoxi_f_name, phoxi_f_name_result_refined, penpos_list, rgba=(0, 0, 1, 1))
    # show_thickness(phoxi_f_name, phoxi_f_name_result)

    for pos in penpos_list:
        base.pggen.plotSphere(base.render, pos, rgba=(0, .7, .7, 1), radius=1)
    # base.run()
