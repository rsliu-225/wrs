import numpy as np
from sklearn.mixture import GaussianMixture

import basis.robot_math as rm
import modeling.geometric_model as gm
import utils.pcd_utils as pcdu
import utils.recons_utils as rcu
# import vision.depth_camera.surface.bibspline_surface as b_surface
import vision.depth_camera.surface.rbf_surface as rbf_surface
import visualization.panda.world as wd


def sort_kpts(kpts, seed):
    sort_ids = []
    while len(sort_ids) < len(kpts):
        dist_list = np.linalg.norm(kpts - seed, axis=1)
        sort_ids_tmp = np.argsort(dist_list)
        for i in sort_ids_tmp:
            if i not in sort_ids:
                sort_ids.append(i)
                break
        seed = kpts[sort_ids[-1]]
    return kpts[sort_ids]


def get_kpts_gmm(objpcd, n_components=20, show=True, rgba=(1, 0, 0, 1)):
    X = np.array(objpcd)
    gmix = GaussianMixture(n_components=n_components, random_state=0).fit(X)
    kpts = sort_kpts(gmix.means_, seed=np.asarray([0, 0, 0]))

    if show:
        for i, p in enumerate(kpts[1:]):
            gm.gen_sphere(p, radius=.001, rgba=rgba).attach_to(base)

    kdt, _ = pcdu.get_kdt(objpcd)
    kpts_rotseq = []
    for i, p in enumerate(kpts[:-1]):
        knn = pcdu.get_knn(kpts[i], kdt, k=50)
        pcv, pcaxmat = rm.compute_pca(knn)
        inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
        y_v = kpts[i + 1] - kpts[i]
        x_v = pcaxmat[:, inx[0]]
        if len(kpts_rotseq) != 0:
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 0], x_v) > np.pi / 2:
                x_v = -x_v
            if rm.angle_between_vectors(kpts_rotseq[-1][:, 1], y_v) > np.pi / 2:
                y_v = -y_v
        z_v = np.cross(x_v, y_v)

        rot = np.asarray([rm.unit_vector(x_v), rm.unit_vector(y_v), rm.unit_vector(z_v)]).T
        kpts_rotseq.append(rot)

    return kpts[:-1], np.asarray(kpts_rotseq)


if __name__ == '__main__':
    import bendplanner.bend_utils as bu

    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[0, 0, 0], lookat_pos=[0, 0, 1])
    # fo = 'nbc_pcn/plate_a_cubic'
    # fo = 'nbc/plate_a_cubic'
    fo = 'opti/plate_a_cubic'
    # fo = 'seq/plate_a_quadratic'
    gm.gen_frame().attach_to(base)

    width = .008
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    icp = False

    seed = (.116, -.1, .1)
    center = (.116, 0, .0155)

    x_range = (.065, .2)
    y_range = (-.15, .15)
    z_range = (.0165, .2)
    # z_range = (-.2, -.0155)
    # gm.gen_frame().attach_to(base)
    pcd_cropped_list = rcu.reg_armarker(fo, seed, center, x_range=x_range, y_range=y_range, z_range=z_range,
                                        toggledebug=False, icp=False)
    base.run()
    pts = []
    for pcd in pcd_cropped_list:
        pts.extend(pcd)
    pts = rcu.remove_outliers(pts)
    kpts, kpts_rotseq = get_kpts_gmm(pts, rgba=(1, 1, 0, 1))

    # kpts = bu.linear_inp3d_by_step(kpts)
    # kpts, kpts_rotseq = bu.inp_rotp_by_step(kpts, kpts_rotseq)

    for i, rot in enumerate(kpts_rotseq):
        gm.gen_frame(kpts[i], kpts_rotseq[i], thickness=.001, length=.03).attach_to(base)
    objcm = bu.gen_swap(kpts, kpts_rotseq, cross_sec)
    objcm.attach_to(base)

    # surface = b_surface.BiBSpline(pts[:, :2], pts[:, 2])
    surface = rbf_surface.RBFSurface(pts[:, :2], pts[:, 2])
    surface_gm = surface.get_gometricmodel([[min(pts[:, 0]), max(pts[:, 0])], [min(pts[:, 1]), max(pts[:, 1])]],
                                           rgba=[.5, .7, 1, .5])
    surface_gm.attach_to(base)

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
