import numpy as np
from pygem import RBF

import basis.trimesh as trm
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd

base = wd.World(cam_pos=[.2, .2, 0], lookat_pos=[0, 0, 0])


def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def gen_plate_ctr_pts(pts, num=5):
    x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), num)
    ctr_pts = []
    for x in x_range:
        ctr_pts.extend([
            [x, pts[:, 1].min(), pts[:, 2].min()],
            [x, pts[:, 1].max(), pts[:, 2].min()],
            [x, pts[:, 1].max(), pts[:, 2].max()],
            [x, pts[:, 1].min(), pts[:, 2].max()],
        ])
    for p in ctr_pts:
        gm.gen_sphere(p, radius=.001).attach_to(base)
    return np.asarray(ctr_pts)


def gen_deformed_ctr_pts(ctr_pts, pos_diff, rot_diff):
    new_pts = []
    if len(ctr_pts) != len(pos_diff) * 4:
        print('Wrong goal_diff size!', ctr_pts.shape, pos_diff.shape)
        return None
    for i in range(len(rot_diff)):
        transmat4 = rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle((1, 0, 0), rot_diff[i]))
        new_pts.extend(trans_pcd(ctr_pts[i * 4:(i + 1) * 4], transmat4))
    pos_diff = np.repeat(pos_diff, 4, axis=0)
    deformed_ctr_pts = np.asarray(new_pts) + pos_diff

    org_kpts = ctr_pts[[i for i in range(len(ctr_pts)) if i % 4 == 0]]
    deformed_kpts = deformed_ctr_pts[[i for i in range(len(deformed_ctr_pts)) if i % 4 == 0]]
    org_len = np.linalg.norm(np.diff(org_kpts, axis=0), axis=1).sum()
    deformed_len = np.linalg.norm(np.diff(deformed_kpts, axis=0), axis=1).sum()
    print(org_len, deformed_len)

    for p in deformed_ctr_pts:
        gm.gen_sphere(p, radius=.001, rgba=(0, 1, 0, 1)).attach_to(base)
    return np.asarray(deformed_ctr_pts)


if __name__ == '__main__':
    objcm = cm.CollisionModel('../obstacles/plate.stl')

    vs = objcm.objtrm.vertices
    ctr_num = 5
    radius = .1
    original_ctr_pts = gen_plate_ctr_pts(vs, num=ctr_num)
    diff = np.asarray([[0, 0, 0],
                       [0, 0, .005],
                       [0, 0, 0],
                       [0, 0, -.005],
                       [0, 0, -.002]])
    rot_diff = np.radians([10, 10, -10, 10, 0])

    deformed_ctr_pts = gen_deformed_ctr_pts(original_ctr_pts, diff, rot_diff)
    rbf = RBF(original_control_points=original_ctr_pts, deformed_control_points=deformed_ctr_pts, radius=radius)

    new_vs = rbf(vs)

    objtrm = trm.Trimesh(vertices=np.asarray(new_vs), faces=objcm.objtrm.faces)
    new_objcm = cm.CollisionModel(initor=objtrm, btwosided=True, name='plate_deform')
    objcm.attach_to(base)
    objcm.set_rgba((.7, .7, .7, .7))
    new_objcm.attach_to(base)
    base.run()
