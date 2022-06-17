import copy

import numpy as np
from pygem import RBF

import basis.trimesh as trm
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd



def trans_pcd(pcd, transmat):
    pcd = np.asarray(pcd)
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def trans_pos(pts, pos):
    return rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray(pos), np.eye(3)), pts)


def rot_new_orgin(pts, new_orgin, rot):
    trans_pts = trans_pos(pts, new_orgin)
    trans_pts = rm.homomat_transform_points(rm.homomat_from_posrot(np.asarray([0, 0, 0]), rot), trans_pts)
    return trans_pos(trans_pts, -new_orgin).tolist()


def get_rotseq_by_pseq(pseq):
    rotseq = []
    for i in range(1, len(pseq)):
        v1 = pseq[i] - pseq[i - 1]
        y = np.asarray([0, 1, 0])
        n = np.cross(v1, y)
        if rm.angle_between_vectors(n, [0, 0, 1]) > np.pi / 2:
            n = -n
        rot = np.asarray([rm.unit_vector(v1), rm.unit_vector(y), rm.unit_vector(n)]).T
        rotseq.append(rot)
    rotseq = [rotseq[0]] + rotseq
    return np.asarray(rotseq)


def gen_plate_ctr_pts(pts, goal_pseq, edge=0.0):
    org_len = np.linalg.norm(pts[:, 0].max() - pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_diff_uni = org_len * goal_diff / goal_diff.sum()
    # x_range = np.linspace(pts[:, 0].min(), pts[:, 0].max(), num)
    x = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min()) - edge
    y_max = float(pts[:, 1].max()) + edge
    z_min = float(pts[:, 2].min()) - edge
    z_max = float(pts[:, 2].max()) + edge
    ctr_pts = [[x, y_min, z_min],
               [x, y_max, z_min],
               [x, y_max, z_max],
               [x, y_min, z_max]]
    for i in range(len(goal_diff_uni)):
        x += goal_diff_uni[i]
        ctr_pts.extend([[x, y_min, z_min],
                        [x, y_max, z_min],
                        [x, y_max, z_max],
                        [x, y_min, z_max]])

    return np.asarray(ctr_pts)


def gen_deformed_ctr_pts(ctr_pts, goal_pseq, rot_diff=None):
    goal_rotseq = get_rotseq_by_pseq(goal_pseq)
    org_len = np.linalg.norm(ctr_pts[:, 0].max() - ctr_pts[:, 0].min())
    goal_diff = np.linalg.norm(np.diff(goal_pseq, axis=0), axis=1)
    goal_pseq = org_len * goal_pseq / goal_diff.sum()

    org_kpts = ctr_pts.reshape((int(len(ctr_pts) / 4), 4, 3)).mean(axis=1)

    deformed_ctr_pts = []
    if len(ctr_pts) != len(goal_pseq) * 4:
        print('Wrong goal_diff size!', ctr_pts.shape, goal_pseq.shape)
        return None
    for i in range(len(goal_pseq)):
        gm.gen_frame(goal_pseq[i], goal_rotseq[i], length=.01, thickness=.001).attach_to(base)
        gm.gen_frame(org_kpts[i], np.eye(3), length=.01, thickness=.001,
                     rgbmatrix=np.asarray([[1, 1, 0], [1, 0, 1], [0, 1, 1]])).attach_to(base)
        transmat4 = np.dot(rm.homomat_from_posrot(goal_pseq[i], goal_rotseq[i]),
                           np.linalg.inv(rm.homomat_from_posrot(org_kpts[i], np.eye(3))))
        deformed_ctr_pts.extend(trans_pcd(ctr_pts[i * 4:(i + 1) * 4], transmat4))

    if rot_diff is not None:
        deformed_ctr_pts_rot = []
        for i in range(len(rot_diff)):
            # rot_new_orgin(deformed_ctr_pts[i * 4:(i + 1) * 4],
            #               goal_pseq[i],
            #               rm.rotmat_from_axangle(goal_rotseq[i][:, 0], rot_diff[i]))
            # transmat4 = rm.homomat_from_posrot((0, 0, 0), rm.rotmat_from_axangle(goal_rotseq[i][:, 0], rot_diff[i]))
            deformed_ctr_pts_rot.extend(rot_new_orgin(deformed_ctr_pts[i * 4:(i + 1) * 4],
                                                      goal_pseq[i],
                                                      rm.rotmat_from_axangle(goal_rotseq[i][:, 0], rot_diff[i])))
        deformed_ctr_pts = np.copy(deformed_ctr_pts_rot)

    for p in ctr_pts:
        gm.gen_sphere(p, radius=.001, rgba=(1, 1, 0, 1)).attach_to(base)
    for p in deformed_ctr_pts:
        gm.gen_sphere(p, radius=.001, rgba=(1, 0, 0, 1)).attach_to(base)
    return np.asarray(deformed_ctr_pts)


if __name__ == '__main__':
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])

    objcm = cm.CollisionModel('../obstacles/plate.stl')
    objcm.attach_to(base)
    # objcm.set_rgba((.7, .7, .7, .7))
    base.run()
    # gm.gen_frame(length=.02, thickness=.001).attach_to(base)
    vs = objcm.objtrm.vertices
    ctr_num = 5
    radius = .05
    goal_pseq = np.asarray([[0, 0, 0],
                            [.05, 0, .003],
                            [.08, 0, 0],
                            [.10, 0, -.003],
                            [.16, 0, -.001]])

    original_ctr_pts = gen_plate_ctr_pts(vs, goal_pseq)

    # rot_diff = np.radians([0, 5, 10, -10, 0])
    rot_diff = None

    deformed_ctr_pts = gen_deformed_ctr_pts(original_ctr_pts, goal_pseq, rot_diff)
    rbf = RBF(original_control_points=original_ctr_pts, deformed_control_points=deformed_ctr_pts, radius=radius)

    new_vs = rbf(vs)

    objtrm = trm.Trimesh(vertices=np.asarray(new_vs), faces=objcm.objtrm.faces)
    new_objcm = cm.CollisionModel(initor=objtrm, btwosided=True, name='plate_deform')
    new_objcm.set_rgba((.7, .7, 0, 1))
    new_objcm.attach_to(base)
    base.run()
