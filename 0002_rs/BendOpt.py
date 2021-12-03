import math
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import BendSim


def gen_circle(r):
    pts = []
    for a in np.arange(0, 2 * math.pi, math.pi / 90):
        pts.append([r * math.cos(a), r * math.sin(a), 0])
    return pts


def cal_length(pseq):
    length = 0
    for i in range(len(pseq)):
        if i != 0:
            length += np.linalg.norm(np.asarray(pseq[i]) - np.asarray(pseq[i - 1]))
    # print(np.cumsum(np.sqrt(np.sum(np.diff(np.asarray(pseq), axis=1) ** 2, axis=0))))
    return length


def show_pseq(pseq, rgba):
    for p in pseq:
        gm.gen_sphere(pos=np.asarray(p), rgba=rgba, radius=0.0002).attach_to(base)


def align_pseqs(pseq1, pseq2):
    v1 = np.asarray(pseq1[1]) - np.asarray(pseq1[0])
    v2 = np.asarray(pseq2[1]) - np.asarray(pseq2[0])
    rot = rm.rotmat_between_vectors(v1, v2)
    pseq2 = [np.dot(rot, np.asarray(p)) for p in pseq2]
    p1 = np.asarray(pseq1[0])
    p2 = np.asarray(pseq2[0])
    pseq2 = [np.asarray(p) - (p2 - p1) for p in pseq2]
    return pseq2


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, 1], lookat_pos=[0, 0, 0])

    goal_pseq = gen_circle(.01)
    length = cal_length(goal_pseq)
    init_pseq = [(0, 0, 0),(0, .01, 0), (0, length, 0)]
    init_rotseq = [np.eye(3),np.eye(3), np.eye(3)]
    bs = BendSim.BendSim(thickness=0.0015, width=.002, pseq=init_pseq, rotseq=init_rotseq)
    bs.bend(np.radians(20), np.radians(0), insert_l=.02)
    res_pseq = bs.pseq
    # pseq = align_pseqs(goal_pseq, res_pseq)
    print(length)
    print(math.pi * .02)
    show_pseq(goal_pseq, rgba=(1, 0, 0, .1))
    # show_pseq(init_pseq, rgba=(0, 1, 0, 1))
    show_pseq(res_pseq, rgba=(0, 1, 1, 1))
    bs.show()
    base.run()

