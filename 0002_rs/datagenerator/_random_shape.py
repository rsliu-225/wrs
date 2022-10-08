import random

from _shape_dict import *
import utils as utl
import numpy as np
from scipy.interpolate import Rbf


def gen_seed(kpts, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, random=False):
    width = width + (np.random.uniform(0, 0.005) if random else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    if len(kpts) == 3:
        pseq = utl.uni_length(utl.poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
    # elif len(kpts) == 4:
    #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
    else:
        pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
    pseq = np.asarray(pseq) - pseq[0]
    pseq, rotseq = utl.get_rotseq_by_pseq(pseq)
    return utl.gen_swap(pseq, rotseq, cross_sec)


def random_kts(n=3, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append(((j + 1) * .02, random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])

    for i in range(4):
        kpts = random_kts(4, max=random.uniform(0, .04))
        objcm = gen_seed(kpts, n=200)
        objcm.set_rgba((1, 0, 0, .7))
        objcm.attach_to(base)

    base.run()
