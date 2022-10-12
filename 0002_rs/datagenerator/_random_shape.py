import copy
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


def gen_seed_smooth(kpts, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, random=False):
    width = width + (np.random.uniform(0, 0.005) if random else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    if len(kpts) == 3:
        pseq = utl.uni_length(utl.poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
    # elif len(kpts) == 4:
    #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
    else:
        pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
    pseq = np.asarray(pseq) - pseq[0]
    pseq, rotseq = utl.get_rotseq_by_pseq_smooth(pseq)
    return utl.gen_swap(pseq, rotseq, cross_sec)


def random_kts(n=3, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append(((j + 1) * .02, random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


def random_kts_sprl(n=4, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append((random.uniform(0, .2), random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


if __name__ == '__main__':
    import visualization.panda.world as wd
    import basis.robot_math as rm
    import math
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import basis.trimesh as trm

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])

    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icos = trm.creation.icosphere(1)
    icos_cm = cm.CollisionModel(icos)
    icos_cm.attach_to(base)

    path = './tst'

    for i in range(1):
        kpts = random_kts(4, max=random.uniform(0, .02))
        # kpts = random_kts_sprl(n=4, max=.01)
        objcm = gen_seed(kpts, n=200, toggledebug=False)
        for matlist in icomats:
            np.random.shuffle(matlist)
            for j, rot in enumerate(matlist):
                gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001, rgba=(.7, .7, .7, .7)).attach_to(base)
            for j, rot in enumerate(matlist[:10]):
                gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001).attach_to(base)
                # objcm_tmp = copy.deepcopy(objcm)
                # objcm_tmp.set_homomat(rm.homomat_from_posrot(rot=rot))
                # objcm_tmp.attach_to(base)
        objcm.set_rgba((.7, .7, 0, .7))
        objcm.attach_to(base)
        # objcm_smooth = gen_seed_smooth(kpts, n=200)
        # objcm_smooth.set_rgba((1, 1, 1, .7))
        # objcm_smooth.attach_to(base)

        utl.get_objpcd_partial_o3d(objcm, objcm, np.eye(3), (0,0,0), path=path, resolusion=(550, 550),
                                   f_name=f'{str(i)}_{str(cnt).zfill(3)}',
                                   occ_vt_ratio=random.uniform(.5, 1),
                                   noise_vt_ratio=random.uniform(.5, 1),
                                   add_noise=True, add_occ=True, toggledebug=True,
                                   savemesh=True, savedepthimg=True, savergbimg=True)

    base.run()
