import copy
import random

import data_utils as utl
import visualization.panda.world as wd
import basis.robot_math as rm
import math
import numpy as np
import modeling.geometric_model as gm
import basis.trimesh as trm

if __name__ == '__main__':

    base = wd.World(cam_pos=[0, 0, .5], lookat_pos=[0, 0, 0])
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    # icos = trm.creation.icosphere(1)
    # icos_cm = cm.CollisionModel(icos)
    # icos_cm.attach_to(base)

    path = './tst'

    for i in range(2):
        objcm, _, _, _ = utl.gen_seed(20, max=random.uniform(.01, .04), n=200, toggledebug=True)
        # for matlist in icomats:
        #     np.random.shuffle(matlist)
        #     for j, rot in enumerate(matlist):
        #         gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001, rgba=(.7, .7, .7, .7)).attach_to(base)
        #     for j, rot in enumerate(matlist[:10]):
        #         gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001).attach_to(base)
        #         # objcm_tmp = copy.deepcopy(objcm)
        #         # objcm_tmp.set_homomat(rm.homomat_from_posrot(rot=rot))
        #         # objcm_tmp.attach_to(base)
        objcm.set_rgba((.7, .7, 0, 1))
        objcm.attach_to(base)

        utl.get_objpcd_partial_o3d(objcm, objcm, np.eye(3), (0, 0, 0), path=path, f_name=f'tst',
                                   occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1),
                                   rnd_occ_ratio_rng=(.1, .3), visible_threshold=np.pi / 3,
                                   add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                   savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)

    base.run()
