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

    path = './tst'

    for i in range(10):
        # objcm, _, _, _ = utl.gen_seed(200, max=random.uniform(.04, .05), n=100, toggledebug=False)
        objcm, _, _, _ = utl.gen_seed(5, max=random.uniform(.02, .03), n=100, toggledebug=False)

        objcm.set_rgba((.7, .7, 0, 1))
        objcm.attach_to(base)

        utl.get_objpcd_partial_o3d(objcm, objcm, np.eye(3), (0, 0, 0), path=path, f_name=f'tst',
                                   occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1),
                                   rnd_occ_ratio_rng=(.1, .3), visible_threshold=np.radians(75),
                                   add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                   savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)

    # base.run()
