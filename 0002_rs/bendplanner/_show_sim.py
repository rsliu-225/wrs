import copy
import math
import random

import numpy as np

import basis.robot_math as rm
import basis.trimesh as trm
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import modeling.collision_model as cm
import modeling.geometric_model as gm
import utils.panda3d_utils as p3u
import visualization.panda.world as wd
import bendplanner.BendSim as bsim

if __name__ == '__main__':
    base = wd.World(cam_pos=[.075, .1, .05], lookat_pos=[0, 0, 0])
    bs = bsim.BendSim(show=False, cm_type='stick', granularity=np.pi / 30)
    bs.set_stick_sec(180)
    bs.pillar_center = cm.gen_stick(spos=np.asarray([0, 0, -bconfig.PILLAR_H / 3]),
                                    epos=np.asarray([0, 0, bconfig.PILLAR_H / 3]),
                                    thickness=bs.r_center * 2, sections=90,
                                    rgba=[.9, .9, .9, 1]).attach_to(base)
    bendset = [[np.radians(45), np.radians(0), np.radians(45), .01]]
    bs.reset([(0, 0, 0), (0, max(np.asarray(bendset)[:, 3]) + .015, 0)], [np.eye(3), np.eye(3)], extend=False)
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    bs.show(rgba=(.7, .7, 0, .7), show_frame=True)
    # bu.visualize_voxel([bs.voxelize()], colors=['r'])

    # bs.get_updown_primitive()

    # bs.move_to_org(.04)
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    # key_pseq, key_rotseq = bs.get_pull_primitive(.12, .04, toggledebug=True)
    # resseq = bs.pull(key_pseq, key_rotseq, np.pi)
    # bs.move_to_org(.04)
    # bs.show(rgba=(.7, .7, .7, .7), show_frame=True)

    base.run()
