import os
import pickle

import config
import motionplanner.motion_planner as m_planner
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.run_utils as ru
from localenv import envloader as el

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs(camp=[3000, -1000, 2000], lookatpos=[1000, 0, 1000])
    rbt, rbtmg, rbtball = el.loadUr3e(showrbt=True)
    rbt.opengripper(armname='rgt')
    rbt.opengripper(armname='lft')

    phoxi_f_path = 'phoxi_tempdata_cylinder_mtp.pkl'

    paintingobj_f_name = 'cylinder'
    match_rotz = False
    load = True

    sample_num = 1000000

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    '''
    init planner
    '''
    mp_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='lft')
    mp_rgt = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname='rgt')

    '''
    process image
    '''
    grayimg, depthnparray_float32, pcd = pickle.load(open(os.path.join("../img", phoxi_f_path), 'rb'))
    amat = pickle.load(open(os.path.join("../camcalib/data", config.AMAT_F_NAME), 'rb'))
    real_pcd = pcdu.trans_pcd(pcd, amat)
    pcdu.show_pcd([p for p in real_pcd if p[2] < 900], rgba=(.7, .7, .7, .1))
    base.run()

    for y in range(240, 500, 80):
        pen_item = \
            ru.get_obj_from_phoxiinfo_withmodel(phxilocator, config.PEN_STL_F_NAME, phoxi_f_name=phoxi_f_path,
                                                match_rotz=match_rotz, load=load,
                                                x_range=(700, 1000), y_range=(y, y + 100), z_range=(810, 870))
        pen_item.show_objcm(show_localframe=True)
        pen_item.show_objpcd(rgba=(1, 1, 0, 1))

    paintingobj_item = \
        ru.get_obj_from_phoxiinfo_withmodel(phxilocator, paintingobj_f_name + '.stl',
                                            phoxi_f_name=phoxi_f_path, match_rotz=match_rotz, load=True,
                                            x_range=(200, 900), y_range=(-100, 300), z_range=(790, 1000))
    paintingobj_item.show_objcm(show_localframe=True)
    paintingobj_item.show_objpcd(rgba=(1, 0, 0, 1))

    base.run()
