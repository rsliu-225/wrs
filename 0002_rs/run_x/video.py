import pickle
import time

import cv2

import localenv.envloader as el
import motionplanner.motion_planner as mp
import motionplanner.rbtx_motion_planner as mpx
import utils.phoxi as phoxi
import visualization.panda.world as wd
from pcn.inference import *

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    fo = 'template_2'
    dump_path = f'phoxi/nbc_opt/{fo}'

    rbt = el.loadXarm(showrbt=False)
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    m_planner = mp.MotionPlanner(env=None, rbt=rbt, armname='arm')
    rbtx = el.loadXarmx(ip='10.2.0.201')
    m_planner_x = mpx.MotionPlannerRbtX(env=None, rbt=rbt, rbtx=rbtx, armname='arm')

    i = 0

    m_planner_x.goto_init_x()
    time.sleep(5)

    pcd_cmp = np.asarray([])
    pcd_icp_list = []
    pcd_pcn_list = []
    trans_icp = np.eye(4)

    while i < 4:
        if i != 0:
            seedjntagls = m_planner_x.get_armjnts()
            try:
                pts_nbv, nrmls_nbv, confs_nbv, transmat4, jnts, pcd_pcn = \
                    pickle.load(open(os.path.join(config.ROOT, 'img', dump_path, f'{str(i).zfill(3)}_res.pkl'), 'rb'))
            except:
                continue
            print(jnts)
            path = m_planner.plan_start2end(start=seedjntagls, end=jnts)
            m_planner_x.movepath(path)
            time.sleep(3)
        textureimg, depthimg, pcd = phxi.getalldata()
        cv2.imwrite(os.path.join(config.ROOT, 'img', dump_path, f'{str(i).zfill(3)}.jpg'), textureimg)
        textureimg = cv2.imread(os.path.join(config.ROOT, 'img', dump_path, f'{str(i).zfill(3)}.jpg'))
        # cv2.imshow('', textureimg)
        # cv2.waitKey(0)
        i += 1

    base.run()
