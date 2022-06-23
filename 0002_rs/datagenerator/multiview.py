import itertools
import os
import pickle
import random

import numpy as np
import open3d as o3d

import datagenerator.utils as utl
import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as wd


def gen_multiview(view_num, comb_num=1, path='', trans_diff=(.01, .01, .01), rot_diff=np.radians((10, 10, 10)),
                  add_occ=True, toggledebug=False):
    if not os.path.exists(os.path.join(path, 'multiview/')):
        os.mkdir(os.path.join(path, 'multiview/'))

    homomat4_dict = pickle.load(open(f'{path}/homomat4_dict.pkl', 'rb'))
    for objid, v in homomat4_dict.items():
        rid_list = list(v.keys())
        for rid_init in rid_list:
            init_homomat = v[rid_init]
            o3dpcd = o3d.io.read_point_cloud(f"{path}/partial/{objid}_{rid_init}.pcd")
            if add_occ:
                pcd_mv = list(utl.add_random_occ_narry(o3dpcd.points, occ_ratio_rng=(.3, .6)))
            else:
                pcd_mv = o3dpcd.points
            combs = random.choices(list(itertools.combinations([i for i in rid_list if i != rid_init], view_num)),
                                   k=comb_num)
            print(rid_init, combs)
            for i, comb in enumerate(combs):
                for rot_id in comb:
                    homomat4 = v[rot_id]
                    random_homomat4 = utl.gen_random_homomat4(trans_diff, rot_diff)
                    o3dpcd = o3d.io.read_point_cloud(f"{path}/partial/{objid}_{rot_id}.pcd")
                    pcd = np.asarray(o3dpcd.points)
                    pcd = utl.trans_pcd(pcd, np.dot(init_homomat, np.linalg.inv(homomat4)))
                    # gm.gen_pointcloud(pcd, rgbas=[[1, 1, 0, 1]]).attach_to(base)
                    pcd = utl.trans_pcd(pcd, random_homomat4)
                    if add_occ:
                        pcd = utl.add_random_occ_narry(pcd, occ_ratio_rng=(.3, .6))
                    pcd_mv.extend(pcd)
                o3d.io.write_point_cloud(os.path.join(path, 'multiview', f'{objid}_{rid_init}_{str(i).zfill(3)}.pcd'),
                                         utl.nparray2o3dpcd(pcd_mv))

                if toggledebug:
                    o3dpcd_complete = o3d.io.read_point_cloud(f"{path}/complete/{objid}_{rid_init}.pcd")
                    gm.gen_pointcloud(o3dpcd_complete.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
                    gm.gen_pointcloud(pcd_mv).attach_to(base)
                    base.run()


def show(folder_name='./'):
    for f in sorted(os.listdir(os.path.join(folder_name, 'multiview/'))):
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, 'multiview/', f))
            gm.gen_pointcloud(o3dpcd.points).attach_to(base)
            print(f.split('_'))
            o3dpcd = o3d.io.read_point_cloud(
                os.path.join(folder_name, 'complete/', f'{f.split("_")[0]}_{f.split("_")[1]}.pcd'))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)

    base.run()


if __name__ == '__main__':
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    folder_name = 'tst'
    trans_diff = (.001, .001, .001)
    rot_diff = np.radians((1, 1, 1))
    view_num = 3
    comb_num = 1

    gen_multiview(view_num=view_num, comb_num=comb_num, path=folder_name, trans_diff=trans_diff, rot_diff=rot_diff,
                  toggledebug=False)

    show(folder_name)
