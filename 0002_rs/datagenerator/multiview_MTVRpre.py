import os
import open3d as o3d
import numpy as np
import datagenerator.utils as utl
import visualization.panda.world as wd
import basis.robot_math as rm

import pickle
import random
import itertools
from _shape_dict import *
import modeling.geometric_model as gm


def write_info(transmat, f_name, path='./'):
    header = [0, 0, 0, 0]
    np.savetxt(os.path.join(path, f'{f_name}.info.txt'), np.asarray([header] + list(transmat)))


def load_info(f_name, path='./', skip_header=1):
    return np.genfromtxt(os.path.join(path, f'{f_name}.info.txt'), skip_header=skip_header)


if __name__ == '__main__':
    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    '''
    config params
    '''
    plate_len = .15
    width = .005
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    add_occ = True

    path = 'D:/3d_match_plate/'
    # path = 'D:/3d_match'
    multiview_folder_name = 'raw_data'
    if not os.path.exists(os.path.join(path, multiview_folder_name)):
        os.mkdir(os.path.join(path, multiview_folder_name))

    '''
    gen data
    '''
    icomats = []
    for v in rm.gen_icorotmats(rotation_interval=np.radians(45)):
        for icomat in v:
            icomats.append(icomat)

    for obj_id, pseq in shape_dict.items():
        if not os.path.exists(os.path.join(path, multiview_folder_name, obj_id)):
            os.mkdir(os.path.join(path, multiview_folder_name, obj_id))
        cnt = 0
        pseq = utl.cubic_inp(pseq=np.asarray(pseq))
        pseq = utl.uni_length(pseq, goal_len=.15)
        rotseq = utl.get_rotseq_by_pseq(pseq)

        objcm = utl.gen_swap(pseq, rotseq, cross_sec)
        # objcm.attach_to(base)
        for i, icomat in enumerate(random.choices(icomats, k=20)):
            rot_center = (random.uniform(-.1, .1), random.uniform(-.1, .1), random.uniform(-.1, .1))
            o3dpcd = utl.get_objpcd_partial_o3d(objcm, icomat, rot_center, path=path,
                                                f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}', ext_name='.pcd',
                                                add_noise=False, add_occ=True, toggledebug=False)

            o3d.io.write_point_cloud(
                os.path.join(path, multiview_folder_name, obj_id, f'cloud_bin_{str(cnt).zfill(3)}.ply'),
                o3dpcd)
            write_info(rm.homomat_from_posrot(rot_center, icomat), f'cloud_bin_{str(cnt).zfill(3)}',
                       path=os.path.join(path, multiview_folder_name, obj_id))
            cnt += 1
    # '''
    # show
    # '''
    # for fo in os.listdir(os.path.join(path, multiview_folder_name)):
    #     print(fo)
    #     for f in os.listdir(os.path.join(path, multiview_folder_name, fo)):
    #         if f[-3:] == 'ply':
    #             print(f)
    #             o3dpcd = o3d.io.read_point_cloud(os.path.join(path, multiview_folder_name, f))
    #             o3d.visualization.draw_geometries([o3dpcd])
    #             transmat4 = load_info(f[:-4], path=os.path.join(path, multiview_folder_name))
    #             print(transmat4)
    #
    # base.run()
