import os
import open3d as o3d
import numpy as np
import datagenerator.utils as utl
import visualization.panda.world as wd
import basis.robot_math as rm

import pickle

"""Record"""
x, x_2 = 0, 0
# Linear(total = 3*588)
a = [[0, 0, 0], [.03, 0, 0.005], [.07, 0, 0.01], [.15, 0.01, 0.01]]  # Twisting point near the ends
b = [[0, 0.001, 0.01], [.08, 0, 0.01], [0.14, 0, 0.0096], [.15, 0, 0.0096]]  # Twisting point near the middle
c = [[0, 0.001, 0.001], [.08, 0, 0.001], [0.14, 0, 0.001], [.15, 0.001, 0.001]]  # Flat

# Quadratic (total = 3*588)
aa = [[0, 0, 0], [.15, 0.005, 0.035], [.2, 0.01, 0]]  # 1 turning point with varying magnitude
bb = [[0, 0, 0], [.05, 0.01, 0], [.1, 0.03, 0.015], [.15 + x, 0.06, 0.025 + x * 3]]
# 1 turning point with varying magnitude & twisting point (2 points are near) x=(0, 0.02)
cc = [[0, 0, 0], [.05, 0.01, -0.02], [.1, 0.03, 0.01], [.15 + x, 0.07 + x, 0.055 + x]]
# 1 turning point with varying magnitude & twisting point (2 points are far) x=(0, 0.03)

# Cubic (total = 588*2+840*2)
aaa = [[0, 0.03 + x, 0], [.08, 0, 0], [0.14, 0, 0],
       [0.2, -0.03 - x_2, 0]]  # 2 turning points with varying magnitude & distance between each other x=(0, 0.05)=x_2
bbb = [[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]]  # Above aaa + twisting in the middle
ccc = [[0, 0, 0.03 + x], [.08, 0.03 + x_2, 0], [0.14, 0, 0],
       [0.2, -0.03 + x_2, -0.03 - x]]  # x=(0, 0.02), x_2=(0, 0.015)
ddd = [[0, 0, 0.03 - x_2], [.08, 0.04 + x, 0], [0.14, 0 - x, 0],
       [0.2 + x, -0.04, -0.03 - x_2]]  # x=(0, 0.015), x_2=(0, 0.02)


def write_info(transmat, f_name, path='D:/3d_match/raw_data/7-scenes-chess'):
    header = [0, 0, 0, 0]
    np.savetxt(os.path.join(path, f'{f_name}.info.txt'), np.asarray([header] + list(transmat)))


def load_info(f_name, path='D:/3d_match/raw_data/7-scenes-chess'):
    return np.genfromtxt(os.path.join(path, f'{f_name}.info.txt'), skip_header=1)


if __name__ == '__main__':
    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
    width = .005
    thickness = .0015
    folder_name = 'D:/3d_match_plate/'

    pseq = utl.cubic_inp(pseq=np.asarray([[0, 0, 0], [.018, .03, .02], [.06, .06, 0], [.12, 0, 0]]))

    # pseq = gen_sgl_curve(pseq=np.asarray([[0, 0, 0], [.018, .03, 0], [.06, .06, 0], [.12, 0, 0]]))
    rotseq = utl.get_rotseq_by_pseq(pseq)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    objcm = utl.gen_swap(pseq, rotseq, cross_sec)
    objcm.attach_to(base)

    '''
    gen data
    '''
    cnt = 0
    obj_id = 0
    rot_center = (0, 0, 0)
    homomat4_dict = dict()
    homomat4_dict[str(obj_id)] = {}

    icomats = rm.gen_icorotmats(rotation_interval=np.radians(90))
    for i, mats in enumerate(icomats):
        for j, rot in enumerate(mats):
            utl.get_objpcd_partial_o3d(objcm, rot, rot_center, path=folder_name,
                                       f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
                                       add_noise=False, add_occ=True, toggledebug=False)
            homomat4_dict[str(obj_id)][str(cnt).zfill(3)] = rm.homomat_from_posrot(rot_center, rot)
            cnt += 1
            pickle.dump(homomat4_dict, open(f'{folder_name}/homomat4_dict.pkl', 'wb'))
