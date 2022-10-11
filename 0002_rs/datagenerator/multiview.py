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

import utils

def gen_random_homomat4(trans_diff=(.01, .01, .01), rot_diff=np.radians((10, 10, 10))):
    random_pos = np.asarray([random.uniform(-trans_diff[0], trans_diff[0]),
                             random.uniform(-trans_diff[1], trans_diff[1]),
                             random.uniform(-trans_diff[2], trans_diff[2])])
    if rot_diff is None:
        random_rot = np.eye(3)
    else:
        random_rot = rm.rotmat_from_axangle((1, 0, 0), random.uniform(-rot_diff[0], rot_diff[0])).dot(
            rm.rotmat_from_axangle((0, 1, 0), random.uniform(-rot_diff[1], rot_diff[1]))).dot(
            rm.rotmat_from_axangle((0, 0, 1), random.uniform(-rot_diff[2], rot_diff[2])))
    return rm.homomat_from_posrot(random_pos, random_rot)


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


def gen_multiview_lc(comb_num=1, cat='', class_name=None, trans_diff=(.004, .004, .004),
                     rot_diff=np.radians((1, 1, 1))):
    view_num = np.random.randint(2, 5)
    if not os.path.exists(os.path.join(cat, 'multiview/')):
        os.mkdir(os.path.join(cat, 'multiview/'))

    homomat4_dict = pickle.load(open(f'{cat}/partial/{class_name}.pkl', 'rb'))
    cnt = 0
    for objid, rot_dict in homomat4_dict.items():
        print(objid)
        for rot_id in rot_dict:
            init_homomat = rot_dict[rot_id]
            for k in range(2):
                o3dpcd = o3d.io.read_point_cloud(f"{cat}/partial/{objid}_{rot_id}_{class_name}.pcd")
                pcd_mv = o3dpcd.points

                combs = random.choices(list(itertools.combinations([i for i in rot_dict if i != rot_id], view_num)),
                                       k=comb_num)
                # print(rot_id, " || ", combs)
                for i, comb in enumerate(combs):
                    for comb_rot_id in comb:
                        homomat4 = rot_dict[comb_rot_id]
                        random_homomat4 = gen_random_homomat4(trans_diff, rot_diff)

                        o3dpcd = o3d.io.read_point_cloud(
                            f"{cat}/partial/{objid}_{comb_rot_id}_{class_name}.pcd")
                        pcd = np.asarray(o3dpcd.points)
                        pcd = utl.trans_pcd(pcd, np.dot(init_homomat, np.linalg.inv(homomat4)))
                        pcd = utl.trans_pcd(pcd, random_homomat4)
                        pcd = utl.add_random_occ_narry(pcd, occ_ratio_rng=(.5, .6))
                        pcd_mv.extend(pcd)
                    o3d.io.write_point_cloud(
                        os.path.join(cat, 'multiview', f'{objid}_{rot_id}_{class_name}_{i}_multiview.pcd'),
                        utl.nparray2o3dpcd(pcd_mv))
                    cnt += 1
    print(f"A total of {cnt} pcd generated")

def gen_multiview_for_complete_pcd(cat='', class_name=None):
    homomat4_dict = pickle.load(open(f'{cat}/partial/{class_name}.pkl', 'rb'))
    cnt = 0
    complete_folder = os.path.join(cat, "complete")
    for complete_pcd in os.listdir(complete_folder):
        name_com = complete_pcd.split("_")
        if class_name == name_com[2]:
            # print(name_com)
            file_path = os.path.join(complete_folder, complete_pcd)
            obj_id = complete_pcd.split("_")[0]

            for rot_id, rot_mat in homomat4_dict[int(obj_id)].items():
                # print(type(rot_id), name_com)
                if rot_id == 0:
                    continue
                print(name_com)
                pcd = utils.read_pcd(file_path)
                pcd = utl.trans_pcd(pcd, np.linalg.inv(np.dot(homomat4_dict[int(obj_id)][0], np.linalg.inv(rot_mat))))

                new_file_name = complete_pcd.split("_")
                new_file_name[1] = str(rot_id)
                new_file_name = "_".join(new_file_name)
                new_path = os.path.join(complete_folder, new_file_name)
                o3d.io.write_point_cloud(new_path, utl.nparray2o3dpcd(pcd))
                print(new_file_name)
                # partial_pcd = utils.read_pcd(os.path.join(cat, "partial", new_file_name.replace("complete", "partial")))
                # comp_pcd = utils.read_pcd(new_path)
                # output = np.append(partial_pcd, comp_pcd, axis=0)
                # utils.show_pcd_pts(output)
                cnt += 1
    print(f"A total of {cnt} pcd generated")


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
