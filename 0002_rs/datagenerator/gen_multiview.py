import copy
import itertools
import os
import pickle
import random

import numpy as np
import open3d as o3d

import datagenerator.data_utils as utl
import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as wd
from multiprocessing import Process

COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


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


def gen_multiview(cat, comb_num=1, path='', trans_diff=(.01, .01, .01), rot_diff=np.radians((10, 10, 10)),
                  add_occ=True, overwrite=False, toggledebug=False):
    if not os.path.exists(os.path.join(path, 'multiview/')):
        os.mkdir(os.path.join(path, 'multiview'))
        os.mkdir(os.path.join(path, 'multiview/partial'))
        os.mkdir(os.path.join(path, 'multiview/complete'))
    icomats = rm.gen_icorotmats(rotation_interval=np.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    for f in os.listdir(os.path.join(path, cat)):
        if f[-3:] == 'pcd':
            print(cat, f.split('_'))
            f_org = f'{f.split("_")[0]}_{f.split("_")[1]}.pcd'
            # os.remove(os.path.join(path, cat, 'complete', f_org))
            os.remove(os.path.join(path, cat, 'partial', f_org))
            os.remove(os.path.join(path, cat, f))
            continue
    f_name_dict = {}
    for f in os.listdir(os.path.join(path, cat, 'partial')):
        if f[-3:] != 'pcd':
            continue
        objid = f.split('_')[0]
        rid = int(f.split('_')[1].split('.pcd')[0])
        if objid not in f_name_dict.keys():
            f_name_dict[objid] = [rid]
        else:
            f_name_dict[objid].append(rid)
    cnt = 0
    for f in os.listdir(os.path.join(path, cat, 'partial')):
        cnt += 1
        if cnt % 100 == 0:
            print(printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                                   prefix=f'Progress({cat}):', suffix='Complete', length=100), "\r")
        if f[-3:] != 'pcd':
            continue

        objid = f.split('_')[0]
        rid_init = f.split('_')[1].split('.pcd')[0]
        f_name = f"{objid}_{rid_init}"
        if os.path.exists(os.path.join(os.path.join(path, 'multiview', 'complete', f'{cat}_{f_name}.pcd'))) \
                and not overwrite:
            continue
        init_homomat = rm.homomat_from_posrot((0, 0, 0), icomats[int(rid_init)])
        o3dpcd = o3d.io.read_point_cloud(f"{path}/{cat}/partial/{f_name}.pcd")
        pcd_mv = o3dpcd.points
        view_num = random.randint(2, 4)
        combs = random.choices(
            list(itertools.combinations([i for i in f_name_dict[objid] if i != rid_init], view_num)), k=comb_num)
        # print(objid, rid_init, combs)
        for i, comb in enumerate(combs):
            for rid in comb:
                homomat4 = rm.homomat_from_posrot((0, 0, 0), icomats[int(rid)])
                random_homomat4 = utl.gen_random_homomat4(trans_diff, rot_diff)
                o3dpcd = o3d.io.read_point_cloud(f"{path}/{cat}/partial/{objid}_{str(rid).zfill(4)}.pcd")
                pcd = np.asarray(o3dpcd.points)
                pcd = utl.trans_pcd(pcd, np.dot(init_homomat, np.linalg.inv(homomat4)))
                pcd = utl.trans_pcd(pcd, random_homomat4)
                if add_occ:
                    pcd = utl.add_random_occ_narry(pcd, occ_ratio_rng=(.3, .5))
                pcd_mv.extend(pcd)
            o3dpcd_mv = utl.nparray2o3dpcd(np.asarray(pcd_mv))
            o3dpcd_mv = utl.resample(o3dpcd_mv, smp_num=2048)
            o3dpcd_gt = o3d.io.read_point_cloud(f"{path}/{cat}/complete/{f_name}.pcd")
            o3d.io.write_point_cloud(os.path.join(path, 'multiview', 'partial', f'{cat}_{f_name}.pcd'), o3dpcd_mv)
            o3d.io.write_point_cloud(os.path.join(path, 'multiview', 'complete', f'{cat}_{f_name}.pcd'),
                                     o3dpcd_gt)

            if toggledebug:
                o3dpcd_mv_init = o3d.io.read_point_cloud(f"{path}/{cat}/partial/{f_name}.pcd")
                o3dpcd_gt = o3d.io.read_point_cloud(
                    os.path.join(path, 'multiview', 'complete', f'{cat}_{f_name}.pcd'))
                o3dpcd_mv = o3d.io.read_point_cloud(
                    os.path.join(path, 'multiview', 'partial', f'{cat}_{f_name}.pcd'))
                o3dpcd_mv_init.paint_uniform_color(COLOR[0])
                o3dpcd_gt.paint_uniform_color(COLOR[1])
                o3dpcd_mv.paint_uniform_color([.7, .7, .7])
                o3d.visualization.draw_geometries([o3dpcd_gt])
                o3d.visualization.draw_geometries([o3dpcd_mv, o3dpcd_mv_init])
    print(printProgressBar(cnt, len(os.listdir(os.path.join(path, cat, 'partial'))),
                           prefix=f'Progress({cat}):', suffix='Finished!', length=100), "\r")

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
                pcd = utl.read_pcd(file_path)
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


def show(fo='./', cat='bspl'):
    random_f = random.choices(sorted(os.listdir(os.path.join(fo, cat, 'complete'))), k=10)
    for f in random_f:
        if f[-3:] == 'pcd':
            o3dpcd = o3d.io.read_point_cloud(os.path.join(fo, cat, 'complete', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[0, 1, 0, 1]]).attach_to(base)
            o3dpcd = o3d.io.read_point_cloud(os.path.join(fo, cat, 'partial', f))
            gm.gen_pointcloud(o3dpcd.points, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()


if __name__ == '__main__':
    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[.1, .4, 0], lookat_pos=[.1, 0, 0])
    path = 'E:/liu/org_data/dataset_prim'
    trans_diff = (.001, .001, .001)
    rot_diff = np.radians((1, 1, 1))
    comb_num = 1

    cat_list = []
    for fo in os.listdir(path):
        cat_list.append(fo)
    # cat_list = ['bspl']

    # gen_multiview('tmpl', comb_num=comb_num, path=path, trans_diff=trans_diff, rot_diff=rot_diff,
    #               add_occ=True, toggledebug=True)
    proc = []
    for cat in cat_list:
        if cat != 'multiview':
            p = Process(target=gen_multiview, args=(cat, comb_num, path, trans_diff, rot_diff, True, False, False))
            p.start()
            proc.append(p)
    for p in proc:
        p.join()

    show(path, cat='multiview')
