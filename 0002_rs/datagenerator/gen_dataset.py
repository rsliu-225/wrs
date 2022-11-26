import math
import os
import os.path
import random
from multiprocessing import Process

import numpy as np
import open3d as o3d

import basis.robot_math as rm
import data_utils as utl
import modeling.collision_model as cm

# PATH = 'E:/liu/dataset_flat/'
# PATH = 'E:/liu/org_data/dataset_prim/'
PATH = 'E:/liu/org_data/dataset_1/'
# include visible threshold

RND_OCC_RADIO_RNG = (0, .4)
VISIBLE_THRESHOLD = np.radians(80)


def runInParallel(fn, args):
    proc = []
    for arg in args:
        p = Process(target=fn, args=arg)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


def init_gen(cat, num_kpts, max_kts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, length=.2, path=PATH):
    path = os.path.join(path, cat[:4])
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    rotid_list = list(range(len(icomats)))
    random.shuffle(rotid_list)
    objcm, objcm_flat, _, _ = utl.gen_seed(num_kpts, max=max_kts, n=100, length=length, rand_wd=False)
    cnt = 0
    for i in rotid_list:
        if cnt % 10 == 0:
            print(printProgressBar(cnt, max_num, prefix='Progress:', suffix='Complete', length=100), "\r")
        f_name = '_'.join([cat[4:].zfill(4), str(i).zfill(4)])
        flag = utl.get_objpcd_partial_o3d(objcm, objcm_flat, icomats[i], rot_center, pseq=None, rotseq=None,
                                          f_name=f_name, path=path, resolusion=res,
                                          add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                          rnd_occ_ratio_rng=RND_OCC_RADIO_RNG, visible_threshold=VISIBLE_THRESHOLD,
                                          occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1))
        if flag:
            cnt += 1
        else:
            print('Failed:', f_name)
            os.remove(os.path.join(path, f_name + f'_tmp.pcd'))
            os.remove(os.path.join(path, 'partial', f_name + f'.pcd'))
        if cnt == max_num:
            print(printProgressBar(cnt, max_num, prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {cnt} different objects created")
            break

    cal_stats(os.path.join(path, 'partial'), cat)


def init_gen_deform(cat, num_kpts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, path=PATH):
    path = os.path.join(path, cat[:4])
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    rotid_list = list(range(len(icomats)))
    random.shuffle(rotid_list)

    if cat[:4] == 'tmpl':
        objcm = cm.CollisionModel(f'../obstacles/template.stl')
    else:
        objcm = cm.CollisionModel(f'../obstacles/plate.stl')

    if num_kpts == 3:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.08 + random.uniform(-.02, .02), 0, random.uniform(.015, .02)],
                                [.16, 0, 0]])
    elif num_kpts == 4:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .04), 0, random.uniform(-.015, .015)],
                                [.12 + random.uniform(-.04, .02), 0, random.uniform(-.015, .015)],
                                [.16, 0, 0]])
    else:
        goal_pseq = np.asarray([[0, 0, 0],
                                [.04 + random.uniform(-.02, .01), 0, random.uniform(-.005, .005)],
                                [.08 + random.uniform(-.01, .01), 0, random.uniform(-.015, .015)],
                                [.12 + random.uniform(-.01, .02), 0, random.uniform(-.005, .005)],
                                [.16, 0, 0]])
    rot_axial, rot_radial = utl.random_rot_radians(num_kpts)
    rbf_radius = random.uniform(.05, .2)
    deformed_objcm, objcm_gt, kpts, kpts_rotseq = \
        utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=rbf_radius)
    cnt = 0
    for i in rotid_list:
        if cnt % 10 == 0:
            print(printProgressBar(cnt, max_num, prefix='Progress:', suffix='Complete', length=100), "\r")
        f_name = '_'.join([cat[4:].zfill(4), str(i).zfill(4)])
        flag = utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, icomats[i], rot_center, pseq=kpts,
                                          rotseq=kpts_rotseq, f_name=f_name, path=path, resolusion=res,
                                          add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                          rnd_occ_ratio_rng=RND_OCC_RADIO_RNG, visible_threshold=VISIBLE_THRESHOLD,
                                          occ_vt_ratio=random.uniform(.05, .08), noise_vt_ratio=random.uniform(.2, .5))
        if flag:
            cnt += 1
        else:
            print('Failed:', f_name)
            os.remove(os.path.join(path, f_name + f'_tmp.pcd'))
            os.remove(os.path.join(path, 'partial', f_name + f'.pcd'))
        if cnt == max_num:
            print(printProgressBar(cnt, max_num, prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {cnt} different objects created")
            break
    cal_stats(os.path.join(path, 'partial'), cat)


def cal_stats(path, class_name):
    num_pts = []
    class_name = str(class_name[4:]).zfill(4)
    for f in os.listdir(path):
        if not "pcd" in f or f[:4] != class_name:
            continue
        num_pts.append(len(utl.read_pcd(os.path.join(path, f))))
    print("[lowest:highest:average]", min(num_pts), max(num_pts), int(sum(num_pts) / len(num_pts)),
          ": with a total of", len(num_pts), "pcd")
    return len(num_pts)


def show_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])
    return len(pcd.points)


def show_all_pcd(folder_path, class_name):
    for file in os.listdir(folder_path):
        if not "pcd" in file:
            continue
        show_pcd(os.path.join(folder_path, file))
    cal_stats(folder_path, class_name)


def show_some_pcd(path, num, class_name, num_scan, dist=0.01):
    total = cal_stats(path, class_name)
    ids = set(np.random.permutation(total)[:num].tolist())
    print("Random IDs:", ids)
    pcd_np = None
    cnt = 1
    for file in os.listdir(path):
        if "pcd" in file and class_name in file and utl.get_uniq_id(file, num_scan) in ids:
            print(file)
            file_path = os.path.join(path, file)
            if cnt == 1:
                pcd_np = utl.read_pcd(file_path)
                cnt += 1
                continue
            cur_pcd = utl.read_pcd(file_path) + dist * cnt
            pcd_np = np.append(pcd_np, cur_pcd, axis=0)
            cnt += 1
    utl.show_pcd_pts(pcd_np)


def test_pcd(class_name, id='1'):
    gt_dir = os.path.join(PATH, class_name, "complete")
    for f in os.listdir(gt_dir):
        name_com = (f.split('.')[0]).split("_")
        if class_name + id == name_com[2]:
            o3dpcd_i = utl.read_o3dpcd(os.path.join(gt_dir.replace("complete", "partial"), f))
            o3dpcd_gt = utl.read_o3dpcd(os.path.join(gt_dir, f))
            o3dpcd_gt.paint_uniform_color([0, 1, 0])
            o3dpcd_i.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_gt])
            o3d.visualization.draw_geometries([o3dpcd_i])


def gen_args(cat, rng):
    if cat == 'quad':
        args = [[cat + str(i), 3, random.choice([.01, .02, .03, .04])] for i in rng]
    elif cat == 'bspl':
        args = [[cat + str(i), random.choice([4, 5]), random.choice([.01, .02, .03, .04])] for i in rng]
    elif cat == 'sprl':
        args = [[cat + str(i), 20, random.choice([.04, .05])] for i in rng]
    else:
        args = None
    print(args)
    return args


def gen_args_deform(cat, rng):
    args = [[cat + str(i), random.choice([3, 4, 5])] for i in rng]
    print(args)
    return args


def remove_tmp(cat_list, path):
    for cat in cat_list:
        if not os.path.exists(os.path.join(path, cat)):
            continue
        for f in os.listdir(os.path.join(path, cat)):
            if f[-3:] == 'pcd':
                print(cat, f.split('_'))
                f_org = f'{f.split("_")[0]}_{f.split("_")[1]}.pcd'
                # os.remove(os.path.join(path, cat, 'complete', f_org))
                os.remove(os.path.join(path, cat, 'partial', f_org))
                os.remove(os.path.join(path, cat, f))
                continue


if __name__ == '__main__':
    # init_gen('bspl', 4, .02, rot_center=(0, 0, 0), max_num=10, length=.2)
    # init_gen_deform('plat', 4, rot_center=(0, 0, 0), max_num=10)

    start = 0
    end = 100
    for i in range(start, end):
        runInParallel(init_gen, gen_args("bspl", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen, gen_args("quad", range(i * 8, (i + 1) * 8)))
    # for i in range(start, end):
    #     runInParallel(init_gen, gen_args("sprl", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen_deform, gen_args_deform("plat", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen_deform, gen_args_deform("tmpl", range(i * 8, (i + 1) * 8)))

    remove_tmp(["bspl", "quad", "sprl", "plat", "tmpl"], PATH)
