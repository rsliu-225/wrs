import os.path
import random

import numpy
import numpy as np

from _tst_gen_dataset import *
import data_utils as utl
from gen_multiview import *
import open3d as o3d
import pickle
from multiprocessing import Process

# PATH = 'E:/liu/dataset_2048_flat/'
PATH = 'E:/liu/dataset_2048_prim_v10/'


def runInParallel(fn, args):
    proc = []
    for arg in args:
        p = Process(target=fn, args=arg)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def function(num_output, name, class_name):
    gen_multiview_lc(comb_num=num_output, cat=name, class_name=class_name)


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


def random_kts(n=3, max=.02):
    kpts = [(0, 0, 0)]
    for j in range(n - 1):
        kpts.append(((j + 1) * .02, random.uniform(-max, max), random.uniform(-max, max)))
    return kpts


def random_rot_radians(n=3):
    rot_axial = []
    rot_radial = []
    for i in range(n):
        rot_axial.append(random.randint(10, 30) * random.choice([1, -1]))
        rot_radial.append(random.randint(0, 1) * random.choice([1, -1]))
    return np.radians(rot_axial), np.radians(rot_radial)


def gen_seed(num_kpts=4, max=.02, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, rand_wid=False):
    width = width + (np.random.uniform(0, 0.005) if rand_wid else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    flat_sec = [[0, width / 2], [0, -width / 2], [0, -width / 2], [0, width / 2]]
    success = False
    pseq, rotseq = [], []
    while not success:
        kpts = random_kts(num_kpts, max=max)
        if len(kpts) == 3:
            pseq = utl.uni_length(utl.poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
        # elif len(kpts) == 4:
        #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
        else:
            pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
        pseq = np.asarray(pseq) - pseq[0]
        pseq, rotseq = utl.get_rotseq_by_pseq_smooth(pseq)
        for i in range(len(rotseq) - 1):
            if rm.angle_between_vectors(rotseq[i][:, 2], rotseq[i + 1][:, 2]) > np.pi / 15:
                success = False
                break
            success = True
    return utl.gen_swap(pseq, rotseq, cross_sec), utl.gen_swap(pseq, rotseq, flat_sec)


# Name Composition: ObjectID_RotationID_ClassName_Type
# def init_gen(cat, num_kpts, max_kts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, length=.2, path=PATH):
#     path = os.path.join(path, cat[:4])
#     cache_data = dict()
#     icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
#
#     for i, mats in enumerate(icomats):
#         print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
#         kpts = random_kts(num_kpts, max=max_kts)
#         objcm = gen_seed_smooth(kpts, n=100, length=length, rand_wid=False)
#         objcm_flat = gen_seed_smooth(kpts, n=100, thickness=0, rand_wid=False)
#         rot_dict = dict()
#         np.random.shuffle(mats)
#         for j, rot in enumerate(mats[:max_num]):
#             f_name = '_'.join([str(i), str(j), cat])
#             utl.get_objpcd_partial_o3d(objcm, objcm_flat, rot, (0, 0, 0), f_name=f_name, path=path,
#                                        resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True,
#                                        occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1))
#             rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)
#         cache_data[i] = rot_dict
#
#         if i == len(icomats) - 1:
#             print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
#             print(f"A total of {i + 1} different objects created")
#
#     pickle.dump(cache_data, open(os.path.join(path, f'partial/{cat}.pkl'), 'wb'))
#     cal_stats(os.path.join(path, 'partial'), cat)

def init_gen(cat, num_kpts, max_kts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, length=.2, path=PATH):
    path = os.path.join(path, cat[:4])
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    rotid_list = random.choices(range(len(icomats)), k=max_num)
    objcm, objcm_flat = gen_seed(num_kpts, max=max_kts, n=100, length=length, rand_wid=False)
    cnt = 0
    for i in rotid_list:
        if cnt % 10 == 0:
            print(printProgressBar(cnt, len(rotid_list), prefix='Progress:', suffix='Complete', length=100), "\r")
        f_name = '_'.join([cat[4:].zfill(4), str(i).zfill(4)])
        utl.get_objpcd_partial_o3d(objcm, objcm_flat, icomats[i], rot_center, f_name=f_name, path=path,
                                   resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                   occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1))
        if cnt - 1 == len(rotid_list):
            print(printProgressBar(cnt, len(rotid_list), prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {cnt} different objects created")
        cnt += 1

    cal_stats(os.path.join(path, 'partial'), cat)


# def init_gen_deform(class_name, num_kpts, res=(550, 550), rot_center=(0, 0, 0), max_num=5, path=PATH):
#     path = os.path.join(path, class_name[:4])
#     cache_data = dict()
#     icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
#     if class_name[:4] == 'tmpl':
#         objcm = cm.CollisionModel(f'../obstacles/template.stl')
#     else:
#         objcm = cm.CollisionModel(f'../obstacles/plate.stl')
#
#     if num_kpts == 3:
#         goal_pseq = np.asarray([[0, 0, 0],
#                                 [.08 + random.uniform(-.02, .02), 0, random.uniform(.015, .02)],
#                                 [.16, 0, 0]])
#     elif num_kpts == 4:
#         goal_pseq = np.asarray([[0, 0, 0],
#                                 [.04 + random.uniform(-.02, .04), 0, random.uniform(-.015, .015)],
#                                 [.12 + random.uniform(-.04, .02), 0, random.uniform(-.015, .015)],
#                                 [.16, 0, 0]])
#     else:
#         goal_pseq = np.asarray([[0, 0, 0],
#                                 [.04 + random.uniform(-.02, .01), 0, random.uniform(-.005, .005)],
#                                 [.08 + random.uniform(-.01, .01), 0, random.uniform(-.015, .015)],
#                                 [.12 + random.uniform(-.01, .02), 0, random.uniform(-.005, .005)],
#                                 [.16, 0, 0]])
#
#     for i, mats in enumerate(icomats):
#         print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
#         rot_axial, rot_radial = random_rot_radians(num_kpts)
#         deformed_objcm, objcm_gt = \
#             utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=random.uniform(.05, .2))
#
#         rot_dict = dict()
#         np.random.shuffle(mats)
#         for j, rot in enumerate(mats[:max_num]):
#             f_name = '_'.join([str(i), str(j), class_name])
#             utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, rot, (0, 0, 0), f_name=f_name, path=path,
#                                        resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True,
#                                        occ_vt_ratio=random.uniform(.05, .08), noise_vt_ratio=random.uniform(.2, .5))
#             rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)
#         cache_data[i] = rot_dict
#
#         if i == len(icomats) - 1:
#             print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
#             print(f"A total of {i + 1} different objects created")
#
#     pickle.dump(cache_data, open(os.path.join(path, f'partial/{class_name}.pkl'), 'wb'))
#     cal_stats(os.path.join(path, 'partial'), class_name)

def init_gen_deform(cat, num_kpts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, path=PATH):
    path = os.path.join(path, cat[:4])
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    icomats = [x for row in icomats for x in row]
    rotid_list = random.choices(range(len(icomats)), k=max_num)

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
    rot_axial, rot_radial = random_rot_radians(num_kpts)
    rbf_radius = random.uniform(.05, .2)
    deformed_objcm, objcm_gt = utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=rbf_radius)
    cnt = 0
    for i in rotid_list:
        if cnt % 10 == 0:
            print(printProgressBar(cnt, len(rotid_list), prefix='Progress:', suffix='Complete', length=100), "\r")
        f_name = '_'.join([cat[4:].zfill(4), str(i).zfill(4)])
        utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, icomats[i], rot_center, f_name=f_name, path=path,
                                   resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True, add_noise_pts=True,
                                   occ_vt_ratio=random.uniform(.05, .08), noise_vt_ratio=random.uniform(.2, .5))
        if cnt - 1 == len(rotid_list):
            print(printProgressBar(cnt, len(rotid_list), prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {cnt} different objects created")
        cnt += 1
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


def show_some_pcd(path, num, class_name, dist=0.01):
    total, num_scan = cal_stats(path, class_name)
    ids = set(np.random.permutation(total)[:num].tolist())
    print(num_scan, "Random IDs:", ids)
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
        args = [[cat + str(i), 4, random.choice([.01, .02, .03, .04])] for i in rng]
    else:
        args = None
    print(args)
    return args


def gen_args_deform(cat, rng):
    args = [[cat + str(i), random.choice([3, 4, 5])] for i in rng]
    print(args)
    return args


if __name__ == '__main__':
    start = 150
    end = 180
    for i in range(start, end):
        runInParallel(init_gen, gen_args("bspl", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen, gen_args("quad", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen_deform, gen_args_deform("plat", range(i * 8, (i + 1) * 8)))
    for i in range(start, end):
        runInParallel(init_gen_deform, gen_args_deform("tmpl", range(i * 8, (i + 1) * 8)))
