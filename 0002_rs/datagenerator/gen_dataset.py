import os.path
import random

import numpy

from _tst_gen_dataset import *
import data_utils as utl
from multiview import *
import open3d as o3d
import pickle
from multiprocessing import Process

PATH = 'E:/liu/dataset_2048_flat/'


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


def gen_seed(kpts, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, random=False):
    width = width + (np.random.uniform(0, 0.005) if random else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    if len(kpts) == 3:
        pseq = utl.uni_length(utl.poly_inp(step=length / n, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
    # elif len(kpts) == 4:
    #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
    else:
        pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
    pseq = np.asarray(pseq) - pseq[0]
    pseq, rotseq = utl.get_rotseq_by_pseq(pseq)
    return utl.gen_swap(pseq, rotseq, cross_sec)


def gen_seed_smooth(kpts, width=.008, length=.2, thickness=.0015, n=10, toggledebug=False, random=False):
    width = width + (np.random.uniform(0, 0.005) if random else 0)
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    if len(kpts) == 3:
        pseq = utl.uni_length(utl.poly_inp(step=.001, kind='quadratic', pseq=np.asarray(kpts)), goal_len=length)
    # elif len(kpts) == 4:
    #     pseq = utl.uni_length(utl.poly_inp(step=.001, kind='cubic', pseq=np.asarray(kpts)), goal_len=length)
    else:
        pseq = utl.uni_length(utl.spl_inp(pseq=np.asarray(kpts), n=n, toggledebug=toggledebug), goal_len=length)
    pseq = np.asarray(pseq) - pseq[0]
    pseq, rotseq = utl.get_rotseq_by_pseq_smooth(pseq)
    return utl.gen_swap(pseq, rotseq, cross_sec)


# Name Composition: ObjectID_RotationID_ClassName_Type
def init_gen(class_name, num_kpts, max_kts, res=(550, 550), rot_center=(0, 0, 0), max_num=10, length=.2, path=PATH):
    path = os.path.join(path, class_name[:4])
    cache_data = dict()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))

    for i, mats in enumerate(icomats):
        print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
        kpts = random_kts(num_kpts, max=max_kts)
        objcm = gen_seed_smooth(kpts, n=100, length=length, random=False)
        objcm_flat = gen_seed_smooth(kpts, n=100, thickness=0)
        rot_dict = dict()
        np.random.shuffle(mats)
        for j, rot in enumerate(mats[:max_num]):
            f_name = '_'.join([str(i), str(j), class_name])
            utl.get_objpcd_partial_o3d(objcm, objcm_flat, rot, (0, 0, 0), f_name=f_name, path=path,
                                       resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True,
                                       occ_vt_ratio=random.uniform(.5, 1), noise_vt_ratio=random.uniform(.5, 1))
            rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)
        cache_data[i] = rot_dict

        if i == len(icomats) - 1:
            print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {i + 1} different objects created")

    pickle.dump(cache_data, open(os.path.join(path, f'partial/{class_name}.pkl'), 'wb'))
    cal_stats(os.path.join(path, 'partial'), class_name)


def init_gen_deform(class_name, num_kpts, res=(550, 550), rot_center=(0, 0, 0), max_num=5, path=PATH):
    path = os.path.join(path, class_name[:4])
    cache_data = dict()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    if class_name[:4] == 'tmpl':
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

    for i, mats in enumerate(icomats):
        print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
        deformed_objcm, objcm_gt = \
            utl.deform_cm(objcm, goal_pseq, rot_axial, rot_radial, rbf_radius=random.uniform(.05, .2))

        rot_dict = dict()
        np.random.shuffle(mats)
        for j, rot in enumerate(mats[:max_num]):
            f_name = '_'.join([str(i), str(j), class_name])
            utl.get_objpcd_partial_o3d(deformed_objcm, objcm_gt, rot, (0, 0, 0), f_name=f_name, path=path,
                                       resolusion=res, add_occ=True, add_noise=True, add_rnd_occ=True,
                                       occ_vt_ratio=random.uniform(.05, .1), noise_vt_ratio=random.uniform(.2, .5))
            rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)
        cache_data[i] = rot_dict

        if i == len(icomats) - 1:
            print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {i + 1} different objects created")

    pickle.dump(cache_data, open(os.path.join(path, f'partial/{class_name}.pkl'), 'wb'))
    cal_stats(os.path.join(path, 'partial'), class_name)


def cal_stats(path, class_name):
    stats = [50000, 0, 0]
    cnt = 0
    num_scan = 1
    for file in os.listdir(path):
        if not "pcd" in file or not class_name in file:
            continue
        if int(file.split("_")[1]) + 1 > num_scan:
            num_scan = int(file.split("_")[1]) + 1
        no_pts = len(utl.read_pcd(os.path.join(path, file)))
        cnt += 1
        if no_pts < stats[0]:
            stats[0] = no_pts
        elif no_pts > stats[1]:
            stats[1] = no_pts
        stats[2] += no_pts
    stats[2] /= cnt
    stats[2] = int(stats[2])

    print("[lowest:highest:average]", stats, ": with a total of", cnt, "pcd")
    return cnt, num_scan


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
        print(f, name_com)
        if class_name + id == name_com[2]:
            o3dpcd_i = utl.read_o3dpcd(os.path.join(gt_dir.replace("complete", "partial"), f))
            o3dpcd_gt = utl.read_o3dpcd(os.path.join(gt_dir, f))
            o3dpcd_gt.paint_uniform_color([0, 1, 0])
            o3dpcd_i.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_gt])
            o3d.visualization.draw_geometries([o3dpcd_i])


def gen_args(cat, rng):
    if cat == 'quad':
        return [[cat + str(i + 1), 3, random.choice([.01, .02, .03, .04])] for i in range(rng[0], rng[1])]
    elif cat == 'bspl':
        return [[cat + str(i + 1), 4, random.choice([.01, .02, .03, .04])] for i in range(rng[0], rng[1])]


def gen_args_deform(cat, rng):
    args = [[cat + str(i + 1), random.choice([3, 4, 5])] for i in range(rng[0], rng[1])]
    print(args)
    return args


if __name__ == '__main__':
    # runInParallel(init_gen, gen_args("bspl", (0, 8)))
    # runInParallel(init_gen, gen_args("bspl", (24, 32)))
    # runInParallel(init_gen, gen_args("bspl", (32, 40)))
    # runInParallel(init_gen, gen_args("bspl", (104, 112)))
    # runInParallel(init_gen, gen_args("quad", (24, 25)))

    # runInParallel(init_gen_deform, gen_args_deform("tmpl", (0, 8)))
    # runInParallel(init_gen_deform, gen_args_deform("tmpl", (8, 16)))
    # runInParallel(init_gen_deform, gen_args_deform("tmpl", (16, 24)))
    # runInParallel(init_gen_deform, gen_args_deform("tmpl", (24, 32)))
    runInParallel(init_gen_deform, gen_args_deform("plat", (0, 8)))

    # test_pcd('plat', '1')
