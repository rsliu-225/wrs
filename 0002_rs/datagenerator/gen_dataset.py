import os.path
import random

import numpy

from _tst_gen_dataset import *
import utils as utl
from multiview import *
import open3d as o3d
import pickle
from multiprocessing import Process


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


# Name Composition: ObjectID_RotationID_ClassName_Type
def init_gen(class_name, num_kpts, max, res=(550, 550), rot_center=(0, 0, 0), max_num=10, length=.2, path='./'):
    path = os.path.join(path, class_name[:4])
    cache_data = dict()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))

    for i, mats in enumerate(icomats):
        print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
        kpts = random_kts(num_kpts, max=max)
        objcm = gen_seed(kpts, n=100, length=length, random=False)
        rot_dict = dict()
        np.random.shuffle(mats)
        for j, rot in enumerate(mats[:max_num]):
            f_name = '_'.join([str(i), str(j), class_name])
            utl.get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), f_name=f_name, path=path,
                                       resolusion=res, add_occ=True, add_noise=True, occ_vt_ratio=random.uniform(.5, 1),
                                       noise_vt_ration=random.uniform(.5, 1))
            utl.save_complete_pcd(f_name, utl.cm2o3dmesh(gen_seed(kpts, n=100, thickness=0)),
                                  path=path, method='possion', smp_num=2048)
            rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)

        cache_data[i] = rot_dict

        if i == len(icomats) - 1:
            print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            print(f"A total of {i + 1} different objects created")

    pickle.dump(cache_data, open(f'partial/{class_name}.pkl', 'wb'))
    cal_stats("partial", class_name)


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


def test_pcd(class_name):
    comeplte_dir = os.path.join(class_name, "complete")
    for complete_file in os.listdir(comeplte_dir):
        class_name = class_name + "1"
        name_com = complete_file.split("_")
        if class_name == name_com[2]:
            print(os.path.join(class_name, "partial", complete_file.replace("complete", "partial")))
            partial_pcd = utl.read_pcd(
                os.path.join(class_name, "partial", complete_file.replace("complete", "partial")))
            comp_pcd = utl.read_pcd(os.path.join(comeplte_dir, complete_file))
            output = np.append(partial_pcd, comp_pcd, axis=0)
            utl.show_pcd_pts(output)


def gen_args(cat, rng):
    if cat == 'quad':
        return [[cat + str(i + 1), 3, random.choice([.01, .02, .03, .04])] for i in range(rng[0], rng[1])]
    elif cat == 'bspl':
        return [[cat + str(i + 1), 4, random.choice([.01, .02, .03, .04])] for i in range(rng[0], rng[1])]


if __name__ == '__main__':
    args_list = gen_args("quad", (0, 8))
    print(args_list)
    args_list = gen_args("bspl", (0, 8))
    print(args_list)

    runInParallel(init_gen, gen_args("quad", (0, 8)))
    runInParallel(init_gen, gen_args("bspl", (0, 8)))
