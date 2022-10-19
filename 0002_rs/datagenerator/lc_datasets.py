import os.path

import numpy

from _tst_gen_dataset import *
import data_utils as utl
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


def gen_seed(input, kind="cubic", random=True):
    # Width & Thickness of the stick
    width = .008  # + (np.random.uniform(0, 0.005) if random else 0)
    thickness = .0015
    length = .2
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    pseq = utl.uni_length(utl.poly_inp(step=.001, kind=kind, pseq=np.asarray(input)), goal_len=length)
    return utl.gen_swap(pseq, utl.get_rotseq_by_pseq(pseq), cross_sec)


def test_seed(seed):
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360))
    for i, mats in enumerate(icomats):
        for j, rot in enumerate(mats):
            objcm = gen_seed(seed, random=False)
            utl.get_objpcd_partial_o3d(objcm, objcm, rot, (0, 0, 0), path='test', f_name='_'.join(["test"]),
                                       resolusion=(550, 550), add_noise=False, add_occ=True)
            break
        break
    show_pcd(os.path.join("complete", '_'.join(["test", "complete.pcd"])))
    # show_pcd(os.path.join("test", '_'.join(["test", "partial.pcd"])))


# Name Composition: ObjectID_RotationID_ClassName_Type
def init_gen(factor, class_name, seed, res=(550, 550), rot_center=(0, 0, 0), max_num=10):
    cache_data = dict()
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / factor))

    if seed == 0:
        for i, mats in enumerate(icomats):
            print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            x = random.uniform(0, 0.05)
            objcm = gen_seed([[0, 0.03 + x, 0], [.08, 0, 0], [0.14, 0, 0], [0.2, -0.03 - x, 0]])

            rot_dict = dict()
            np.random.shuffle(mats)
            for j, rot in enumerate(mats[:max_num]):
                utl.get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), f_name='_'.join([str(i), str(j), class_name]),
                                           resolusion=res, add_occ=True, add_noise=True,
                                           occ_vt_ratio=random.uniform(.5, 1),
                                           noise_vt_ratio=random.uniform(.5, 1), )
                rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)

            cache_data[i] = rot_dict

            if i == len(icomats) - 1:
                print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
                print(f"A total of {i + 1} different objects created")

        pickle.dump(cache_data, open(f'partial/{class_name}.pkl', 'wb'))
        cal_stats("partial", class_name)
    elif seed == 1:
        for i, mats in enumerate(icomats):
            print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            x = random.uniform(-0.02, 0.02)
            objcm = gen_seed([[0, 0, 0], [0.07, 0.05, 0.05 + x], [0.14, 0.15, -0.05 + x], [.2, 0.3, 0]])

            rot_dict = dict()
            np.random.shuffle(mats)
            for j, rot in enumerate(mats[:max_num]):
                utl.get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), f_name='_'.join([str(i), str(j), class_name]),
                                           resolusion=res, add_occ=True, add_noise=True,
                                           occ_vt_ratio=random.uniform(.5, 1),
                                           noise_vt_ratio=random.uniform(.5, 1), )
                rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)

            cache_data[i] = rot_dict

            if i == len(icomats) - 1:
                print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
                print(f"A total of {i + 1} different objects created")

        pickle.dump(cache_data, open(f'partial/{class_name}.pkl', 'wb'))
        cal_stats("partial", class_name)
    elif seed == 2:
        for i, mats in enumerate(icomats):
            print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            x, x_2 = random.uniform(0, 0.02), random.uniform(0, 0.015)
            objcm = gen_seed([[0, 0, 0.03 + x], [.08, 0.03 + x_2, 0], [0.14, 0, 0], [0.2, -0.03 + x_2, -0.03 - x]])

            rot_dict = dict()
            np.random.shuffle(mats)
            for j, rot in enumerate(mats[:max_num]):
                utl.get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), f_name='_'.join([str(i), str(j), class_name]),
                                           resolusion=res, add_occ=True, add_noise=True,
                                           occ_vt_ratio=random.uniform(.5, 1),
                                           noise_vt_ratio=random.uniform(.5, 1), )
                rot_dict[j] = rm.homomat_from_posrot(rot_center, rot)

            cache_data[i] = rot_dict

            if i == len(icomats) - 1:
                print(printProgressBar(i + 1, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
                print(f"A total of {i + 1} different objects created")

        pickle.dump(cache_data, open(f'partial/{class_name}.pkl', 'wb'))
        cal_stats("partial", class_name)
    elif seed == 3:
        for i, mats in enumerate(icomats):
            print(printProgressBar(i, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
            x, x_2 = random.uniform(0, 0.015), random.uniform(0, 0.02)
            objcm = gen_seed([[0, 0, 0.03 - x_2], [.08, 0.04 + x, 0], [0.14, 0 - x, 0], [0.2 + x, -0.04, -0.03 - x_2]])

            rot_dict = dict()
            np.random.shuffle(mats)
            for j, rot in enumerate(mats[:max_num]):
                utl.get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), f_name='_'.join([str(i), str(j), class_name]),
                                           resolusion=res, add_occ=True, add_noise=True,
                                           occ_vt_ratio=random.uniform(.5, 1),
                                           noise_vt_ratio=random.uniform(.5, 1), )
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


def test_pcd():
    comeplte_dir = os.path.join("cubic", "complete")
    for complete_file in os.listdir(comeplte_dir):
        class_name = "cubic1"
        name_com = complete_file.split("_")
        if class_name == name_com[2]:
            print(os.path.join("cubic", "partial", complete_file.replace("complete", "partial")))
            partial_pcd = utl.read_pcd(os.path.join("cubic", "partial", complete_file.replace("complete", "partial")))
            comp_pcd = utl.read_pcd(os.path.join(comeplte_dir, complete_file))
            output = np.append(partial_pcd, comp_pcd, axis=0)

            utl.show_pcd_pts(output)


def get_args(name, num_class, num_output):
    args_list = []
    for i in range(num_class):
        args_list.append([num_output, name, f"{name}{i + 1}"])
    return args_list


def gen_args(cat):
    return [[cat, cat + str(i + 1)] for i in range(16)]


def new_args(fact, cat, rng):
    # seed_list = [[[0, 0, 0], [.03, 0, 0.005], [.07, 0, 0.01], [.15, 0.01, 0.01]],
    #              [[0, 0.001, 0.01], [.08, 0, 0.01], [0.14, 0, 0.0096], [.15, 0, 0.0096]],
    #              [[0, 0.001, 0.001], [.08, 0, 0.001], [0.14, 0, 0.001], [.15, 0.001, 0.001]]]
    return [[fact, cat + str(i + 1), i % 4] for i in range(rng[0], rng[1])]


if __name__ == '__main__':
    # args_list = new_args(60, "cubic", (0, 8))
    # print(args_list)
    # args_list = new_args(60, "cubic", (8, 16))
    # print(args_list)

    runInParallel(init_gen, new_args(60, "cubic", (0, 8)))
    runInParallel(init_gen, new_args(60, "cubic", (8, 16)))

    # runInParallel(gen_multiview_for_complete_pcd, gen_args("quad"))
    # runInParallel(gen_multiview_for_complete_pcd, gen_args("linear"))
    # gen_multiview_for_complete_pcd("cubic", "cubic1")
    # test_pcd()
    # show_all_pcd("linear/multiview")
    # print(get_args("quad", 6, 3))
    # runInParallel(function, get_args("quad", 6, 2))

# test_seed([[0, 0, 0.03], [.08, 0.05, 0], [0.14, 0, 0], [0.2, -0.05, -0.03]])

# cal_stats("partial","linear1")
# show_some_pcd("./partial", 10, "linear1")

# show_some_pcd("linear/multiview", 4, "linear1", dist=0)
# show_all_pcd("linear/multiview")
# show_pcd("complete/1_linear1_complete.pcd")
# show_pcd("partial/1_linear1_partial.pcd")
# show_pcd("partial/1_linear1.pcd")

# init_gen(50, "cubic9")
# init_gen(50, "cubic13")

"""Record"""
x, x_2 = 0, 0
# Linear(total = 2*3*42*20)
a = [[0, 0, 0], [.03, 0, 0.005], [.07, 0, 0.01], [.15, 0.01, 0.01]]
# Twisting point near the ends
b = [[0, 0.001, 0.01], [.08, 0, 0.01], [0.14, 0, 0.0096], [.15, 0, 0.0096]]
# Twisting point near the middle
c = [[0, 0.001, 0.001], [.08, 0, 0.001], [0.14, 0, 0.001], [.15, 0.001, 0.001]]
# Flat

# Quadratic (total = 2*3*42*20)
aa = [[0, 0, 0], [.15, 0.005, 0.035], [.2, 0.01, 0]]
# 1 turning point with varying magnitude
bb = [[0, 0, 0], [.05, 0.01, 0], [.1, 0.03, 0.015], [.15 + x, 0.06, 0.025 + x * 3]]
# 1 turning point with varying magnitude & twisting point (2 points are near) x=(0, 0.02)
cc = [[0, 0, 0], [.05, 0.01, -0.02], [.1, 0.03, 0.01], [.15 + x, 0.07 + x, 0.055 + x]]
# 1 turning point with varying magnitude & twisting point (2 points are far) x=(0, 0.03)

# Cubic (total = 4*4*42*20)
aaa = [[0, 0.03 + x, 0], [.08, 0, 0], [0.14, 0, 0], [0.2, -0.03 - x, 0]]
# 2 turning points with varying magnitude & distance between each other x=(0, 0.05)
bbb = [[0, 0, 0], [0.07, 0.05, 0.05 + x], [0.14, 0.15, -0.05 + x], [.2, 0.3, 0]]
# Above aaa + twisting in the middle x=(-0.02, 0.02)
ccc = [[0, 0, 0.03 + x], [.08, 0.03 + x_2, 0], [0.14, 0, 0], [0.2, -0.03 + x_2, -0.03 - x]]
# x=(0, 0.02), x_2=(0, 0.015)
ddd = [[0, 0, 0.03 - x_2], [.08, 0.04 + x, 0], [0.14, 0 - x, 0], [0.2 + x, -0.04, -0.03 - x_2]]
# x=(0, 0.015), x_2=(0, 0.02)
# [[0, 0.03 + np.random.uniform(-0.01, 0.05), 0], [.08, 0, 0], [0.14, 0, 0], [0.2, -0.03 - np.random.uniform(-0.01, 0.05), 0]]
