import os.path
from datagenerator.gen_dataset import *
from datagenerator.utils import *
import open3d as o3d


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


def gen_seed(input, random=True):
    # Original Input
    # print("Original:", input)

    # Width & Thickness of the stick
    width = .008  # + (np.random.uniform(0, 0.005) if random else 0)
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    # Magnitude of the turning points (for cubic)
    input[0][-1] += (np.random.uniform(0, 0.02) if random else 0)
    input[-1][-1] -= (np.random.uniform(0, 0.02) if random else 0)

    # x = (np.random.uniform(0, 0.03) if random else 0)
    # input[-1][0] += x
    # input[-1][1] += x
    # input[-1][2] += x

    input[1][1] += (np.random.uniform(0, 0.015) if random else 0)
    input[2][1] -= (np.random.uniform(0, 0.015) if random else 0)

    # print(input)
    # Length of the stick
    input[-1][0] += (np.random.uniform(0, 0.015) if random else 0)  # defualt is 0.05

    # print(input)
    pseq = gen_sgl_curve(step=.001, pseq=np.asarray(input))
    rotseq = get_rotseq_by_pseq(pseq)
    objcm = gen_swap(pseq, rotseq, cross_sec)
    return objcm


def test_seed(seed):
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360))
    for i, mats in enumerate(icomats):
        for j, rot in enumerate(mats):
            objcm = gen_seed(seed, random=False)
            get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), path='test', f_name='_'.join(["test"]),
                                   resolusion=(550, 550), add_noise=False, add_occ=True)
            break
        break
    show_pcd(os.path.join("complete", '_'.join(["test", "complete.pcd"])))
    # show_pcd(os.path.join("test", '_'.join(["test", "partial.pcd"])))


def init_gen(factor, class_name, res=(550, 550), display_pcd=False, display_all_pcd=False):
    progress_cnt = 0
    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / factor))
    name_cnt = 0
    for i, mats in enumerate(icomats):
        progress_cnt += 1
        print(printProgressBar(progress_cnt, len(icomats), prefix='Progress:', suffix='Complete', length=100), "\r")
        for j, rot in enumerate(mats):
            objcm = gen_seed([[0, 0, 0.03], [.08, 0.04, 0], [0.14, 0, 0], [0.2, -0.04, -0.03]])
            get_objpcd_partial_o3d(objcm, rot, (0, 0, 0), path='partial', f_name='_'.join([str(name_cnt), class_name]),
                                   resolusion=res, add_noise=False, add_occ=True)
            if display_pcd:
                show_pcd(os.path.join("partial", '_'.join([str(name_cnt), class_name, "partial.pcd"])))
            name_cnt += 1

    cal_stats("partial")
    if display_all_pcd:
        show_all_pcd("partial")


def cal_stats(path):
    stats = [20000, 0, 0]
    cnt = 0
    for file in os.listdir(path):
        no_pts = len(o3d.io.read_point_cloud(os.path.join(path, file)).points)
        cnt += 1
        if no_pts < stats[0]:
            stats[0] = no_pts
        elif no_pts > stats[1]:
            stats[1] = no_pts
        stats[2] += no_pts
    stats[2] /= cnt
    stats[2] = int(stats[2])
    if "partial" in path:
        cnt //= 2
    print("[lowest:highest:average]", stats, ": with a total of", cnt, "pcd")


def show_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    o3d.visualization.draw_geometries([pcd])
    return len(pcd.points)


def show_all_pcd(folder_path):
    for file in os.listdir(folder_path):
        show_pcd(os.path.join(folder_path, file))
    cal_stats(folder_path)


# test_seed([[0, 0, 0.03], [.08, 0.05, 0], [0.14, 0, 0], [0.2, -0.05, -0.03]])

# init_gen(20, "cubic4", display_pcd=0)
# cal_stats("partial")
# show_all_pcd("complete")

# show_pcd("complete/1_linear1_complete.pcd")
# show_pcd("partial/1_linear1_partial.pcd")
# show_pcd("partial/1_linear1.pcd")
show_pcd(
    "lichuan/PF-Net-Point-Fractal-Network/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts")

"""Record"""
x, x_2 = 0, 0
# Linear(total = 3*588)
a = [[0, 0, 0], [.03, 0, 0.005], [.07, 0, 0.01], [.15, 0.01, 0.01]]  # Twisting point near the ends
b = [[0, 0.001, 0.01], [.08, 0, 0.01], [0.14, 0, 0.0096], [.15, 0, 0.0096]]  # Twisting point near the middle
c = [[0, 0.001, 0.001], [.08, 0, 0.001], [0.14, 0, 0.001], [.15, 0.001, 0.001]]  # Flat

# Quadratic (total = 3*588)
aa = [[0, 0, 0], [.15, 0.005, 0.035], [.2, 0.01, 0]]  # 1 turning point with varying magnitude
bb = [[0, 0, 0], [.05, 0.01, 0], [.1, 0.03, 0.015], [.15 + x, 0.06,
                                                     0.025 + x * 3]]  # 1 turning point with varying magnitude & twisting point (2 points are near) x=(0, 0.02)
cc = [[0, 0, 0], [.05, 0.01, -0.02], [.1, 0.03, 0.01], [.15 + x, 0.07 + x,
                                                        0.055 + x]]  # 1 turning point with varying magnitude & twisting point (2 points are far) x=(0, 0.03)

# Cubic (total = 588*2+840*2)
aaa = [[0, 0.03 + x, 0], [.08, 0, 0], [0.14, 0, 0],
       [0.2, -0.03 - x_2, 0]]  # 2 turning points with varying magnitude & distance between each other x=(0, 0.05)=x_2
bbb = [[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]]  # Above aaa + twisting in the middle
ccc = [[0, 0, 0.03 + x], [.08, 0.03 + x_2, 0], [0.14, 0, 0],
       [0.2, -0.03 + x_2, -0.03 - x]]  # x=(0, 0.02), x_2=(0, 0.015)
ddd = [[0, 0, 0.03 - x_2], [.08, 0.04 + x, 0], [0.14, 0 - x, 0],
       [0.2 + x, -0.04, -0.03 - x_2]]  # x=(0, 0.015), x_2=(0, 0.02)
