from gen_dataset import *
import visualization.panda.world as wd
import open3d as o3d

cam_pos = np.asarray([0, 0, .5])
base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


def gen_seed(input, random=True):
    # Width & Thickness of the stick
    width = .008 + (np.random.uniform(0, 0.005) if random else 0)
    thickness = .0015
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    # Magnitude of the turning points (for cubic)
    input[0][1] += (np.random.uniform(0, 0.05) if random else 0)
    input[-1][1] -= (np.random.uniform(0, 0.05) if random else 0)

    # Length of the stick
    input[-1][0] += (np.random.uniform(-0.05, 0.05) if random else 0)
    # print(input)
    pseq = gen_sgl_curve(step=.001, pseq=np.asarray(input))
    rotseq = get_rotseq_by_pseq(pseq)
    objcm = gen_swap(pseq, rotseq, cross_sec)
    return objcm


def init_gen(rng, fact=4.5, rot_center=(0, 0, 0), path="partial", res=(550, 550)):
    cnt = 0
    for x in rng:
        for y in rng:
            for z in rng:
                print(printProgressBar(cnt + 1, len(rng)**3, prefix='Progress:', suffix='Complete', length=100), "\r")
                cnt+=1
                rot = rm.rotmat_from_axangle((1, 0, 0), x * np.pi / fact) \
                    .dot(rm.rotmat_from_axangle((0, 1, 0), y * np.pi / fact)) \
                    .dot(rm.rotmat_from_axangle((0, 0, 1), z * np.pi / fact))
                objcm = gen_seed([[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]])
                mesh = get_objpcd_partial_o3d(objcm, rot, rot_center, path=path, f_name=''.join([str(x), str(y), str(z)]),
                                                   add_noise=False, add_occ=True, resolusion=res)
                save_complete_pcd(''.join([str(x), str(y), str(z)]), mesh)



def display(rng, show_complete=False, file_path=""):
    path_show = "complete" if show_complete else "partial"
    if file_path:
        path_show = file_path
    stats = [20000,0,0]
    for x in rng:
        for y in rng:
            for z in rng:
                o3dpcd = o3d.io.read_point_cloud(f"./{path_show}/{''.join([str(x), str(y), str(z)])}.pcd")
                no_pts = len(o3dpcd.points)

                if no_pts < stats[0]:
                    stats[0] = no_pts
                elif no_pts > stats[1]:
                    stats[1] = no_pts
                stats[2]+=no_pts

                gm.gen_pointcloud(o3dpcd.points, pntsize=5).attach_to(base)
    stats[2] /= len(rng)**3
    stats[2] = int(stats[2])
    print("[lowest:highest:average]", stats)
    base.run()

# Random display
def rand_display(path, complete=False):
    start = np.random.randint(0,5)
    rng = range(start, start+3)
    display(rng, complete, file_path=path)

# x = 0.05
# seed_arr =  [[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]]
# print(seed_arr)
# objcm = gen_seed(seed_arr, random=True)
# objcm.attach_to(base)
#
# for i in range(3):
#     seed_arr = [[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]]
#     print(seed_arr)
#     objcm = gen_seed(seed_arr, random=True)
#     objcm.attach_to(base)
# base.run()

# rng = list(range(9))
# init_gen(rng)
# display(rng, 0)
# display(list(range(2)))
# rand_display("./partial", 0)

display(list(range(1)), file_path="Training Dataset/cubic/2_partial")

"""Record"""
# Linear(total = 3*7*7*7)
x = 0
a = [[0, 0, 0], [.03, 0, 0.005], [.07, 0, 0.01], [.15, 0.01, 0.01]]                             # Twisting point near the ends
b = [[0, 0.001, 0.01], [.08, 0, 0.01], [0.14, 0, 0.0096], [.15, 0, 0.0096]]                     # Twisting point near the middle
c = [[0, 0.001, 0.001], [.08, 0, 0.001], [0.14, 0, 0.001], [.15, 0.001, 0.001]]                 # Flat

# Quadratic (total = 3*8*8*8)
aa = [[0, 0, 0], [.15, 0.005, 0.035],  [.2, 0.01, 0]]                                           # 1 turning point with varying magnitude
bb = [[0, 0, 0], [.05, 0.01, 0.01], [.1, 0.03, 0.015],  [.15, 0.06, 0.02], [.20, 0.1, 0.03]]    # 1 turning point with varying magnitude & twisting point (2 points are near) --- bb: ele[0]+x & ele[2]+x
cc = [[0, 0, 0], [.05, 0.01, -0.02], [.1, 0.03, 0.01],  [.15+x, 0.07+x, 0.055+x]]               # 1 turning point with varying magnitude & twisting point (2 points are far)

# Cubic (total = 2*9*9*9)
aaa = [[0, 0.03, 0], [.08, 0, 0], [0.14, 0, 0], [0.2, -0.03, 0]]                                # 2 turning points with varying magnitude & distance between each other
bbb = [[0, 0, 0], [0.07, 0.05, 0.05], [0.14, 0.15, -0.05], [.2, 0.3, 0]]                        # Above properties with twisting