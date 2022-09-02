import os
import pickle

import numpy as np

import bendplanner.bend_utils as bu
import matplotlib.pyplot as plt
import bendplanner.BendSim as b_sim
import visualization.panda.world as wd


def load_res(num, fo='180'):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    success_cnt = 0
    for f in os.listdir(f'./{fo}'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f'./{fo}/{f}', 'rb'))
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])
            else:
                fail_tc_list.append(total_tc)
                # print(total_tc, f)
            if type(attemp_cnt_list) == type([]):
                total_cnt_list.append(attemp_cnt_list[-1])
            else:
                total_cnt_list.append(attemp_cnt_list)

    # print('Success Cnt:', success_cnt)
    # print('Fail time cost:', np.average(fail_tc_list))
    # print('Top 10 time cost:', np.average(success_tc_list))
    # print('First time cost:', np.average(success_first_tc_list), success_first_tc_list)
    # print('Num. of solution:', np.average(success_cnt_list), success_cnt_list)
    # print('Num. of attempts:', np.average(total_cnt_list))

    return success_cnt, fail_tc_list, success_tc_list[:10], success_first_tc_list[:10], total_cnt_list[:10]


def tst(bs, num, fo='180'):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    success_cnt = 0
    for f in os.listdir(f'./{fo}'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            print(f'=========={f}==========')
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f'./{fo}/{f}', 'rb'))
            bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
            print(tc_list)
            print(attemp_cnt_list)
            print(total_tc)
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])
                print(result)
                try:
                    bendresseq, seqs = result[1]
                except:
                    seqs = result[1]
                print(seqs)
                bendseq = [bendset[i] for i in seqs]
                is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=True, prune=True, toggledebug=False)
                _, _, _, _, _, pseq, _ = bendresseq[-1]
                pseq = np.asarray(pseq)
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .5
                ax = plt.axes(projection='3d')
                bu.plot_pseq(ax, pseq, c='k')
                bu.scatter_pseq(ax, pseq[1:-2], c='g')
                bu.scatter_pseq(ax, pseq[:1], c='r')
                plt.show()
                bs.show_bendresseq(bendresseq, [True] * len(seqs))
                base.run()
            else:
                fail_tc_list.append(total_tc)
                bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
                bs.gen_by_bendseq(bendset, cc=False)
                ax = plt.axes(projection='3d')
                bu.plot_pseq(ax, bs.pseq, c='k')
                bu.scatter_pseq(ax, bs.pseq[1:-2], c='r')
                plt.show()
            if type(attemp_cnt_list) == type([]):
                total_cnt_list.append(attemp_cnt_list[-1])
            else:
                total_cnt_list.append(attemp_cnt_list)

    print('Success Cnt:', success_cnt)
    print('Fail time cost:', np.average(fail_tc_list))
    print('Success time cost:', np.average(success_tc_list))
    print('First time cost:', np.average(success_first_tc_list), success_first_tc_list)
    print('Num. of solution:', np.average(success_cnt_list), success_cnt_list)
    print('Num. of attempts:', np.average(total_cnt_list))


def plot_success(ax, x_range, fo, clr, marker='+'):
    avg_first_tc_list = []
    avg_top10_tc_list = []
    avg_attempt_list = []

    for num in x_range:
        print(f'---------{fo}, {num}---------')
        success_cnt, fail_tc, top10_tc, first_tc, attemps = load_res(num, fo=fo)
        print(success_cnt, top10_tc, first_tc)
        avg_first_tc_list.append(np.average(first_tc))
        avg_top10_tc_list.append(np.average(top10_tc))
        avg_attempt_list.append(np.average(attemps))
        # ax.scatter([num] * len(top10_tc), top10_tc, color=clr, marker=marker, s=100)
        ax.scatter([num] * len(first_tc), first_tc, color=clr, marker=marker, s=100)
    ax.plot(x_range, avg_first_tc_list, color=clr)
    # ax.plot(x_range, avg_top10_tc_list, color=clr)


def plot_failed(ax, x_range, fo, clr):
    avg_fail_tc_list = []
    avg_attempt_list = []

    for num in x_range:
        print(f'---------{num}---------')
        success_cnt, fail_tc, top10_tc, first_tc, attemps = load_res(num, fo=fo)
        print(success_cnt, fail_tc)
        avg_fail_tc_list.append(np.average(fail_tc))
        avg_attempt_list.append(np.average(attemps))
    ax.plot(x_range, avg_fail_tc_list, color=clr)


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bs = b_sim.BendSim(show=True)
    fo = '90'
    # tst(bs, 5, fo=fo)
    x_range = range(3, 9)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    ax = plt.axes()
    ax.grid()
    plot_success(ax, x_range, '45', clr='tab:green', marker='1', )
    plot_success(ax, x_range, '90', clr='tab:blue', marker='2')
    # plot_success(ax, x_range, '135', clr='tab:green', marker='3')
    plot_success(ax, x_range, '180', clr='tab:orange', marker='4')
    plt.show()

    ax = plt.axes()
    ax.grid()
    plot_failed(ax, x_range, '45', clr='tab:blue')
    plot_failed(ax, x_range, '90', clr='tab:orange')
    plot_failed(ax, x_range, '135', clr='tab:green')
    plot_failed(ax, x_range, '180', clr='tab:red')

    plt.show()
