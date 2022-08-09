import os
import pickle

import numpy as np

import bendplanner.bend_utils as bu
import matplotlib.pyplot as plt
import bendplanner.BendSim as b_sim
import visualization.panda.world as wd


def load_res(num):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    success_cnt = 0
    for f in os.listdir('./'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f, 'rb'))
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])

            else:
                fail_tc_list.append(total_tc)
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

    return success_cnt, np.average(fail_tc_list), np.average(success_tc_list), \
           np.average(success_first_tc_list), np.average(total_cnt_list)


def tst(num):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    success_cnt = 0
    plot_success = True
    plot_fail = True
    for f in os.listdir('./'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            print(f'----------------{f}----------------')
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f, 'rb'))
            print(tc_list)
            print(attemp_cnt_list)
            print(total_tc)
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])
                bendresseq, seqs = result[-1]
                print(seqs)
                _, _, _, _, _, pseq, _ = bendresseq[-1]
                pseq = np.asarray(pseq)
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
                if plot_success:
                    ax = plt.axes(projection='3d')
                    bu.plot_pseq(ax, pseq, c='k')
                    bu.scatter_pseq(ax, pseq[1:-2], c='g')
                    bu.scatter_pseq(ax, pseq[:1], c='r')
                    plt.show()
                # bs.show_bendresseq(bendresseq, [True] * len(seqs))
                # base.run()
            else:
                fail_tc_list.append(total_tc)
                bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
                bs.gen_by_bendseq(bendset, cc=False)
                if plot_fail:
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


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bs = b_sim.BendSim(show=True)
    tst(5)

    avg_fail_tc_list = []
    avg_first_tc_list = []
    avg_top10_tc_list = []
    avg_attempt_list = []

    for num in range(3, 8):
        success_cnt, avg_fail_tc, avg_top10_tc, avg_first_tc, avg_attemps = load_res(num)
        print(f'---------{num}---------')
        print(success_cnt, avg_fail_tc, avg_top10_tc, avg_first_tc, avg_attemps)
        avg_fail_tc_list.append(avg_fail_tc)
        avg_first_tc_list.append(avg_first_tc)
        avg_top10_tc_list.append(avg_top10_tc)
        avg_attempt_list.append(avg_attemps)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    ax = plt.axes()
    # ax.plot([str(i) for i in range(3, 8)], avg_fail_tc_list)
    ax.plot([str(i) for i in range(3, 8)], avg_first_tc_list)
    ax.plot([str(i) for i in range(3, 8)], avg_top10_tc_list)
    # ax.grid(linestyle='dotted')
    ax.grid()
    plt.show()
