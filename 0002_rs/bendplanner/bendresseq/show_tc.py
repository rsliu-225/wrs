import os
import pickle
import cv2 as cv2

import numpy as np

import bendplanner.bend_utils as bu
import matplotlib.pyplot as plt
import bendplanner.BendSim as b_sim
import visualization.panda.world as wd


def pnp_cnt(l):
    def _intersec(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    cnt = 0
    for i in range(len(l) - 1):
        if l[i + 1] < l[i]:
            cnt += 1
        elif len(_intersec(l[:i], range(l[i], l[i + 1]))) > 0:
            cnt += 1
    return cnt


def load_res(num, fo='180'):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    pnp_cnt_list = []
    success_cnt = 0
    for f in os.listdir(f'./{fo}'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f'./{fo}/{f}', 'rb'))
            # print(f, total_tc)
            if len(result) != 0:
                pnp_cnt_tmp = []
                for l in result:
                    try:
                        pnp_cnt_tmp.append(pnp_cnt(l))
                    except:
                        pnp_cnt_tmp.append(pnp_cnt(l[-1]))

                pnp_cnt_list.append(pnp_cnt_tmp)
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

    return success_cnt, fail_tc_list, success_tc_list[:10], success_first_tc_list[:10], total_cnt_list[:10], \
           pnp_cnt_list[:10]


def show_shape(bs, num, fo='180'):
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
            bs.reset([(0, 0, 0), (0, max([v[3] for v in bendset]), 0)], [np.eye(3), np.eye(3)])
            print(tc_list)
            # print(attemp_cnt_list)
            # print(total_tc)
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])
                try:
                    bendresseq, seqs = result[0]
                except:
                    seqs = result[0]
                # print(seqs)
                bendseq = [bendset[i] for i in seqs]
                is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=False, prune=True, toggledebug=False)
                _, _, _, _, _, pseq, _ = bendresseq[-1]
                pseq = np.asarray(pseq) * 1000
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
                pseq[-1] = pseq[-1] - (pseq[-1] - pseq[-2]) * .8
                curture_list, r_list, torsion_list = bu.cal_curvature(pseq, show=False)
                print(sum(curture_list), sum(torsion_list))

                ax = plt.axes(projection='3d')
                center = pseq.mean(axis=0)
                eps = 40
                ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
                ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
                ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)

                bu.plot_pseq(ax, pseq, c='k')
                bu.scatter_pseq(ax, pseq[1:-2], c='r')
                bu.scatter_pseq(ax, pseq[:1], c='g')
                plt.show()
                # bs.show_bendresseq(bendresseq, [True] * len(seqs))
                # base.run()
            else:
                fail_tc_list.append(total_tc)
                bs.reset([(0, 0, 0), (0, max([v[3] for v in bendset]), 0)], [np.eye(3), np.eye(3)])
                bs.gen_by_bendseq(bendset, cc=False)
                pseq = np.asarray(bs.pseq) * 1000
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .5
                pseq[-1] = pseq[-1] - (pseq[-1] - pseq[-2]) * .8

                ax = plt.axes(projection='3d')
                center = pseq.mean(axis=0)
                eps = 40
                ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
                ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
                ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)

                bu.plot_pseq(ax, pseq, c='k')
                bu.scatter_pseq(ax, pseq[1:-2], c='grey')
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


def plot_curveture_tc(bs, num=8, fo_list=['45', '90', '180']):
    plt.grid()
    for fo in fo_list:
        success_tc_list = []
        success_first_tc_list = []
        curveture_sum_list = []
        torsion_sum_list = []
        for f in os.listdir(f'./{fo}'):
            if f[-3:] == 'pkl' and f[0] == str(num):
                print(f'=========={f}==========')
                result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f'./{fo}/{f}', 'rb'))
                bs.reset([(0, 0, 0), (0, max([v[3] for v in bendset]), 0)], [np.eye(3), np.eye(3)])
                if len(result) != 0:
                    print('time cost:', tc_list)
                    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, prune=True, toggledebug=False)
                    _, _, _, _, _, pseq, _ = bendresseq[-1]
                    pseq = np.asarray(pseq) * 1000
                    pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .5
                    try:
                        curture_list, r_list, torsion_list = bu.cal_curvature(pseq, show=False)
                    except:
                        continue
                    print(sum(curture_list), sum(torsion_list))
                    if sum(curture_list) < 200:
                        curveture_sum_list.append(sum(curture_list))
                        torsion_sum_list.append(sum(torsion_list))
                        success_tc_list.append(total_tc)
                        success_first_tc_list.append(tc_list[0])

        plt.scatter(curveture_sum_list, success_first_tc_list)
    # plt.scatter(success_first_tc_list, torsion_sum_list)
    plt.show()


def gen_img(bs, num, fo='180', show=False):
    success_tc_list = []
    success_cnt_list = []
    success_first_tc_list = []
    fail_tc_list = []
    total_cnt_list = []
    success_cnt = 0
    img_success = None
    img_failed = None
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    eps = 50
    cnt = 0

    for f in os.listdir(f'./{fo}'):
        if f[-3:] == 'pkl' and f[0] == str(num):
            if cnt >= 10:
                continue
            print(f'=========={f}==========')
            result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f'./{fo}/{f}', 'rb'))
            bs.reset([(0, 0, 0), (0, max([v[3] for v in bendset]), 0)], [np.eye(3), np.eye(3)])
            print(tc_list)
            print(attemp_cnt_list)
            print(total_tc)
            if len(result) != 0:
                success_cnt += 1
                success_cnt_list.append(len(result))
                success_tc_list.append(total_tc)
                success_first_tc_list.append(tc_list[0])
                try:
                    bendresseq, seqs = result[0]
                except:
                    seqs = result[0]

                print(seqs, attemp_cnt_list[0])
                bendseq = [bendset[i] for i in seqs]
                is_success, bendresseq, _ = bs.gen_by_bendseq(bendseq, cc=False, prune=True, toggledebug=False)
                _, _, _, _, _, pseq, _ = bendresseq[-1]
                pseq = np.asarray(pseq) * 1000
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
                pseq[-1] = pseq[-1] - (pseq[-1] - pseq[-2]) * .8
                kpts = []
                for i in range(len(pseq) - 1):
                    dist = np.linalg.norm(pseq[i + 1] - pseq[i])
                    if dist > 1:
                        kpts.append(pseq[i + 1])
                ax = plt.axes(projection='3d')
                center = pseq.mean(axis=0)
                ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
                ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
                ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)
                ax.text2D(0, 0.85, f'{str(round(tc_list[0], 2))} s, {attemp_cnt_list[0]} times',
                          transform=ax.transAxes, fontsize=16)
                # ax.text2D(0, 0.78, f'{str(round(tc_list[-1], 2))} s', transform=ax.transAxes, fontsize=16)
                ax.text2D(0, 0.78, f'{str([len(seqs) - 1 - v for v in seqs])}', transform=ax.transAxes, fontsize=16)
                # for i, v in enumerate(seqs):
                #     print(v)
                #     ax.text(kpts[v][0], kpts[v][1], kpts[v][2], str(i), color='red')

                bu.plot_pseq(ax, pseq, c='k')
                bu.scatter_pseq(ax, pseq[1:-2], c='r')
                bu.scatter_pseq(ax, pseq[:1], c='g', s=10)
                plt.savefig('img/final/success.png', dpi=200)
                img_tmp = cv2.imread('img/final/success.png')

                # cv2.putText(img_tmp,
                #             text=f'Find First Solution: {str(round(tc_list[0], 2))} s',
                #             org=(250, 200), thickness=2,
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1.2, color=(0, 0, 0))
                # cv2.putText(img_tmp,
                #             text=f'Find First 10 Solutions: {str(round(tc_list[-1], 2))} s',
                #             org=(250, 250), thickness=2,
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1.2, color=(0, 0, 0))
                # cv2.putText(img_tmp,
                #             text=f'Planed Sequence: {str(seqs)}',
                #             org=(250, 300), thickness=2,
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=1.2, color=(0, 0, 0))
                img_tmp = img_tmp[160:900, 200:1100]
                if img_success is None:
                    img_success = img_tmp
                else:
                    img_success = np.hstack((img_success, img_tmp))
                if show:
                    cv2.imshow('', img_success)
                    cv2.waitKey(0)
                plt.clf()
                cnt += 1
                # bs.show_bendresseq(bendresseq, [True] * len(seqs))
                # base.run()
            else:
                fail_tc_list.append(total_tc)
                bs.reset([(0, 0, 0), (0, max([v[3] for v in bendset]), 0)], [np.eye(3), np.eye(3)])
                bs.gen_by_bendseq(bendset, cc=False)
                pseq = np.asarray(bs.pseq) * 1000
                pseq[0] = pseq[0] - (pseq[0] - pseq[1]) * .8
                pseq[-1] = pseq[-1] - (pseq[-1] - pseq[-2]) * .8

                ax = plt.axes(projection='3d')
                center = pseq.mean(axis=0)
                ax.axes.set_xlim3d(left=center[0] - eps, right=center[0] + eps)
                ax.axes.set_ylim3d(bottom=center[1] - eps, top=center[1] + eps)
                ax.axes.set_zlim3d(bottom=center[2] - eps, top=center[2] + eps)
                bu.plot_pseq(ax, pseq, c='k')
                bu.scatter_pseq(ax, pseq[1:-2], c='grey')
                plt.savefig('img/final/failed.png', dpi=200)
                img_tmp = cv2.imread('img/final/failed.png')
                print(img_tmp.shape)
                img_tmp = img_tmp[160:900, 260:1100]
                if img_failed is None:
                    img_failed = img_tmp
                else:
                    img_failed = np.hstack((img_failed, img_tmp))
                if show:
                    cv2.imshow('', img_failed)
                    cv2.waitKey(0)
                plt.clf()
            if type(attemp_cnt_list) == type([]):
                total_cnt_list.append(attemp_cnt_list[-1])
            else:
                total_cnt_list.append(attemp_cnt_list)

    cv2.imwrite(f'img/final/{fo}_{str(num)}_success.png', img_success)
    if img_failed is not None:
        cv2.imwrite(f'img/final/{fo}_{str(num)}_failed.png', img_failed)

    print('Success Cnt:', success_cnt)
    print('Fail time cost:', np.average(fail_tc_list))
    print('Success time cost:', np.average(success_tc_list))
    print('First time cost:', np.average(success_first_tc_list), success_first_tc_list)
    print('Num. of solution:', np.average(success_cnt_list), success_cnt_list)
    print('Num. of attempts:', np.average(total_cnt_list))


def plot_success(ax, x_range, fo, clr, d, alpha=1., marker='+'):
    avg_first_tc_list = []
    avg_top10_tc_list = []
    avg_attempt_list = []
    first_tc_box = []
    top10_tc_box = []
    for num in x_range:
        print(f'---------{fo}, {num}---------')
        success_cnt, fail_tc, top10_tc, first_tc, attemps, pnp_cnts = load_res(num, fo=fo)
        print(success_cnt, top10_tc, first_tc)
        print('first:', ' & '.join([str(round(t, 2)) for t in first_tc + [np.asarray(first_tc).mean()]]))
        print('top10:', ' & '.join([str(round(t, 2)) for t in top10_tc + [np.asarray(top10_tc).mean()]]))
        print('pnp times:', ' & '.join([str(round(np.asarray(l).mean(), 2)) for l in pnp_cnts]))
        print('Num. of sol:', ' & '.join([str(len(l)) for l in pnp_cnts]))
        first_tc_box.append(first_tc)
        top10_tc_box.append(top10_tc)
        avg_first_tc_list.append(np.average(first_tc))
        avg_top10_tc_list.append(np.average(top10_tc))
        avg_attempt_list.append(np.average(attemps))
        ax.set_xticks(x_range)
        # ax.scatter([num] * len(first_tc), first_tc, color=clr, marker=marker, s=150)
        # ax.scatter([num] * len(top10_tc), top10_tc, color=clr, marker=marker, s=150)
    # box1 = ax.boxplot(first_tc_box, positions=x_range, patch_artist=True)
    # box1 = ax.boxplot(top10_tc_box, positions=x_range, patch_artist=True)
    if d == 1:
        box1 = ax.boxplot(first_tc_box, positions=[d + 3 * (x - 1) + .25 for x in x_range], patch_artist=True)
        # box1 = ax.boxplot(top10_tc_box, positions=[d + 3 * (x - 1) + .25 for x in x_range], patch_artist=True)
    elif d == 2:
        box1 = ax.boxplot(first_tc_box, positions=[d + 3 * (x - 1) for x in x_range], patch_artist=True)
        # box1 = ax.boxplot(top10_tc_box, positions=[d + 3 * (x - 1) for x in x_range], patch_artist=True)
    else:
        box1 = ax.boxplot(first_tc_box, positions=[d + 3 * (x - 1) - .25 for x in x_range], patch_artist=True)
        # box1 = ax.boxplot(top10_tc_box, positions=[d + 3 * (x - 1) - .25 for x in x_range], patch_artist=True)

    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box1[item], color=clr, alpha=alpha)
    plt.setp(box1["boxes"], facecolor=clr)
    plt.setp(box1["fliers"], markeredgecolor=clr)
    # ax.plot(x_range, avg_first_tc_list, color=clr)
    # ax.plot(x_range, avg_top10_tc_list, color=clr)


def plot_failed(ax, x_range, fo, clr):
    avg_fail_tc_list = []
    avg_attempt_list = []

    for num in x_range:
        print(f'---------{num}---------')
        success_cnt, fail_tc, top10_tc, first_tc, attemps, pnp_cnts = load_res(num, fo=fo)
        print(success_cnt, fail_tc)
        avg_fail_tc_list.append(np.average(fail_tc))
        avg_attempt_list.append(np.average(attemps))
    ax.plot(x_range, avg_fail_tc_list, color=clr)


def grid_on(ax):
    ax.minorticks_on()
    ax.grid(b=True, which='major')
    ax.grid(b=True, which='minor', linestyle='--', alpha=.2)


if __name__ == '__main__':
    base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    bs = b_sim.BendSim(show=True)
    # fo_list = ['final_45', 'final_90', 'final_180']
    # for fo in fo_list:
    #     for i in range(7, 8):
    #         gen_img(bs, i, fo=fo, show=False)
    # show_shape(bs, 7, fo=fo)
    # plot_curveture_tc(bs, num=7, fo_list=[ 'final_90', 'final_180'])

    x_range = range(3, 9)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    ax = plt.axes()
    grid_on(ax)
    plot_success(ax, x_range, '45', d=1, clr='tab:gray', marker='1', alpha=.2)
    plot_success(ax, x_range, 'final_45', d=1, clr='tab:green', marker='1')
    plot_success(ax, x_range, '90', d=2, clr='tab:gray', marker='2', alpha=.2)
    plot_success(ax, x_range, 'final_90', d=2, clr='tab:blue', marker='2')
    plot_success(ax, x_range, '180', d=3, clr='tab:gray', marker='3', alpha=.2)
    plot_success(ax, x_range, 'final_180', d=3, clr='tab:orange', marker='3')
    plt.xticks([2 + 3 * (x - 1) for x in x_range], x_range)
    # plt.xticks(x_range, x_range)
    plt.show()

    # grid_on(ax)
    # ax = plt.axes()
    # ax.grid()
    # plot_failed(ax, x_range, '90', clr='tab:blue')
    # plot_failed(ax, x_range, '180', clr='tab:orange')
    #
    # plt.show()
