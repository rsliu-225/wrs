import os
import pickle

import numpy as np

import bendplanner.bend_utils as bu
import matplotlib.pyplot as plt
import bendplanner.BendSim as b_sim
import visualization.panda.world as wd

base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
bs = b_sim.BendSim(show=True)

total_tc_list = []
success_cnt_list = []
success_first_tc_list = []
total_cnt_list = []
success_cnt = 0
for f in os.listdir('./'):
    if f[-3:] == 'pkl' and f == '4_0.pkl':
        print(f)
        result, tc_list, attemp_cnt_list, total_tc, bendset = pickle.load(open(f, 'rb'))
        print(tc_list)
        print(attemp_cnt_list)
        if len(result) != 0:
            success_cnt += 1
            bendresseq, seqs = result[3]
            _, _, _, _, _, pseq, _ = bendresseq[-1]
            bs.show_bendresseq(bendresseq, [True] * len(seqs))
            base.run()
            ax = plt.axes(projection='3d')
            bu.plot_pseq(ax, pseq, c='k')
            bu.scatter_pseq(ax, pseq[1:-2], c='g')
            plt.show()
            success_first_tc_list.append(tc_list[0])
        else:
            bs.reset([(0, 0, 0), (0, bendset[-1][3], 0)], [np.eye(3), np.eye(3)])
            bs.gen_by_bendseq(bendset, cc=False)
            ax = plt.axes(projection='3d')
            bu.plot_pseq(ax, bs.pseq, c='k')
            bu.scatter_pseq(ax, bs.pseq[1:-2], c='r')
            plt.show()
        total_tc_list.append(total_tc)
        success_cnt_list.append(len(tc_list))
        total_cnt_list.append(attemp_cnt_list[-1])

print('Success Cnt:', success_cnt)
print('Total time cost:', np.average(total_tc_list))
print('First time cost:', np.average(success_first_tc_list))
print('Num. of solution:', np.average(success_cnt_list))
print('Num. of attempts:', np.average(total_cnt_list))
