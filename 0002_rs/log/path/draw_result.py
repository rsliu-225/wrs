import os

import matplotlib.pyplot as plt
import numpy as np


def padding(m_dict, total):
    cost = []
    time_cost = []
    nodes = []
    success_cnt = 0

    for i in range(total):
        # if i in list(m_dict.keys()) and int(m_dict[i][2]) >= int(path_cnt) and float(m_dict[i][0]) < 3000:
        if i in list(m_dict.keys()) and float(m_dict[i][0]) < 3000:
            cost.append(float(m_dict[i][0]))
            time_cost.append(float(m_dict[i][1]))
            nodes.append(int(m_dict[i][2]))
            success_cnt += 1
        else:
            cost.append(None)
            time_cost.append(None)
            nodes.append(None)

    print(f"Success {success_cnt} of {total}",
          "Avg. cost", round(np.average([v for v in cost if v is not None]), 3),
          "Avg. time cost", round(np.average([v for v in time_cost if v is not None]), 3))

    return cost, time_cost, nodes


def get_avg_cost(cost, nodes):
    avg_cost = []
    for i, v in enumerate(cost):
        if v is not None:
            avg_cost.append(v / path_cnt)
        else:
            avg_cost.append(None)
    return avg_cost


def get_f_from_floder_list(floder_list):
    f_list = []
    for folder in floder_list:
        for dirpath, dirnames, filenames in os.walk(folder):
            f_list.append([dirpath, dirnames, filenames])
    # print(f_list)
    return f_list


if __name__ == '__main__':
    total = 30
    path_cnt = 149
    folder_list = ["ik", "nlopt", "discrete"]
    # m_list = ["ik3",  "m1r10_ik3", "m1w5_ik3"]
    # m_list = ["bowlik3", "nocolw5", "colw5", "toolcolw5"]
    # legend_list = ["baseline", "opt", "+hand collision", "+tool collision"]
    m_list = ["bucketcol", "boxcolw45"]

    m_dict = {}

    for m_name in m_list:
        m_dict[m_name] = {}

    for dirpath, dirnames, filenames in get_f_from_floder_list(folder_list):
        for f in filenames:
            if f.endswith(".png"):
                f = f.split(".png")[0]
                try:
                    m, grasp_id, cost, time_cost, node_num = f.split("_")
                    print(m, grasp_id, cost, time_cost, node_num)
                except:
                    m1, m2, grasp_id, cost, time_cost, node_num = f.split("_")
                    m = m1 + "_" + m2
                    print(m, grasp_id, cost, time_cost, node_num)
                for m_name in m_list:
                    if m == m_name:
                        m_dict[m_name][int(grasp_id)] = [cost, time_cost, node_num]

    x = range(total)
    major_ticks = np.arange(0, total, 5)
    minor_ticks = np.arange(0, total, 1)

    fig = plt.figure(1, figsize=(6.4 * 3, 4.8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for i, m_name in enumerate(m_list):
        print(m_name)
        cost, time_cost, nodes = padding(m_dict[m_name], total)
        avg_cost = get_avg_cost(cost, nodes)

        # ax1.scatter(x, cost, label=[m_name], marker=str(i + 1), s=80)
        # ax1.set_xticks(major_ticks)
        # ax1.set_xticks(minor_ticks, minor=True)
        # ax1.grid(which='minor', linestyle='dotted', alpha=0.5)
        # ax1.grid(which='major', linestyle='dotted', alpha=1)
        # ax1.grid(axis="x", linestyle='dotted')
        # ax1.legend(legend_list)
        # ax1.set_title("Total Joints Displacement")

        ax1.scatter(x, avg_cost, label=[m_name], marker=str(int(i % 4 + 1)), s=80)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.grid(which='minor', linestyle='dotted', alpha=0.5)
        ax1.grid(which='major', linestyle='dotted', alpha=1)
        ax1.grid(axis="x", linestyle='dotted')
        ax1.legend(legend_list)
        ax1.set_title("Average Joints Displacement")

        ax2.scatter(x, time_cost, label=[m_name], marker=str(int(i % 4 + 1)), s=80)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', linestyle='dotted', alpha=0.5)
        ax2.grid(which='major', linestyle='dotted', alpha=1)
        ax2.legend(legend_list)
        ax2.set_title("Time Cost")

        ax3.axhline(y=path_cnt, color='r', linestyle=':', linewidth=1, alpha=0.5)
        ax3.scatter(x, nodes, label=[m_name], marker=str(int(i % 4 + 1)), s=80)
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='minor', linestyle='dotted', alpha=0.5)
        ax3.grid(which='major', linestyle='dotted', alpha=1)
        ax3.legend(m_list)
        ax3.set_title("Number of Nodes")

    plt.show()
