import pickle

import numpy as np


def print_info(info_list):
    if len(info_list) != 0:
        print(len(info_list), info_list)
        print('min:', min(info_list))
        print('mean:', np.mean(info_list))
        print('std:', np.std(info_list))
        print('max:', max(info_list))
    else:
        print(len(info_list), info_list)


folder_name = 'tc_helmet_ik'
time_cost_dict = pickle.load(open(f'./{folder_name}/time_cost.pkl', 'rb'))
success_tc = []
success_gp_tc = []
success_draw_tc = []
success_mp_tc = []
fail_mp_tc = []
fail_gp_tc = []
fail_draw_tc = []
fail_tc = []

gp_tc = 0
mp_tc = 0

path_dict = pickle.load(open(f'./{folder_name}/draw_circle.pkl', 'rb'))
for k, v in path_dict.items():
    _, _, path = v
    print(k, len(path))

for k, v in time_cost_dict.items():
    fail = v['time_cost_gp'] + v['time_cost_mp']
    gp_tc += v['time_cost_gp']
    mp_tc += v['time_cost_mp']

    if v['flag_gp']:
        print("gp success:", k, gp_tc)
    if v['flag_gp'] and v['flag_mp'] and v['flag_draw']:
        print("all success:", k, mp_tc)
    if v['flag_gp'] and v['flag_mp'] and v['flag_draw']:
        success_gp_tc.append(v['time_cost_gp'])
        success_tc.append(v['time_cost_gp'] + v['time_cost_mp'])
        success_draw_tc.append(v['time_cost_draw'])
        success_mp_tc.append(v['time_cost_mp'] - v['time_cost_draw'])
    else:
        fail_tc.append(fail)
        # print(k, v['flag_gp'], v['flag_mp'], v['flag_draw'])

    if not v['flag_gp']:
        fail_gp_tc.append(fail)
    elif v['flag_gp'] and not v['flag_mp'] and not v['flag_draw']:
        fail_mp_tc.append(fail)
    if v['flag_gp'] and v['flag_mp'] and not v['flag_draw']:
        fail_draw_tc.append(fail)

print("------------success(grasp reasoning)------------")
print_info(success_gp_tc)
print("------------success(except plan draw)------------")
print_info(success_mp_tc)
print("------------success(plan draw)------------")
print_info(success_draw_tc)
print("------------success(total)------------")
print_info(success_tc)
print("------------fail(grasp reasoning)------------")
print_info(fail_gp_tc)
print("------------fail(motion planning)------------")
print_info(fail_mp_tc)
print("------------fail(draw motion)------------")
print_info(fail_draw_tc)
print("------------fail(total)------------")
print_info(fail_tc)

print("------------overall time cost------------")
print(gp_tc)
print(mp_tc)
