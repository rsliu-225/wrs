import pickle

import numpy as np

import utiltools.robotmath as rm
import config


def cnt_pos_in_gmap(objmat4ngrasp_pair_dict, graspid_list=None):
    cnt = 0
    if graspid_list is None:
        for k, objmat4_dict in objmat4ngrasp_pair_dict.items():
            for obj_k, bool in objmat4_dict.items():
                if bool:
                    cnt += 1
    else:
        for k, objmat4_dict in objmat4ngrasp_pair_dict.items():
            if k in graspid_list:
                for obj_k, bool in objmat4_dict.items():
                    if bool:
                        cnt += 1
    print("Num of avalible objmat4final-grasp pair:", cnt)
    return cnt


def get_candidate_objmat4_list(x_range, y_range, z_range, roll_range, pitch_range, yaw_range, pos_step=20, rot_step=20):
    def __get_value_list(range_value, step):
        if isinstance(range_value, int):
            return [range_value]
        else:
            return [v for v in range(range_value[0], range_value[1] + 1, step)]

    objpos_list = []
    objrot_list = []
    objmat4_list = []

    for x in __get_value_list(x_range, pos_step):
        for y in __get_value_list(y_range, pos_step):
            for z in __get_value_list(z_range, pos_step):
                objpos_list.append((x, y, z))

    for roll in __get_value_list(roll_range, rot_step):
        for pitch in __get_value_list(pitch_range, rot_step):
            for yaw in __get_value_list(yaw_range, 180):
                objrot_list.append(np.dot(np.dot(rm.rodrigues([1, 0, 0], roll), rm.rodrigues([0, 1, 0], pitch)),
                                          rm.rodrigues([0, 0, 1], yaw)))
    for objpos in objpos_list:
        for objrot in objrot_list:
            objmat4 = rm.homobuild(objpos, objrot)
            objmat4_list.append(objmat4)
    print("Num of objmat4:", len(objmat4_list))

    return objmat4_list


def get_graspmap(motion_planner, obj, objmat4_list, pregrasp_list):
    graspmap = {}
    print("Num of grasp:", len(pregrasp_list))
    for i in range(len(pregrasp_list)):
        grasp = pregrasp_list[i]
        objmat4_dict = {}
        for j in range(len(objmat4_list)):
            objmat4 = objmat4_list[j]
            objmat4_dict[j] = motion_planner.is_grasp_available(grasp, obj, objmat4)
            if j % 500 == 0:
                print(i, j, objmat4_dict[j])
                cnt_pos_in_gmap(graspmap)
        graspmap[i] = objmat4_dict

    return graspmap


if __name__ == '__main__':
    pen_f_name = "pentip"
    graspmap = pickle.load(open(config.ROOT + "/graspplanner/graspmap/" + pen_f_name + "_graspmap.pkl", "rb"))
    print(graspmap[17].values())
