import multiprocessing
import time
import random


def fff(v_ids, fs, start_idx, rl):
    for idx, f in enumerate(fs):
        if list(set(v_ids) & set(f)):
            rl.append(start_idx + idx)


def filter_face_mp(faces, v_ids, slices_num=4):
    time_start = time.time()

    step = int(len(faces) / slices_num)
    ss = 0

    manager = multiprocessing.Manager()
    res_list = manager.list()
    processes = []

    for i in range(slices_num):
        se = ss + step
        if se > len(faces):
            se = len(faces)

        sub_faces = faces[ss:se]
        p = multiprocessing.Process(
            target=fff, args=(v_ids, sub_faces, ss, res_list)
        )
        processes.append(p)
        p.start()
        ss += step

    for p in processes:
        p.join()
    print("time cost:", time.time() - time_start)

    return res_list


def filter_face(faces, v_ids):
    time_start = time.time()

    res_list = []
    for i, f in enumerate(faces):
        if list(set(v_ids) & set(f)):
            res_list.append(i)
    print("time cost:", time.time() - time_start)
    return res_list


# multi processing
if __name__ == "__main__":
    faces = [random.choices(range(10000), k=3) for _ in range(1000000)]
    v_ids = random.choices(range(10000), k=1000)
    print(len(faces))
    res_list = filter_face_mp(faces, v_ids, slices_num=100)
    # print(len(res_list))
    res_list = filter_face(faces, v_ids)
    # print(len(res_list))
