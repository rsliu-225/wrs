import itertools


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def pnp_cnt(l):
    cnt = 0
    for i in range(len(l) - 1):
        if l[i + 1] < l[i]:
            cnt += 1
        elif len(intersection(l[:i], range(l[i], l[i + 1]))) > 0:
            cnt += 1
    return cnt


def unstable_cnt(l):
    cnt = 0
    for i in range(1, len(l)):
        if l[i] < max(l[:i]):
            cnt += 1
    return cnt


def rank_combs(combs):
    pnp_cnt_list = []
    unstable_cnt_list = []
    for l in combs:
        pnp_cnt_list.append(pnp_cnt(l))
        unstable_cnt_list.append(unstable_cnt(l))
    _, _, combs = zip(*sorted(zip(pnp_cnt_list, unstable_cnt_list, combs)))
    for l in combs:
        if pnp_cnt(l) < 5:
            print(l, pnp_cnt(l), unstable_cnt(l))
    return combs


def remove_combs(rmv_l, combs):
    new_combs = []
    for i, comb in enumerate(combs):
        print(i, comb)
        if set(comb[:len(rmv_l) - 1]) == set(rmv_l[:len(rmv_l) - 1]) and comb[len(rmv_l) - 1] == rmv_l[-1]:
            continue
        new_combs.append(comb)
    return new_combs


if __name__ == '__main__':
    l = range(8)
    combs = list(itertools.permutations(l, len(l)))
    print(l, len(combs))

    # new_combs = remove_combs([0], combs)
    # for comb in new_combs:
    #     print(comb)
    # print(len(combs), len(new_combs))
    rank_combs(combs)
