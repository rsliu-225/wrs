from _shape_dict import *
def gen_seed(input, kind="cubic", random=True):
    # Width & Thickness of the stick
    width = .008  # + (np.random.uniform(0, 0.005) if random else 0)
    thickness = .0015
    length = .2
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]
    pseq = uni_length(cubic_inp(step=.001, kind=kind, pseq=np.asarray(input)), goal_len=length)
    return gen_swap(pseq, get_rotseq_by_pseq(pseq), cross_sec)
if __name__ == '__main__':
    print(shape_dict)