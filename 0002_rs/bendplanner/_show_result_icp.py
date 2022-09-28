import math
import os
import pickle

import numpy as np

import basis.o3dhelper as o3dh
import basis.robot_math as rm
import bendplanner.BendRbtPlanner as br_planner
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import config
import localenv.envloader as el
import modeling.geometric_model as gm
import motionplanner.motion_planner as m_planner
import utils.pcd_utils as pcdu

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def mirror(pseq, n, d):
    def _point(p, n, d):
        px, py, pz = p
        a, b, c = n
        squaredsum = a * a + b * b + c * c
        normlen = 1.0 / math.sqrt(squaredsum)
        nx = a * normlen
        ny = b * normlen
        nz = c * normlen
        ox = a * d / squaredsum
        oy = b * d / squaredsum
        oz = c * d / squaredsum
        dott = nx * (px - ox) + ny * (py - oy) + nz * (pz - oz)
        rx = px - nx * dott
        ry = py - ny * dott
        rz = pz - nz * dott
        symx = 2 * rx - px
        symy = 2 * ry - py
        symz = 2 * rz - pz
        return (symx, symy, symz)

    res = []
    for p in pseq:
        res.append(_point(p, n, d))
    return np.asarray(res)


if __name__ == '__main__':
    fo = 'stick'
    f = 'chair'
    rbt_name = 'yumi'
    base, env = el.loadEnv_yumi(camp=[3, -.5, 2.5], lookatpos=[0, 0, 0])

    if rbt_name == 'yumi':
        rbt = el.loadYumi(showrbt=True)
    else:
        rbt = el.loadUr3e(showrbt=True)

    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f}.pkl'), 'rb'))
    seqs, _, bendresseq = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f}_bendresseq.pkl', 'rb'))
    _, bendset = \
        pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f}_bendset.pkl', 'rb'))
    transmat4 = pickle.load(open(f'{config.ROOT}/bendplanner/planres/{fo}/{rbt_name}/{f}_transmat4.pkl', 'rb'))

    bs = b_sim.BendSim(show=False)
    mp = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    # print(armjntsseq_list[0])
    # mp.ah.show_armjnts(armjnts=armjntsseq_list[1][1][1])

    init_pseq = [(0, 0, 0), (0, max([b[-1] for b in bendset]), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]
    # brp = br_planner.BendRbtPlanner(bs, init_pseq, init_rotseq, mp)
    # brp.set_up(bendset, [], transmat4)

    if f == 'penta':
        x_range = (.4, 1)
        y_range = (-.3, .3)
        z_range = (.2, .25)
    if f == 'randomc':
        x_range = (.32, 1)
        y_range = (0, .1)
        z_range = (.22, .5)
    if f == 'chair':
        x_range = (.3, 1)
        y_range = (.03, .13)
        z_range = (.1, .5)

    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi/exp_bend', fo, f, 'result.pkl'), 'rb'))
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd))
    pcd_bg = pcdu.crop_pcd(pcd, x_range=(0, 2), y_range=(-1, 1), z_range=(.1, 1))
    pcd_crop = pcdu.crop_pcd(pcd, x_range=x_range, y_range=y_range, z_range=z_range)
    pcdu.show_pcd(pcd_bg, rgba=(1, 1, 1, .5))
    pcdu.show_pcd(pcd_crop, rgba=(1, 0, 0, 1))
    # pcd_crop = pcdu.get_kpts_gmm(pcd_crop, n_components=50, show=True)

    goal_pseq = mirror(goal_pseq, (0, 0, 1), 0)
    goal_pseq = pcdu.trans_pcd(goal_pseq,
                               rm.homomat_from_posrot(np.mean(pcd_crop, axis=0) + np.asarray([-.018, -.035, .01]),
                                                      rm.rotmat_from_axangle((1, 0, 0), np.pi / 4)))
    goal_pseq = bu.linear_inp3d_by_step(goal_pseq, step=.001)
    # for p in goal_pseq:
    #     gm.gen_sphere(p, radius=.001, rgba=(0, 1, 0, 1)).attach_to(base)

    # obj_init, obj_end = brp.show_bend(bendresseq[-1], show_end=True, show_start=False)
    # goal_pcd, _ = obj_end.sample_surface(radius=.001)
    # goal_pcd = pcdu.get_objpcd_partial_bycampos(obj_end, smp_num=2000, cam_pos=(0, 0, -1))
    # for p in goal_pseq:
    #     gm.gen_sphere(p, radius=.001, rgba=(0, 1, 0, 1)).attach_to(base)
    # pcdu.show_pcd(goal_pcd)
    # rmse, fitness, trans = o3dh.registration_ptpt(src=goal_pcd, tgt=pcd_crop, toggledebug=False)
    # print(rmse, fitness)
    # goal_pcd = rm.homomat_transform_points(trans, goal_pcd)
    # pcdu.show_pcd(goal_pcd)

    rmse, fitness, trans = o3dh.registration_icp_ptpt(src=goal_pseq, tgt=pcd_crop, toggledebug=False)
    # rmse, fitness, trans = o3dh.registration_ptpt(src=np.asarray(goal_pseq), tgt=np.asarray(pcd_crop),
    #                                               downsampling_voxelsize=.002,
    #                                               toggledebug=False)
    print(rmse, fitness)
    goal_pseq = rm.homomat_transform_points(trans, goal_pseq)
    for p in goal_pseq:
        gm.gen_sphere(p, radius=.001, rgba=(0, 1, 0, 1)).attach_to(base)

    base.run()
