import numpy as np
import cv2

import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import basis.robot_math as rm
from utils.run_script_utils import *
import visualization.panda.world as wd
import bendplanner.BendSim as b_sim
import bendplanner.bender_config as bconfig
import motorcontrol.Motor as motor
import utils.phoxi as phoxi
import modeling.geometric_model as gm

affine_mat = np.asarray([[0.00282079054, -1.00400178, -0.000574846621, 0.31255359],
                         [-0.98272743, -0.00797055, 0.19795055, -0.15903892],
                         [-0.202360828, 0.00546017392, -0.96800006, 0.94915224],
                         [0.0, 0.0, 0.0, 1.0]])


def _action(fo, f_name, goal_angle, z_range, line_thresh, line_size_thresh, ulim=None, rgba=(0, 1, 0, 1)):
    textureimg, depthimg, pcd = phxi.dumpalldata(f_name=os.path.join('img/phoxi/', 'exp_bend', fo, f_name))
    # cv2.imshow("depthimg", depthimg)
    # cv2.waitKey(0)
    pcd = rm.homomat_transform_points(affine_mat, np.asarray(pcd) / 1000)
    # textureimg = vu.enhance_grayimg(textureimg)
    lines = pcdu.extract_lines_from_pcd(textureimg, pcd, z_range=z_range, line_thresh=line_thresh,
                                        line_size_thresh=line_size_thresh, toggledebug=True)

    center = np.asarray([.4, 0, bconfig.BENDER_H])
    dist_list = []
    for _, line_pts in lines:
        pcdu.show_pcd(line_pts, rgba=(1, 1, 1, .2))
        kdt, _ = pcdu.get_kdt(line_pts)
        dist_list.append(pcdu.get_min_dist(center, kdt))
    sort_inx = np.argsort(np.asarray(dist_list))
    print(sort_inx, dist_list)
    angle = np.degrees(rm.angle_between_vectors(lines[sort_inx[0]][0], lines[sort_inx[1]][0]))
    pcdu.show_pcd(lines[sort_inx[0]][1], rgba=rgba)
    pcdu.show_pcd(lines[sort_inx[1]][1], rgba=rgba)

    if ulim is None:
        if abs(angle - goal_angle) > abs((180 - angle) - goal_angle):
            angle = abs(180 - angle)
    else:
        if abs(angle - goal_angle) > abs((180 - angle) - goal_angle) and abs(180) - angle < ulim:
            angle = abs(180 - angle)
    print(angle)
    return angle


if __name__ == '__main__':
    """
    set up env and param
    """
    base = wd.World(cam_pos=[4, 0, 1.7], lookat_pos=[0, 0, 1])
    env = None
    rbt = el.loadYumi(showrbt=True)
    rbtx = el.loadYumiX()

    motor = motor.MotorNema23()
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)

    f_name = 'penta'
    folder_name = 'stick'
    transmat4 = rm.homomat_from_posrot((.4, .3, bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    z_range = (.12, .15)
    line_thresh = 0.003
    line_size_thresh = 200

    """
    init class
    """
    bs = b_sim.BendSim(show=True)
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_x_rgt = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="rgt_arm")
    mp_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtx, armname="lft_arm")

    # mp_x_lft.goto_init_x(speed_n=200)
    # mp_x_rgt.goto_init_x(speed_n=200)

    # mp_x_rgt.goto_armjnts_x(armjnts=np.array([-0.0137881, -0.97703532, -1.50848807, 0.87929688, -1.99840199,
    #                                           0.13788101, 1.51669112]))

    '''
    show result
    '''
    goal_pseq, bendset = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendseq.pkl', 'rb'))
    _, bendresseq = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_bendresseq.pkl', 'rb'))
    pathseq_list = pickle.load(
        open(f'{config.ROOT}/bendplanner/planres/{folder_name}/{f_name}_pathseq.pkl', 'rb'))

    for bendres in bendresseq:
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendres

    grasp, pathseq = pathseq_list[0]
    for i, path in enumerate(pathseq):
        eepos, eerot = mp_x_lft.get_ee(armjnts=mp_x_lft.get_armjnts())
        print(eepos)
        init_a, end_a, plate_a, pseq_init, rotseq_init, pseq_end, rotseq_end = bendresseq[i]
        bend_a = np.degrees(end_a - init_a) + 5
        print(bend_a)
        if len(path) == 1:
            mp_x_lft.goto_armjnts_x(path[0])
        else:
            mp_x_lft.movepath(path)
        motor.rot_degree(clockwise=0, rot_deg=bend_a)
        goal = _action(os.path.join(folder_name, f_name), f"{str(i)}_goal.pkl",
                       bend_a, z_range, line_thresh=line_thresh, line_size_thresh=line_size_thresh, ulim=None,
                       rgba=(0, 1, 0, 1))
        motor.rot_degree(clockwise=1, rot_deg=bend_a)
        res = _action(os.path.join(folder_name, f_name), f"{str(i)}_res.pkl",
                      bend_a, z_range, line_thresh=line_thresh, line_size_thresh=line_size_thresh, ulim=None,
                      rgba=(0, 1, 0, 1))

        time.sleep(3)
    # _action(os.path.join(folder_name, f_name), f"tst_goal.pkl", 75, z_range,
    #         line_thresh=line_thresh, line_size_thresh=line_size_thresh, ulim=None, rgba=(0, 1, 0, 1))
    # base.run()
