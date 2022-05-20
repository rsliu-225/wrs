import pickle
import copy
import numpy as np
import basis.robot_math as rm
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import localenv.envloader as el
import bender_config as bconfig
import modeling.geometric_model as gm


def transseq(pseq, rotseq, transmat4):
    return np.asarray(rm.homomat_transform_points(transmat4, pseq)), \
           np.asarray([transmat4[:3, :3].dot(r) for r in rotseq])


if __name__ == '__main__':
    base, env = el.loadEnv_yumi()
    # rbt = el.loadYumi(showrbt=False)
    # transmat4 = rm.homomat_from_posrot((.4, 0, bconfig.BENDER_H + .2), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    rbt = el.loadUr3e(showrbt=False)
    transmat4 = rm.homomat_from_posrot((.75, -.1, .8 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    # pseq, rotseq = transseq(pseq, rotseq, transmat4)

    bs = b_sim.BendSim(show=False, granularity=np.pi / 90, cm_type='plate')
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    grasp_list = mp_lft.load_all_grasp('plate')

    ptsseq = []
    for a in range(5, 90, 10):
        bendset = [[np.radians(-a), np.radians(0), np.radians(180), .1]]
        bs.reset([(0, 0, 0), (0, max(np.asarray(bendset)[:, 3]) + .3, 0)], [np.eye(3), np.eye(3)], extend=False)
        bs.gen_by_bendseq(bendset, cc=False)
        pseq, rotseq = transseq(copy.deepcopy(bs.pseq), copy.deepcopy(bs.rotseq), transmat4)
        for i in range(len(pseq) - 1):
            if a == 90:
                gm.gen_stick(pseq[i], pseq[i + 1], thickness=.005).attach_to(base)
            else:
                gm.gen_stick(pseq[i], pseq[i + 1], rgba=(1, 0, 0, .2), thickness=.005).attach_to(base)

        ptsseq.append([pseq, rotseq])

    armjnts_lft_list = []
    for i, grasp in enumerate(grasp_list):
        print(f'----------grasp_id: {i}  of {len(grasp_list)}----------')
        for pseq, rotseq in ptsseq:
            bs.reset(pseq, rotseq, extend=False)
            _, _, objpos, objrot = bs.get_posrot_by_l(.3, pseq, rotseq)
            # gm.gen_frame(objpos, objrot).attach_to(base)
            armjnts_lft = \
                mp_lft.get_armjnts_by_objmat4ngrasp(grasp, [], rm.homomat_from_posrot(objpos, objrot),
                                                    obj=bs.objcm,
                                                    msc=armjnts_lft_list[-1] if len(armjnts_lft_list) != 0 else None)
            print(armjnts_lft)
            if armjnts_lft is None:
                armjnts_lft_list = []
                break
            armjnts_lft_list.append(armjnts_lft)
        if len(armjnts_lft_list) != 0:
            break

    grasp_list_rgt = mp_lft.load_all_grasp('plate_rgt')
    pseq, rotseq = ptsseq[0]
    for i, grasp in enumerate(grasp_list_rgt):
        print(f'----------grasp_id: {i}  of {len(grasp_list_rgt)}----------')
        bs.reset(pseq, rotseq, extend=False)
        _, _, objpos, objrot = bs.get_posrot_by_l(.08, pseq, rotseq)
        armjnts_rgt = mp_rgt.get_armjnts_by_objmat4ngrasp(grasp, [], rm.homomat_from_posrot(objpos, objrot),
                                                          obj=bs.objcm)
        print(armjnts_rgt)
        if armjnts_rgt is not None:
            break

    rbt.jaw_to(mp_lft.hnd_name, jaw_width=.001)
    rbt.jaw_to(mp_rgt.hnd_name, jaw_width=.001)
    for i in range(len(armjnts_lft_list)):
        armjnts_lft = armjnts_lft_list[i]
        rbt.fk(component_name=mp_lft.armname, jnt_values=armjnts_lft)
        if i == 0 or i == len(armjnts_lft_list) - 1:
            rbt.gen_meshmodel().attach_to(base)
        else:
            rbt.gen_meshmodel(rgba=(0, 1, 0, .3)).attach_to(base)

    # rbt.fk(component_name=mp_lft.armname, jnt_values=armjnts_lft)
    # rbt.fk(component_name=mp_rgt.armname, jnt_values=armjnts_rgt)

    # rbt.gen_meshmodel().attach_to(base)
    base.run()
