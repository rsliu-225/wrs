import pickle
import numpy as np
import basis.robot_math as rm
import motionplanner.motion_planner as m_planner
import bendplanner.BendSim as b_sim
import localenv.envloader as el
import bender_config as bconfig


def transseq(pseq, rotseq, transmat4):
    return rm.homomat_transform_points(transmat4, pseq).tolist(), \
           np.asarray([transmat4[:3, :3].dot(r) for r in rotseq])


if __name__ == '__main__':
    base, env = el.loadEnv_yumi()
    # rbt = el.loadYumi(showrbt=False)
    # transmat4 = rm.homomat_from_posrot((.4, 0, bconfig.BENDER_H + .2), rm.rotmat_from_axangle((0, 0, 1), np.pi))

    rbt = el.loadUr3e(showrbt=False)
    transmat4 = rm.homomat_from_posrot((.8, 0, .90 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 1, 0), np.pi / 2))
    pseq, rotseq = pickle.load(open('./tmp.pkl', 'rb'))
    pseq, rotseq = transseq(pseq, rotseq, transmat4)

    bs = b_sim.BendSim(show=False, granularity=np.pi / 90, cm_type='plate')
    mp_lft = m_planner.MotionPlanner(env, rbt, armname="lft_arm")
    mp_rgt = m_planner.MotionPlanner(env, rbt, armname="rgt_arm")
    grasp_list = mp_lft.load_all_grasp('plate')

    ptsseq = []
    for a in range(0, 45):
        bendset = [[np.radians(-a), np.radians(0), np.radians(180), .12]]
        bs.reset([(0, 0, 0), (0, max(np.asarray(bendset)[:, 3]) + .15, 0)], [np.eye(3), np.eye(3)], extend=False)
        pseq, rotseq = transseq(bs.pseq, bs.rotseq, transmat4)
        ptsseq.append([pseq, rotseq])

    for i, grasp in enumerate(grasp_list):
        print(f'----------grasp_id: {i}  of {len(grasp_list)}----------')
        for pseq,rotseq in ptsseq:
            bs.reset(pseq, rotseq, extend=False)
            bs.show()
        _, _, objpos, objrot = bs.get_posrot_by_l(.08, pseq, rotseq)
        armjnts_lft = mp_lft.get_armjnts_by_objmat4ngrasp(grasp, [], rm.homomat_from_posrot(objpos, objrot),
                                                          obj=bs.objcm)
        print(armjnts_lft)
        if armjnts_lft is not None:
            break

    for i, grasp in enumerate(grasp_list):
        print(f'----------grasp_id: {i}  of {len(grasp_list)}----------')
        bs.reset(pseq, rotseq, extend=False)
        _, _, objpos, objrot = bs.get_posrot_by_l(.2, pseq, rotseq)
        armjnts_rgt = mp_rgt.get_armjnts_by_objmat4ngrasp(grasp, [], rm.homomat_from_posrot(objpos, objrot),
                                                          obj=bs.objcm)
        print(armjnts_rgt)
        if armjnts_rgt is not None:
            break

    # rbt.fk(component_name=mp_lft.armname, jnt_values=armjnts_lft)
    # rbt.fk(component_name=mp_rgt.armname, jnt_values=armjnts_rgt)
    rbt.jaw_to(mp_lft.hnd_name, jaw_width=.001)
    rbt.jaw_to(mp_rgt.hnd_name, jaw_width=.001)
    rbt.gen_meshmodel().attach_to(base)
    base.run()
