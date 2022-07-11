import pickle

import numpy as np

import basis.robot_math as rm
import bendplanner.BendSim as b_sim
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import config
import modeling.geometric_model as gm
import visualization.panda.world as wd

if __name__ == '__main__':
    # base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
    base = wd.World(cam_pos=[.2, .2, .2], lookat_pos=[0, 0, 0])

    f_name = 'skull'

    bs = b_sim.BendSim(show=True, granularity=np.pi / 180, cm_type='plate')

    transmat4 = rm.homomat_from_posrot((.9, -.35, .78 + bconfig.BENDER_H), rm.rotmat_from_axangle((0, 0, 1), np.pi))
    goal_pseq, goal_rotseq = pickle.load(open(config.ROOT + f'/data/bend/rotpseq/{f_name}.pkl', 'rb'))
    init_pseq = [(0, 0, 0), (0, bu.cal_length(goal_pseq), 0)]
    init_rotseq = [np.eye(3), np.eye(3)]

    fit_pseq, fit_rotseq = bu.decimate_rotpseq(goal_pseq, goal_rotseq, tor=.0002, toggledebug=False)
    bendset = bu.rotpseq2bendset(fit_pseq, fit_rotseq, toggledebug=True)
    init_rot = fit_rotseq[0]

    bs.reset(init_pseq, init_rotseq, extend=True)
    bs.move_to_org(bu.cal_length(goal_pseq))
    is_success, bendresseq, _ = bs.gen_by_bendseq(bendset, cc=False, toggledebug=False)
    # bs.show_bendresseq(bendresseq, is_success)

    goal_pseq, goal_rotseq = bu.align_with_init(bs, goal_pseq, init_rot, goal_rotseq)
    fit_pseq, fit_rotseq = bu.align_with_init(bs, fit_pseq, init_rot, fit_rotseq)
    goal_cm = bu.gen_surface(fit_pseq, fit_rotseq, bconfig.THICKNESS / 2, width=bconfig.WIDTH)
    goal_cm.attach_to(base)
    for i in range(len(fit_rotseq)):
        gm.gen_frame(fit_pseq[i], fit_rotseq[i], length=.01, thickness=.0005).attach_to(base)
    # err, _ = bu.avg_polylines_dist_err(res_pseq, goal_pseq, toggledebug=True)
    kpts2 = bu.mindist_err(bs.pseq[1:], goal_pseq, res_rs=bs.rotseq[1:], goal_rs=goal_rotseq, toggledebug=True)

    # pickle.dump(res_pseq, open('res.pkl', 'wb'))
    # res_pseq_tmp = pickle.load(open('res.pkl', 'rb'))
    # ax = plt.axes(projection='3d')
    # ax.set_xlim([-0.05, 0.05])
    # ax.set_ylim([-0.02, 0.08])
    # ax.set_zlim([-0.05, 0.05])
    # bu.plot_pseq(ax, res_pseq, c='r')
    # bu.plot_pseq(ax, res_pseq_tmp, c='g')
    # bu.plot_pseq(ax, goal_pseq, c='black')
    # plt.show()
    bs.show(rgba=(0, .7, 0, .7), show_frame=True)
    base.run()
