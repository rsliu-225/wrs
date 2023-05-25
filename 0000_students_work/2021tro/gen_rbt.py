import copy
import math

import numpy as np

import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import visualization.panda.world as wd


def gen_circle(r=.02, interval=15):
    p_list = []
    for i in range(-180, 180, interval):
        p_list.append((r * math.sin(math.radians(i)), r * math.cos(math.radians(i))))
    return p_list


if __name__ == '__main__':
    base = wd.World(cam_pos=[1, 0, .5], lookat_pos=[0, 0, .1])
    # gm.gen_frame().attach_to(base)
    p_list = gen_circle(r=.03)
    rbt = ur3ed.UR3EDual()
    # rbt.gen_meshmodel().attach_to(base)

    '''
    Fig 1
    '''
    # pencm = cm.CollisionModel(initor="./objects/pentip.stl")
    # pencm.set_rgba([.3, .3, .3, .3])
    #
    # cylindercm = cm.CollisionModel(initor="./objects/cylinder.stl")
    # cylindercm.set_rgba([1, 1, 1, 1])
    # cylindercm.set_rotmat(rm.rotmat_from_euler(0, 0, 0))
    # cylindercm.attach_to(base)
    # gm.gen_circarrow()
    #
    # for i, p in enumerate(p_list[::-1]):
    #     # gm.gen_sphere(pos=(p[0], p[1], .1), radius=.001).attach_to(base)
    #     prj_p, prj_n = cylindercm.ray_hit(point_from=np.asarray((p[0], p[1], .1)), point_to=np.asarray((p[0], p[1], 0)),
    #                                       option="closest")
    #     prj_p = np.asarray(prj_p)
    #     prj_n = np.asarray(prj_n)
    #     gm.gen_sphere(pos=prj_p, radius=.001, rgba=(1, 1, 0, 1)).attach_to(base)
    #     if i in range(12, 18):
    #         penlocal = copy.deepcopy(pencm)
    #         rot = rm.rotmat_between_vectors(np.asarray([1, 0, 0]), prj_n)
    #         penmat4 = rm.homomat_from_posrot(prj_p, rot)
    #         penlocal.set_homomat(penmat4)
    #         penlocal.set_rgba([.7, .7, .3, .5])
    #         penlocal.attach_to(base)
    #         # gm.gen_arrow(spos=prj_p, epos=prj_p + prj_n * .05, rgba=(1, 0, 0, 1), thickness=.002).attach_to(base)
    #         gm.gen_dasharrow(spos=prj_p, epos=prj_p + prj_n * .05, rgba=(0, 1, 1, 1),thickness=.002).attach_to(base)
    #         gm.gen_cone(spos=prj_p + prj_n * .07, epos=prj_p, radius=.025, sections=50).attach_to(base)
    #         # if i == 17:
    #         #     penlocal.show_localframe()
    # base.run()

    '''
    Fig 2
    '''
    pencm = cm.CollisionModel(initor="./objects/pentip2.stl")
    # gm.gen_frame(length=.14, thickness=.005).attach_to(base)
    hnd = rtqhe.RobotiqHE()
    p = np.asarray([0, 0, 0])
    rot = rm.rotmat_from_euler(math.pi, 0, 0)
    penmat4 = rm.homomat_from_posrot(p, rot)
    hnd.grip_at_with_jcpose(np.asarray([.1, .006, 0]), rm.rotmat_from_euler(math.pi, math.pi / 2, math.pi / 3), .02)
    hnd.gen_meshmodel().attach_to(base)
    spos = np.asarray([0, 0, 0])
    epos = np.asarray([.15, -.03, 0])
    vec = rm.unit_vector(epos - spos)

    gm.gen_cone(spos=np.asarray([.15, -.03, 0]), epos=np.asarray([0, 0, 0]), radius=.06, sections=50).attach_to(base)
    # gm.gen_dasharrow(spos=np.asarray([0, 0, 0]), epos=np.asarray([.15, -.03, 0]),rgba=(0,1,1,1)).attach_to(base)
    gm.gen_dasharrow(spos=spos, epos=spos + vec * .2, rgba=(0, 1, 1, 1)).attach_to(base)
    vec_t = rm.unit_vector(rm.orthogonal_vector(vec))
    gm.gen_dasharrow(spos=spos, epos=spos + vec_t * .1, rgba=(1, 0, 1, 1)).attach_to(base)
    vec_n = rm.unit_vector(np.cross(a=vec, b=vec_t))
    gm.gen_dasharrow(spos=spos, epos=spos + vec_n * .1, rgba=(1, 1, 0, 1)).attach_to(base)

    pencm.set_homomat(penmat4)
    pencm.set_rgba([.7, .7, .3, .5])
    pencm.attach_to(base)
    gm.gen_arrow(spos=p, epos=p + rot[:3, 0] * .2, rgba=(1, 0, 0, 1)).attach_to(base)

    # pencm.show_localframe()
    base.run()

    '''
    Fig 3
    '''
    # gm.gen_frame(length=.14, thickness=.005).attach_to(base)
    # pencm = cm.CollisionModel(initor="./objects/pentip2.stl")
    # buckercm = cm.CollisionModel(initor="./objects/bucket.stl")
    #
    # hnd = rtqhe.RobotiqHE()
    # penpos = np.asarray([0, 0, 0])
    # penrot = rm.rotmat_from_euler(math.pi, 0, 0)
    # penmat4 = rm.homomat_from_posrot(penpos, penrot)
    # hndpos = np.asarray([.08, 0, 0])
    # hndrot = rm.rotmat_from_euler(math.pi, math.pi / 2, math.pi / 4)
    # hnd.grip_at_with_jcpose(np.asarray([.08, 0, 0]), hndrot, .02)
    # hnd.gen_meshmodel(rgba=[.1, .1, .1, .7]).attach_to(base)
    # gm.gen_cone(spos=np.asarray([.15, 0, 0]), epos=np.asarray([0, 0, 0]), radius=.06, sections=50).attach_to(base)
    #
    # pencm.set_homomat(penmat4)
    # pencm.set_rgba([.7, .7, .3, .7])
    # pencm.attach_to(base)
    # gm.gen_stick(spos=penpos + penrot[:3, 0] * .015, epos=np.asarray([.164, 0, 0])).attach_to(base)
    # gm.gen_stick(spos=hndpos + hndrot[:3, 2] * .014, epos=hndpos - hndrot[:3, 2] * .15).attach_to(base)
    #
    # bucketmat4 = rm.homomat_from_posrot(np.array([.06, -.12, 0]), rm.rotmat_from_euler(0, math.pi / 2, math.pi / 3))
    # buckercm.set_homomat(bucketmat4)
    # buckercm.set_rgba([.7, .7, .7, .3])
    # buckercm.attach_to(base)
    # pts, _ = buckercm.sample_surface(toggle_option='normals', radius=.01)
    # # gm.GeometricModel(pts).attach_to(base)
    # base.run()
