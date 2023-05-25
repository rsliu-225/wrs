import pickle
import config
import time

from localenv import envloader as el
import motionplanner.motion_planner as m_planner
import motionplanner.rbtx_motion_planner as m_plannerx
import utils.run_utils as ru
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl

if __name__ == '__main__':
    '''
    set up env and param
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    rbtx = el.loadUr3ex(rbt)
    rbt.opengripper(armname="rgt")
    rbt.opengripper(armname="lft")
    dump_id = '210527'
    amat_f_name = f"/phoxi_calibmat_{dump_id}.pkl"
    phoxi_f_path = "tst_calibmat.pkl"

    pen_f_name = "pentip"

    match_rotz = True
    load = False
    resolution = 1

    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=amat_f_name)

    '''
    init planner
    '''
    motion_planner_lft = m_planner.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    motion_planner_x_lft = m_plannerx.MotionPlannerRbtX(env, rbt, rbtmg, rbtball, rbtx, armname="lft")
    motion_planner_x_lft.goto_init_x()
    time.sleep(1)

    '''
    process image
    '''
    pen_item = ru.get_obj_from_phoxiinfo_withmodel(phxilocator, pen_f_name + ".stl", phoxi_f_name=phoxi_f_path,
                                                   match_rotz=match_rotz, load=load, resolution=resolution,
                                                   x_range=(200, 900), y_range=(-100, 400), z_range=(790, 850))
    pen_cm = pen_item.objcm
    pen_item.show_objpcd(rgba=(1,0,0,1))
    pen_item.show_objcm()

    '''
    set obj start/goal position/rotation
    '''
    # draw the object at the initial object pose
    objmat4_init = pen_item.objmat4
    print("Pen origin position:", objmat4_init[:3, 3])

    '''
    get available grasp
    '''
    pen_grasplist_all = pickle.load(open(config.ROOT + "/graspplanner/pregrasp/" + pen_f_name + "_pregrasps.pkl", "rb"))

    '''
    plan pen grasp motion
    '''
    for grasp in pen_grasplist_all:
        objrelpos, objrelrot = motion_planner_lft.get_rel_posrot(grasp, objmat4_init[:3, 3], objmat4_init[:3, :3])
        if objrelpos is None:
            continue

        print("---------------init to pick---------------")
        path_init2pick = motion_planner_lft.plan_gotopick(grasp, objmat4_init, pen_item.objcm, objrelpos,
                                                          objrelrot)
        if path_init2pick is not None :
            motion_planner_x_lft.movepath(path_init2pick)
            motion_planner_lft.ah.show_animation_hold(path_init2pick, pen_cm, objrelpos, objrelrot)
            base.run()
