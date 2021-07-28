import os
import pickle

import config
import motionplanner.motion_planner as mp
from localenv import envloader as el


def load_path(folder_name, method_name, grasp_id):
    for dirpath, dirnames, filenames in os.walk(f"{folder_name}/"):
        for f in filenames:
            if f.endswith(".pkl") and f.find(f"{method_name}_{grasp_id}_") != -1:
                print(f)
                return pickle.load(open(os.path.join(folder_name, f), "rb"))
    print("File not found!")
    return None, None, None


if __name__ == '__main__':
    import utils.phoxi as phoxi
    import utils.phoxi_locator as pl

    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e(showrbt=False)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
    exp_name = "cylinder_cad"
    phoxi_f_name = f"phoxi_tempdata_{exp_name}.pkl"

    mp_lft = mp.MotionPlanner(env, rbt, rbtmg, rbtball, armname="lft")
    # obj_item = ru.get_obj_from_phoxiinfo_nobgf(phxilocator, phoxi_f_name=phoxi_f_name, load=True,
    #                                            reconstruct_surface=True,
    #                                            x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
    # obj_item.show_objcm()
    # objcm = el.loadObj("bowl.stl", pos=(800, 100, 780))
    # objcm = el.loadObj("box.stl", pos=(800, 200, 780))
    objcm = el.loadObj("bucket.stl", pos=(800, 200, 780))
    objcm.reparentTo(base.render)

    pen = el.loadObj("pentip.stl")
    # objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + "/cylinder_pcd_circle.pkl", "rb"))
    # objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + "/bowl_cad_circle.pkl", "rb"))
    # objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + "/box_cad_circle.pkl", "rb"))
    objmat4_list = pickle.load(open(config.PENPOSE_REL_PATH + "/bucket_cad_circle.pkl", "rb"))
    grasp_list = pickle.load(
        open(config.PREGRASP_REL_PATH + config.PEN_STL_F_NAME.split(".stl")[0] + "_pregrasps.pkl", "rb"))
    folder_name = "./nlopt"
    method_name = "boxcolw45"
    grasp_id_list = range(62)

    # mp_lft.ah.show_objmat4_list(objmat4_list, pen, rgba=(1, 1, 1, .1))
    mp_lft.ah.show_objmat4_list_pos(objmat4_list, rgba=(1, 0, 0, 1))
    for grasp_id in grasp_id_list:
        objrelpos, objrelrot, path = load_path(folder_name, method_name, grasp_id)
        if path is not None and len(path) == 129:
            mp_lft.ah.show_animation_hold(path, pen, objrelpos, objrelrot)

    # fig = plt.figure(1, figsize=(6.4, 4.8))
    # plt.ion()
    # mp_lft.rbth.plot_armjnts(path)
    # plt.show()
    # plt.savefig(f"{f_name}.png")
    # plt.close(fig)

    base.run()
