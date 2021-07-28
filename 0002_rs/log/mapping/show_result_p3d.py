import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

import config
import localenv.envloader as el
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.pprj_utils as ppu
import utils.prj_utils as pu
import utils.run_utils as ru


def clearbase(sceneflag=3):
    for i in base.render.children:
        if sceneflag > 0:
            sceneflag -= 1
        else:
            i.removeNode()


def get_item(dump_f_name):
    if dump_f_name == 'cylinder_cad':
        item = el.loadObjitem('cylinder.stl')
    elif dump_f_name == 'ball':
        item = el.loadObjitem('ball.stl')
    elif dump_f_name == 'cube':
        item = el.loadObjitem('cube.stl')
    elif dump_f_name == 'helmet':
        phoxi_f_path = 'phoxi_tempdata_0525.pkl'
        item = ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                   reconstruct_surface=True, sample_num=SAMPLE_NUM,
                                   x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
    elif dump_f_name == 'cylinder_pcd':
        phoxi_f_path = 'phoxi_tempdata_0524.pkl'
        item = ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                   reconstruct_surface=True, sample_num=SAMPLE_NUM,
                                   x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
    else:
        return None
    return item


if __name__ == '__main__':
    import pandaplotutils.pandactrl as pc
    from panda3d.core import NodePath

    # import direct.directbase.DirectStart

    SAMPLE_NUM = 100000
    OBJPOS = np.asarray([0, 0, 0])

    # base.pggen.plotAxis(base.render)
    phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)

    dump_f_name = 'cube'
    # dump_f_name = 'cylinder_cad'
    # dump_f_name = 'ball'
    # dump_f_name = 'cylinder_pcd'
    # dump_f_name = 'helmet'
    item = get_item(dump_f_name)
    if dump_f_name == 'ball':
        DRAWREC_SIZE = [60, 60]
    else:
        DRAWREC_SIZE = [80, 80]

    center = pcdu.get_pcd_center(item.pcd)
    if dump_f_name == 'cube':
        base = pc.World(camp=[center[0] + 100, center[1] - 200, center[2] + 200],
                        lookatpos=[center[0], center[1], center[2]], w=500, h=500)
    elif dump_f_name == 'helmet':
        base = pc.World(camp=[center[0], center[1], center[2] + 300], lookatpos=[center[0], center[1], center[2]],
                        w=500, h=500)
    else:
        base = pc.World(camp=[center[0], center[1], center[2] + 200], lookatpos=[center[0], center[1], center[2]],
                        w=500, h=500)

    # tmp = NodePath()
    # tmp.reparentTo(base.render)
    # tmp.detachNode()

    # method_list = ['quad', 'RBF']
    # step_list = ['s20']

    # step_list = ['s1', 's2', 's5', 's10', 's20']
    # # step_list = ['s5', 's10', 's20']
    # # method_list = ['quad', 'RBF', 'RBF-G']
    # method_list = ['bs']
    # for i, method in enumerate(method_list):
    #     for step in step_list:
    #         item.show_objcm(rgba=(1, 1, 1, 1))
    #         f_name = f'{dump_f_name}_{step}_{method}'
    #         res_dict = pickle.load(open(f'{f_name}.pkl', 'rb'))
    #         pos_nrml_list = res_dict['pos_nrml_list']
    #         color = (0, 0, 1)
    #         print(method, color)
    #         pu.show_drawpath(pu.flatten_nested_list(pos_nrml_list), color=color)
    #         base.graphicsEngine.renderFrame()
    #         base.win.saveScreenshot(base.pg.Filename(f'img/{f_name}.jpg'))
    #         clearbase()
    # base.run()

    # method_list = ['DI', 'EI', 'SI', 'II', 'quad', 'RBF', 'RBF-G']
    method_list = ['RBF-G']
    for i, method in enumerate(method_list):
        item = get_item(dump_f_name)
        if dump_f_name == 'ball':
            item.show_objcm(rgba=(1, 1, 1, .5))
        else:
            item.show_objcm(rgba=(1, 1, 1, 1))
        f_name = f'{dump_f_name}_{method}'
        res_dict = pickle.load(open(f'{f_name}.pkl', 'rb'))
        pos_nrml_list = res_dict['pos_nrml_list']
        color = (0, 0, 1)
        print(method, color)
        pu.show_drawpath(pu.flatten_nested_list(pos_nrml_list), color=color)
        connection_num, global_error, global_angle_div = pu.get_connection_error(pos_nrml_list, size=DRAWREC_SIZE, step=10)
        base.run()

        base.graphicsEngine.renderFrame()
        base.win.saveScreenshot(base.pg.Filename(f'img/{f_name}_ge.jpg'))
        clearbase()

    base.run()
