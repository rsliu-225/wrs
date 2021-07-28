from prj_utils import *

if __name__ == '__main__':
    """
    set up env and param
    """
    import pickle
    import os
    import pandaplotutils.pandactrl as pc

    SNAP_QI = False
    SNAP_SFC_G = False
    SNAP_SFC = True

    dump_f_name = 'helmet'
    DRAWREC_SIZE = [80, 80]
    stl_f_name = None

    if dump_f_name == 'ball':
        stl_f_name = 'ball_surface.stl'
        DRAWREC_SIZE = [60, 60]
        SAMPLE_NUM = 10
    elif dump_f_name == 'cylinder_cad':
        stl_f_name = 'cylinder_surface.stl'
        SAMPLE_NUM = 10000
    elif dump_f_name == 'cube':
        stl_f_name = 'cube_surface_2.stl'
        SAMPLE_NUM = 10000
    elif dump_f_name == 'helmet':
        phoxi_f_path = 'phoxi_tempdata_0525.pkl'
        SAMPLE_NUM = 10
    elif dump_f_name == 'cylinder_pcd':
        phoxi_f_path = 'phoxi_tempdata_0524.pkl'
        SAMPLE_NUM = 10
    else:
        SAMPLE_NUM = None
        print('error')
    # objpos = (800, 200, 780)

    """
    load mesh model
    """
    if stl_f_name is None:
        phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
        tgt_item = ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                       reconstruct_surface=True, sample_num=SAMPLE_NUM,
                                       x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
    else:
        objpos = (0, 0, 0)
        # objrot = (0, 180, 0)
        objrot = (0, 0, 0)
        tgt_item = el.loadObjitem(stl_f_name, pos=objpos, rot=objrot, sample_num=SAMPLE_NUM)
    direction = np.asarray((0, 0, 1))
    # direction = np.asarray((0, -1, 0))
    if dump_f_name == 'cube':
        tgt_item.set_drawcenter((0, -25, 0))  # cube
    if dump_f_name == 'cylinder_pcd':
        tgt_item.set_drawcenter((0, -5, 0))  # cylinder_pcd
    if dump_f_name == 'bowl':
        tgt_item.set_drawcenter((-60, -60, 0))  # bowl
    if dump_f_name == 'box':
        tgt_item.set_drawcenter((60, 50, 20))

    center = pcdu.get_pcd_center(tgt_item.pcd)
    base = pc.World(camp=[center[0], center[1], center[2] + 300],
                    lookatpos=[center[0], center[1], center[2]], w=500, h=500)
    # base.pggen.plotAxis(base.render)
    # tgt_item.show_objcm(rgba=[.7, .7, .3, .3])
    # base.run()

    """
    multiple strokes
    """
    error_method = 'ED'
    for step in [1, 2, 5, 10, 20]:
        if step > 5:
            snap = True
        else:
            snap = False
        drawpath_ms = du.gen_grid(side_len=int(DRAWREC_SIZE[0]), grid_len=10, step=step)

        # pos_nrml_list, _, time_cost = \
        #     prj_drawpath_ms_on_pcd(tgt_item, drawpath_ms, mode='gaussian', step=1, error_method=error_method,
        #                            toggledebug=False)
        # print('gussian time cost', time_cost)
        # show_drawpath(flatten_nested_list(pos_nrml_list), color=(1, 0, 0), show_nrmls=False, transparency=1)
        # dump_mapping_res(f'{dump_f_name}_s{str(step)}_gaussian.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)
        #
        # pos_nrml_list, _, time_cost = \
        #     prj_drawpath_ms_on_pcd(tgt_item, drawpath_ms, mode='EI', step=1, error_method=error_method,
        #                            toggledebug=False)
        # print('metrology method time cost', time_cost)
        # show_drawpath(flatten_nested_list(pos_nrml_list), color=(1, 1, 0), show_nrmls=False, transparency=1)
        # dump_mapping_res(f'{dump_f_name}_s{str(step)}_EI.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)
        #
        # pos_nrml_list, _, time_cost = \
        #     prj_drawpath_ms_on_pcd(tgt_item, drawpath_ms, mode='quad', step=1, error_method=error_method,
        #                            toggledebug=False)
        # print('quadratic time cost', time_cost)
        # show_drawpath(flatten_nested_list(pos_nrml_list), color=(1, 0, 0), show_nrmls=False, transparency=1)
        # dump_mapping_res(f'{dump_f_name}_s{str(step)}_quad.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)
        #
        # pos_nrml_list, _, time_cost = \
        #     prj_drawpath_ms_on_pcd(tgt_item, drawpath_ms, mode='rbf', step=1, error_method=error_method,
        #                            toggledebug=False)
        # print('rbf time cost', time_cost)
        # show_drawpath(flatten_nested_list(pos_nrml_list), color=(0, 1, 0), show_nrmls=False, transparency=1)
        # dump_mapping_res(f'{dump_f_name}_s{str(step)}_RBF.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)
        #
        # KERNEL = 'thin_plate_spline'
        # pos_nrml_list, _, time_cost = \
        #     prj_drawpath_ms_on_pcd(tgt_item, drawpath_ms, mode='rbf_g', step=1, error_method=error_method,
        #                            toggledebug=False)
        # print('rbf time cost', time_cost)
        # show_drawpath(flatten_nested_list(pos_nrml_list), color=(0, 0, 1), show_nrmls=False, transparency=1)
        # dump_mapping_res(f'{dump_f_name}_s{str(step)}_RBF-G.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)

        pos_nrml_list, _, time_cost = \
            prj_drawpath(tgt_item, drawpath_ms, mode='bs', step=1, error_method=error_method,
                         toggledebug=False, snap=snap)
        print('rbf time cost', time_cost)
        show_drawpath(flatten_nested_list(pos_nrml_list), color=(0, 0, 1), show_nrmls=False, transparency=1)
        dump_mapping_res(f'{dump_f_name}_s{str(step)}_bs.pkl', tgt_item, drawpath_ms, pos_nrml_list, time_cost)

    base.run()
