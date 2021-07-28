import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import time

import config
import localenv.envloader as el
import utils.pcd_utils as pcdu
import utils.phoxi as phoxi
import utils.phoxi_locator as pl
import utils.pprj_utils as ppu
import utils.prj_utils as pu
import utils.run_utils as ru
import utils.math_utils as mu
import surface as sfc


def plot_error_method(error_list_ED, error_list_GD, title=''):
    plt.plot(error_list_ED[1:], label='ED', alpha=0.8)
    plt.plot(error_list_GD[1:], label='GD', alpha=0.8)

    plt.legend()
    plt.xlabel('point id')
    plt.ylabel('error')
    # plt.title('projection error')
    plt.title(title)
    plt.show()


def compare_error_method(result_f_name):
    res_dict = pickle.load(open(result_f_name, 'rb'))
    pos_nrml_list = res_dict['pos_nrml_list']
    drawpath_ms = res_dict['drawpath']

    stroke_error_ED, point_error_ED = cal_error(pos_nrml_list, drawpath_ms, method='ED')
    stroke_error_GD, point_error_GD = cal_error(pos_nrml_list, drawpath_ms, method='GD', objpcd=res_dict['objpcd'])

    print(f'avg. error(ED):{np.mean(stroke_error_ED)}')
    print(f'avg. error(GD):{np.mean(stroke_error_GD)}')

    plot_error_method(point_error_ED, point_error_GD, title=result_f_name)


def cal_error(pos_nrml_list, drawpath, method='ED', objpcd=None, mode='ms', objcm=None):
    if method in ['GD', 'rbf', 'rbf-g']:
        kdt_d3, _ = pu.get_kdt(objpcd)
    else:
        kdt_d3 = None
    surface = None
    transmat = np.eye(4)
    if method == 'rbf-g':
        time_start = time.time()
        pca_trans = True
        if len(objpcd) > 50000:
            objpcd = np.asarray(random.choices(objpcd, k=40000))
            objpcd = np.array(list(set([tuple(t) for t in objpcd])))

        if pca_trans:
            pcd_tr, transmat = mu.trans_data_pcv(objpcd, random_rot=False)
            surface = sfc.RBFSurface(pcd_tr[:, :2], pcd_tr[:, 2])
        else:
            transmat = np.eye(3)
            surface = sfc.RBFSurface(objpcd[:, :2], objpcd[:, 2])
        time_cost_rbf = time.time() - time_start
        print('time cost(rbf global):', time_cost_rbf)

        surface_cm = surface.get_gometricmodel(rgba=[.8, .8, .1, 1])
        mat4 = np.eye(4)
        mat4[:3, :3] = transmat
        surface_cm.sethomomat(mat4)
        surface_cm.reparentTo(base.render)
        pcdu.show_pcd(objpcd)
        # base.run()

    if mode == 'ms':
        stroke_error = []
        point_error = []
        for i, stroke in enumerate(drawpath):
            error, error_list = pu.get_prj_error(stroke, pos_nrml_list[i], method=method, kdt_d3=kdt_d3, pcd=objpcd,
                                                 objcm=objcm, surface=surface, transmat=transmat)
            point_error.extend(error_list)
            stroke_error.append(error)
    else:
        error, error_list = pu.get_prj_error(drawpath, pos_nrml_list, method=method, kdt_d3=kdt_d3, pcd=objpcd,
                                             objcm=objcm, surface=surface, transmat=transmat)
        point_error = error_list
        stroke_error = error
    # base.run()
    return np.asarray(stroke_error), np.asarray(point_error)


def compare_prj_method(result_f_name_dict, error_method='ED', savef=True, objcm=None, objpcd=None):
    png_name = f'{list(result_f_name_dict.values())[0].split(".pkl")[0]}_{error_method}'
    for k, f_name in result_f_name_dict.items():
        print(f'--------{k}, {error_method}--------')
        res_dict = pickle.load(open(f_name, 'rb'))
        pos_nrml_list = res_dict['pos_nrml_list']
        # global_error = 0.0
        # connection_num = len(pos_nrml_list)
        stroke_error, point_error = \
            cal_error(pos_nrml_list, res_dict['drawpath'], method=error_method,
                      objpcd=res_dict['objpcd'] if objpcd is None else objpcd, objcm=objcm)
        connection_num, global_error, global_angle_div = \
                pu.get_connection_error(pos_nrml_list, size=DRAWREC_SIZE, step=STEP)
        # base.run()
        print(f'time cost:{res_dict["time_cost"]}')
        print(f'avg. error:{np.mean(stroke_error)}')
        print(f'global error: {global_error}({connection_num})')
        print(f'global angle deviation: {global_angle_div}({connection_num})')
        # print(f'avg. point error:{np.mean(point_error)}')
        plt.plot(point_error[1:], label=k, alpha=1)
        png_name += f'_{k}'

    plt.xlabel('Point ID')
    plt.ylabel('Error')
    plt.title('Projection Error')
    if savef:
        plt.savefig(f"./Fig/{png_name}.png")
        plt.close()


def compare_step(result_f_name_dict, error_method='ED', label='', objcm=None):
    x = []
    y = []
    for k, f_name in result_f_name_dict.items():
        print(f'--------{k}--------')
        res_dict = pickle.load(open(f_name, 'rb'))
        pos_nrml_list = res_dict['pos_nrml_list']
        stroke_error, point_error = \
            cal_error(pos_nrml_list, res_dict['drawpath'], method=error_method, objpcd=res_dict['objpcd'], objcm=objcm)
        # connection_num, global_error = pu.get_connection_error(pos_nrml_list, size=DRAWREC_SIZE, step=STEP)
        # print(f'global error: {global_error}({connection_num})')
        print(f'avg. point error:{np.mean(point_error)}')
        print(f'avg. error:{np.mean(abs(stroke_error))}')
        print(f'time cost:{res_dict["time_cost"]}')
        x.append(k)
        y.append(np.mean(stroke_error))

    plt.plot(x, y, label=label, alpha=0.8)
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('error')
    plt.title('projection error')


def compare_start(f_name, error_method='ED'):
    res_dict = pickle.load(open(f_name, 'rb'))
    error_list = []
    loop_pos_nrml_list = res_dict['pos_nrml_list']
    for i, pos_nrml_list in enumerate(loop_pos_nrml_list):
        # print(pos_nrml_list)
        stroke_error, point_error = \
            cal_error(pos_nrml_list, res_dict['drawpath'], method=error_method, objpcd=res_dict['objpcd'], mode='ss')
        error_list.append(stroke_error)
        base.pggen.plotSphere(base.render, pos_nrml_list[0][0],
                              rgba=(1 - i * 1 / len(loop_pos_nrml_list), i * 1 / len(loop_pos_nrml_list), 0, 1))

    plt.plot(error_list, label=f_name.split('_')[-1].split('.pkl')[0], alpha=0.8)
    # plt.plot(res_dict['error'], label='org', alpha=0.8)
    plt.legend()
    plt.xlabel('start point ID')
    plt.ylabel('error')
    # plt.ylim(min(error_list)-0.001, max(error_list)+0.001)
    plt.title('projection error')
    plt.show()


def cal_error_with_gt(pos_nrml_list, pos_nrml_list_gt):
    error_list = []
    pos_gt_list = np.asarray([v[0] for v in pos_nrml_list_gt])
    pos_list = np.asarray([v[0] for v in pos_nrml_list])
    kdt_d3, _ = pu.get_kdt(pos_gt_list)
    # rmse, fitness, transmat = o3dh.registration_ptpt(np.asarray(pos_gt_list), np.asarray(pos_list),
    #                                                  toggledebug=False)
    # print(rmse, fitness)
    # if rmse > 0.002:
    #     return None
    # pos_list = pcdu.trans_pcd(np.asarray(pos_list), transmat)
    for i in range(len(pos_list)):
        pos = pos_list[i]
        pos_gt = pu.get_knn(np.asarray(pos), kdt_d3, k=3)[0]
        # pos_gt, _ = pos_nrml_list_gt[i]
        error_list.append(np.linalg.norm(pos - pos_gt))
    return error_list


def compare_start_with_gt(f_name, f_name_gt):
    res_dict = pickle.load(open(f_name, 'rb'))
    res_dict_gt = pickle.load(open(f_name_gt, 'rb'))
    loop_pos_nrml_list = res_dict['pos_nrml_list']
    loop_pos_nrml_list_gt = res_dict_gt['pos_nrml_list']
    mean_error_list = []

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('start point ID')
    ax1.set_ylabel('error')
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.set_xlabel('start point ID')
    ax1.set_ylabel('avg. error')

    for i, pos_nrml_list in enumerate(loop_pos_nrml_list):
        error_list = cal_error_with_gt(pos_nrml_list, loop_pos_nrml_list_gt[i])
        if error_list is None:
            continue
        print('mean error:', np.mean(error_list))
        mean_error_list.append(np.mean(error_list))
        ax1.plot(error_list)
    ax2.plot(mean_error_list, label=f_name.split('.pkl')[0].split('_')[-1])
    plt.show()


if __name__ == '__main__':
    import pandaplotutils.pandactrl as pc

    SAMPLE_NUM = 100000
    STEP = 10
    OBJPOS = np.asarray([0, 0, 0])

    # base.pggen.plotAxis(base.render)
    # base, env = el.loadEnv_wrs()
    # stl_f_name = None
    stl_f_name = 'cylinder'

    if stl_f_name is not None:
        item = el.loadObjitem(f_name=stl_f_name + '.stl')
        base = pc.World(camp=np.array([0, 0, 1000]), lookatpos=np.array([0, 0, 50]))
        # base = pc.World(camp=np.array([500, -1000, 1000]), lookatpos=np.array([0, 0, 50]))
    else:
        phoxi_f_path = 'phoxi_tempdata_0524.pkl'
        # phoxi_f_path = 'phoxi_tempdata_helmet.pkl'
        phxilocator = pl.PhxiLocator(phoxi, amat_f_name=config.AMAT_F_NAME)
        item = ru.get_obj_by_range(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                   reconstruct_surface=True, sample_num=SAMPLE_NUM,
                                   x_range=(200, 1000), y_range=(-100, 300), z_range=(790, 1000))
        base = pc.World(camp=[600, 0, 1700], lookatpos=[600, 0, 1000])

    # dump_f_name = 'cube'
    # dump_f_name = 'cylinder_cad'
    # dump_f_name = 'cube'
    # dump_f_name = 'cylinder_cad'
    # dump_f_name = 'ball'
    # dump_f_name = 'cylinder_pcd'
    # dump_f_name = 'helmet'

    '''
    1. render mapping result;
    2. compare mapping method;
    3. compare evaluation criteria;
    4. compare different step;
    5. compare different start;
    6. compare different start, with ground truth;
    7. prospective projection
    '''

    run = 2

    if run == 1:
        import random

        dump_f_name = 'helmet'
        # method_list = ['baseline', 'DI', 'EI', 'SI', 'II', 'RBF', 'RBF-G']
        # method_list = ['quad', 'RBF', 'RBF-G', 'RBF-G-linear']
        method_list = ['RBF-G']
        item.show_objcm(rgba=(1, 1, 1, .5))
        if dump_f_name == 'ball':
            DRAWREC_SIZE = [60, 60]
        else:
            DRAWREC_SIZE = [80, 80]
        for i, method in enumerate(method_list):
            res_dict = pickle.load(open(f'{dump_f_name}_{method}.pkl', 'rb'))
            # res_dict = pickle.load(open(f'{dump_f_name}_s20_{method}.pkl', 'rb'))
            pos_nrml_list = res_dict['pos_nrml_list']
            connection_num, global_error, global_angle_div = \
                pu.get_connection_error(pos_nrml_list, size=DRAWREC_SIZE, step=STEP)
            # color = (random.choice((0, 1)), random.choice((0, 1)), random.choice((0, 1)))
            color = (0, 0, 1)
            print(method, color)
            pu.show_drawpath(pu.flatten_nested_list(pos_nrml_list), color=color)

        base.run()

    elif run == 2:
        dump_f_name = 'cylinder_pcd'
        objcm = None
        objpcd = None
        if dump_f_name == 'cylinder_cad':
            objcm = el.loadObj('cylinder_surface.stl')
        elif dump_f_name == 'ball':
            objcm = el.loadObj('ball_surface.stl')
        elif dump_f_name == 'cube':
            objcm = el.loadObj('cube_surface_2.stl')

        if dump_f_name == 'ball':
            DRAWREC_SIZE = [60, 60]
        else:
            DRAWREC_SIZE = [80, 80]
        error_method = 'ED'
        result_f_name_dict = {
            # 'baseline': f'{dump_f_name}_baseline.pkl',
            # 'DI': f'{dump_f_name}_DI.pkl',
            # 'DI\'': f'{dump_f_name}_DI\'.pkl',
            # 'EI': f'{dump_f_name}_EI.pkl',
            # 'QI': f'{dump_f_name}_QI.pkl',
            # 'RBF': f'{dump_f_name}_RBF.pkl',
            # 'quad': f'{dump_f_name}_quad.pkl',
            # 'bs': f'{dump_f_name}_bs.pkl',
            # 'RBF-G': f'{dump_f_name}_RBF-G.pkl',
            # 'RBF-G-linear': f'{dump_f_name}_RBF-G-linear.pkl',
            # 'bp': f'{dump_f_name}_bp.pkl',
            'SI': f'{dump_f_name}_SI.pkl',
            # 'II': f'{dump_f_name}_II.pkl',
        }
        compare_prj_method(result_f_name_dict, error_method=error_method, savef=False, objcm=objcm, objpcd=objpcd)


    elif run == 8:
        # for dump_f_name in ['ball', 'cube', 'cylinder_cad', 'cylinder_pcd', 'helmet']:
        for dump_f_name in ['cylinder_pcd']:
            objcm = None
            if dump_f_name == 'cylinder_cad':
                objcm = el.loadObj('cylinder_surface.stl')
            if dump_f_name == 'ball':
                objcm = el.loadObj('ball_surface.stl')
            if dump_f_name == 'cube':
                objcm = el.loadObj('cube_surface_2.stl')
            if dump_f_name == 'ball':
                DRAWREC_SIZE = [60, 60]
            else:
                DRAWREC_SIZE = [80, 80]
            if dump_f_name == 'ball':
                DRAWREC_SIZE = [60, 60]
            else:
                DRAWREC_SIZE = [80, 80]
            error_method = 'inc'
            # method_list = ['DI', 'DI\'', 'EI', 'SI', 'II', 'quad', 'RBF', 'RBF-G', 'RBF-G-linear']
            method_list = ['SI']
            for method in method_list:
                result_f_name_dict = {
                    method: f'{dump_f_name}_{method}.pkl',
                    'baseline': f'{dump_f_name}_baseline.pkl',
                }
                compare_prj_method(result_f_name_dict, error_method=error_method, savef=True, objcm=objcm)

    elif run == 3:
        dump_f_name = 'cylinder_cad'
        result_f_name = f'{dump_f_name}_QI.pkl'
        compare_error_method(result_f_name)
        plt.show()

        result_f_name = f'{dump_f_name}_EI.pkl'
        compare_error_method(result_f_name)
        plt.show()

    elif run == 4:
        dump_f_name = 'helmet'
        objcm = None
        if dump_f_name == 'cylinder_cad':
            objcm = el.loadObj('cylinder_surface.stl')
        if dump_f_name == 'ball':
            objcm = el.loadObj('ball_surface.stl')
        if dump_f_name == 'cube':
            objcm = el.loadObj('cube_surface_2.stl')
        if dump_f_name == 'ball':
            DRAWREC_SIZE = [60, 60]
        else:
            DRAWREC_SIZE = [80, 80]
        error_method = 'inc'
        method_list = ['EI', 'quad', 'bs', 'RBF-G', 'bp']
        for method in method_list:
            if method in ['EI', 'RBF-G', 'bp', 'bs']:
                result_f_name_dict = {
                    1: f'{dump_f_name}_s1_{method}.pkl',
                    2: f'{dump_f_name}_s2_{method}.pkl',
                    5: f'{dump_f_name}_s5_{method}.pkl',
                    10: f'{dump_f_name}_s10_{method}.pkl',
                    20: f'{dump_f_name}_s20_{method}.pkl'
                }
            else:
                result_f_name_dict = {
                    1: f'{dump_f_name}_s1_{method}.pkl',
                    2: f'{dump_f_name}_s2_{method}.pkl',
                    5: f'{dump_f_name}_s5_{method}.pkl',
                    10: f'{dump_f_name}_s10_snap_{method}.pkl',
                    20: f'{dump_f_name}_s20_snap_{method}.pkl'
                }
            compare_step(result_f_name_dict, error_method=error_method, label=method, objcm=objcm)

        plt.show()
        # base.run()

    elif run == 5:
        dump_f_name = 'cylinder_cad'
        error_method = 'ED'
        drawpath_name = 'circle'
        compare_start(f'{dump_f_name}_{drawpath_name}_QI.pkl', error_method=error_method)
        compare_start(f'{dump_f_name}_{drawpath_name}_EI.pkl', error_method=error_method)
        compare_start(f'{dump_f_name}_{drawpath_name}_SI.pkl', error_method=error_method)
        compare_start(f'{dump_f_name}_{drawpath_name}_DI.pkl', error_method=error_method)

        # base.run()

    elif run == 6:
        dump_f_name = 'cylinder_cad'
        compare_start_with_gt(f'{dump_f_name}_circle_EI.pkl', f'{dump_f_name}_circle_SI.pkl')
        compare_start_with_gt(f'{dump_f_name}_circle_QI.pkl', f'{dump_f_name}_circle_SI.pkl')
        compare_start_with_gt(f'{dump_f_name}_circle_DI.pkl', f'{dump_f_name}_circle_SI.pkl')
        plt.show()
        # base.run()

    elif run == 7:
        dump_f_name = 'cylinder_cad'
        # item.show_objcm(rgba=(1, 1, 1, 1))
        res_dict = pickle.load(open(f'{dump_f_name}_SI.pkl', 'rb'))
        pos_nrml_list = np.asarray(res_dict['pos_nrml_list'])
        # pu.show_drawpath(pu.flatten_nested_list(pos_nrml_list), color=(0, 0, 1))
        base.pggen.plotAxis(base.render, thickness=1, alpha=1, length=75)

        ps = np.asarray([[p[0], p[1], 0] for p in pu.flatten_nested_list(res_dict['drawpath'])])
        for p in ps:
            base.pggen.plotSphere(base.render, p, rgba=(0, 0, 1, 1))
        # ps = np.asarray([p for p, n in pu.flatten_nested_list(pos_nrml_list)])

        plane_n = np.asarray([0, 0, -1])
        view_p = np.array((np.mean(ps[:, 0]), np.mean(ps[:, 1]), np.mean(ps[:, 2]))) + np.asarray([0, -50, 100])
        view_plane_dist = 50
        pts = ppu.trans_projection(ps, plane_n, view_p, view_plane_dist)

        # base.pggen.plotArrow(base.render, spos=view_p, epos=view_p + view_plane_dist * plane_n, rgba=(1, 0, 1, 1),
        #                      thickness=2)
        # view_p = view_p + np.asarray([- 50, 0, 0])
        # base.pggen.plotArrow(base.render, spos=view_p, epos=view_p + view_plane_dist * plane_n, rgba=(1, 0, 1, 1),
        #                      thickness=2)
        # view_p = view_p + np.asarray([50, -50, 0])
        # base.pggen.plotArrow(base.render, spos=view_p, epos=view_p + view_plane_dist * plane_n, rgba=(1, 0, 1, 1),
        #                      thickness=2)

        ax = plt.gca()
        print(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=1)
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_aspect(1)

        ax.set_xlim(-50, 25)
        ax.set_ylim(-50, 25)

        plt.show()

        base.run()
