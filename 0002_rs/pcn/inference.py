import logging
import os
import warnings
import config as config

import munch
import numpy as np
import open3d as o3d
import torch

import pcn.models.pcn as model_module

warnings.filterwarnings("ignore")
COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def inference_sgl(input_narry, model_name='pcn', load_model='pcn_emd_rlen/best_cd_p_network.pth', toggledebug=False):
    load_model = os.path.join(config.ROOT, model_name, 'trained_model', load_model)
    device = torch.device('cpu')
    args = munch.munchify({'num_points': 2048, 'loss': 'cd', 'eval_emd': False})
    input_narry = torch.from_numpy(input_narry)
    net = torch.nn.DataParallel(model_module.Model(args))
    net.module.load_state_dict(torch.load(load_model, map_location=device)['net_state_dict'])
    net = net.module.to(device)
    logging.info("%s's previous weights loaded." % model_name)
    net.eval()

    logging.info('Testing...')
    with torch.no_grad():
        inputs_cpu = input_narry
        inputs = inputs_cpu.float().cpu()
        inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.transpose(2, 1).contiguous()
        # print(inputs.shape)
        result_dict = net(inputs)
        output = result_dict['result'].cpu().numpy()
    if toggledebug:
        o3dpcd_i = nparray2o3dpcd(input_narry)
        o3dpcd_o = nparray2o3dpcd(output[0])
        o3dpcd_i.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd_i, o3dpcd_o])

    return output[0]


def nparray2o3dpcd(nx3nparray_pnts, nx3nparray_nrmls=None, estimate_normals=False):
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(nx3nparray_pnts[:, :3])
    if nx3nparray_nrmls is not None:
        o3dpcd.normals = o3d.utility.Vector3dVector(nx3nparray_nrmls[:, :3])
    elif estimate_normals:
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    return o3dpcd


if __name__ == "__main__":
    import config

    f_name = 'test'

    model_name = 'pcn'
    load_model = 'pcn_emd_rlen/best_emd_network.pth'

    # o3dpcd_1 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/020.pcd')
    # o3dpcd_2 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/001.pcd')
    # o3dpcd_1 = o3d.io.read_point_cloud(config.ROOT + '/recons_data/nbc/plate_a_cubic/000.pcd')
    # path = os.path.join(config.ROOT, 'recons_data/seq/plate_a_quadratic')
    # path = 'D:/liu/MVP_Benchmark/completion/data_real/'
    # path = f'{config.ROOT}/recons_data/nbc_pcn/plate_a_cubic'
    path = f'{config.ROOT}/recons_data/nbc/extrude_1'

    for i, f in enumerate(os.listdir(path)):
        o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f))
        o3dpcd = o3dpcd.voxel_down_sample(voxel_size=.001)

        o3dpcd, ind = o3dpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
        outlier_cloud = o3dpcd.select_by_index(ind, invert=True)
        o3dpcd = o3dpcd.select_by_index(ind)
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        o3dpcd.paint_uniform_color([0.8, 0.8, 0.8])

        o3d.visualization.draw_geometries([o3dpcd, outlier_cloud])

        # o3dpcd_2 = o3d.io.read_point_cloud(os.path.join(path, os.listdir(path)[i + 1]))
        # print(np.asarray(o3dpcd.points).shape)
        # o3dpcd = o3dpcd + o3dpcd_2
        # print(np.asarray(o3dpcd.points).shape)
        # if len(np.asarray(o3dpcd.points)) > 2048:
        #     o3dpcd = o3dpcd.uniform_down_sample(int(len(np.asarray(o3dpcd.points)) / 2048))
        # print(np.asarray(o3dpcd.points).shape)

        result = inference_sgl(np.asarray(o3dpcd.points), model_name, load_model)
        o3dpcd_o = nparray2o3dpcd(result)
        o3dpcd.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd])
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o])
