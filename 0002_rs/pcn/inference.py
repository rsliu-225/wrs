import logging
import importlib
import numpy as np
import h5py
import subprocess
import os

from numpy.lib.index_tricks import AxisConcatenator
import munch
import torch
import open3d as o3d

import warnings

import config
import pcn.models.pcn as model_module

warnings.filterwarnings("ignore")
COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14]]) / 255


def inference_sgl(input_narry, model_name='pcn', load_model='pcn_emd_prim_mv/best_cd_p_network.pth', toggledebug=False):
    load_model = os.path.join(config.ROOT, model_name, 'trained_model', load_model)
    device = torch.device('cpu')
    args = munch.munchify({'num_points': 2048, 'loss': 'cd', 'eval_emd': False})
    input_narry = torch.from_numpy(input_narry)
    net = torch.nn.DataParallel(model_module.Model(args))
    net.module.load_state_dict(torch.load(load_model, map_location=torch.device('cpu'))['net_state_dict'])
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

    f_name = 'test'

    model_name = 'pcn'
    load_model = 'pcn_emd_prim_mv_2/best_cd_p_network.pth'

    # o3dpcd_1 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/020.pcd')
    # o3dpcd_2 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/001.pcd')
    # o3dpcd_1 = o3d.io.read_point_cloud(config.ROOT + '/recons_data/nbc/plate_a_cubic/000.pcd')
    # path = os.path.join(config.ROOT, 'recons_data/nbc/plate_a_cubic')
    path = 'D:/liu/MVP_Benchmark/completion/data_real/'
    for f in os.listdir(path):
        o3dpcd = o3d.io.read_point_cloud(os.path.join(path, f))
        if len(np.asarray(o3dpcd.points)) > 2048:
            o3dpcd = o3dpcd.uniform_down_sample(int(len(np.asarray(o3dpcd.points)) / 2048))
        print(np.asarray(o3dpcd.points).shape)
        result = inference_sgl(np.asarray(o3dpcd.points), model_name, load_model)
        o3dpcd_o = nparray2o3dpcd(result)
        o3dpcd.paint_uniform_color(COLOR[0])
        o3dpcd_o.paint_uniform_color(COLOR[2])
        o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o])
