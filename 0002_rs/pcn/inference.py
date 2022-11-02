import logging
import os
import sys
import importlib
import argparse
import numpy as np
import h5py
import subprocess

from numpy.lib.index_tricks import AxisConcatenator
import munch
import yaml
import open3d as o3d

import warnings
import torch

warnings.filterwarnings("ignore")


def inference_sgl(input_data):
    input_data = np.asarray(input_data)
    input_data = torch.from_numpy(input_data)
    logging.info(str(args))
    # dataloader = torch.utils.data.DataLoader(input_data, batch_size=args.batch_size,
    # shuffle=False, num_workers=int(args.workers))
    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    logging.info('Testing...')
    with torch.no_grad():
        inputs_cpu = input_data
        # torch.Size([64, 2048, 3])
        # inputs_cpu = input_data

        inputs = inputs_cpu.float().cuda()
        inputs = torch.unsqueeze(inputs, 0)
        # inputs = inputs.transpose(2, 1).contiguous()
        inputs = inputs.transpose(2, 1).contiguous()
        inputs = inputs.repeat(64, 1, 1)
        # torch.Size([64, 3, 2048])
        print(inputs.shape)
        result_dict = net(inputs, prefix="test")
        output = result_dict['result'].cpu().numpy()
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
    GOAL_DATA_PATH = 'D:/liu/MVP_Benchmark/completion/data_2048_flat/'
    COLOR = np.asarray([[31, 119, 180], [44, 160, 44], [214, 39, 40]]) / 255
    f_name = 'test'

    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file',
                        default="D:/liu/MVP_Benchmark/completion/cfgs/pcn_inf.yaml")
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    f = h5py.File(f'{GOAL_DATA_PATH}/{f_name}.h5', 'r')
    print(f.name, f.keys())
    # for i in range(len(f['complete_pcds'])):
    #     if f['labels'][i] == 1:
    #         o3dpcd_i = nparray2o3dpcd(np.asarray(f['incomplete_pcds'][i]))
    #         o3dpcd_gt = nparray2o3dpcd(np.asarray(f['complete_pcds'][i]))
    #         o3dpcd_i.paint_uniform_color(COLOR[0])
    #         o3dpcd_gt.paint_uniform_color(COLOR[1])
    #         result = inference_sgl(np.asarray(f['incomplete_pcds'][i]))
    #         print(result)
    #         o3dpcd_o = nparray2o3dpcd(result)
    #         o3dpcd_o.paint_uniform_color(COLOR[2])
    #         o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_i])
    #         o3d.visualization.draw_geometries([o3dpcd_o, o3dpcd_gt])

    o3dpcd_1 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/020.pcd')
    o3dpcd_2 = o3d.io.read_point_cloud('D:/liu/MVP_Benchmark/completion/data_real/021.pcd')
    o3dpcd = o3dpcd_1+o3dpcd_2
    o3dpcd = o3dpcd.uniform_down_sample(int(len(np.asarray(o3dpcd.points)) / 2048))
    print(np.asarray(o3dpcd.points).shape)
    result = inference_sgl(np.asarray(o3dpcd.points))
    o3dpcd_o = nparray2o3dpcd(result)
    o3dpcd.paint_uniform_color(COLOR[0])
    o3dpcd_o.paint_uniform_color(COLOR[2])
    o3d.visualization.draw_geometries([o3dpcd, o3dpcd_o])
