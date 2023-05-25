import argparse

import numpy as np
import open3d as o3d
import tensorflow as tf


def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def show_pcds(pcd_np_arrs):
    pcd_list = []
    colors = [[0, 1, 1], [0, 0.2, 1], [0.6, 0.2, 0.9]]
    for idx, pcd_np in enumerate(pcd_np_arrs):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.paint_uniform_color(colors[idx])
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list)  # , width=3840, height=2160


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
               for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i, f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)


class Model:
    def __init__(self, inputs, npts, gt, alpha):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs, npts)
        self.coarse, self.fine = self.create_decoder(self.features)
        self.outputs = self.fine
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    def create_encoder(self, inputs, npts):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
        return features

    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3]) + center
        return coarse, fine


def run(args):
    np.set_printoptions(suppress=True)
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    gt = tf.placeholder(tf.float32, (1, 16384, 3))
    npts = tf.placeholder(tf.int32, (1,))
    model = Model(inputs, npts, gt, tf.constant(1.0))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.allow_soft_placement = False
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    partial = read_pcd(args.input_path)
    partial = resample_pcd(partial, 8192)
    complete = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})[0]
    print("Shape: ", complete.shape, "Type: ", type(complete))
    if args.gap:
        show_pcds([partial, complete + args.gap])
    else:
        show_pcds([partial, complete])

    if args.output_path is None:
        save_pcd(args.output_path, complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default="partial/38_1_linear7_partial.pcd")
    parser.add_argument('-o', '--output_path', type=str, help='test.pcd')
    parser.add_argument('-g', '--gap', type=float, help='DISPLAY PCD: distance between pcds shown in open3D')
    parser.add_argument('-c', '--checkpoint', type=str, default="model-380000")
    args = parser.parse_args()
    run(args)

# python main.py -i partial/38_1_linear7_partial.pcd -o test.pcd -g 0.01
