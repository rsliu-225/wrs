import copy
import os
import pickle
import time

import cv2
import numpy as np

import config
import utils.phoxi as phoxi
import robot_con.xarm_shuidi.xarm_shuidi_x as xarmx
import drivers.nokov.nokov_client as nc


def get_jnt(f_name, img_num, armname='rgt', dump=False):
    rbtx = xarmx.XArmShuidiX()
    jnts = rbtx.arm_get_jnt_values()
    i = 0
    while i < img_num:
        if dump:
            if img_num == 1:
                pickle.dump(jnts, open(config.ROOT + '/img/jnts/' + '_'.join([f_name, armname]) + '.pkl', 'wb'))
            else:
                pickle.dump(jnts, open(config.ROOT + '/img/jnts/' + '_'.join([f_name, str(i), armname]) + '.pkl', 'wb'))
        i += 1
        print(jnts)


def get_img(f_name, img_num, path=''):
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    i = 0
    while i < img_num:
        if img_num == 1:
            grayimg, depthnparray_float32, pcd = phxi.dumpalldata(f_name='img/' + path + f_name + '.pkl')
        else:
            grayimg, depthnparray_float32, pcd = \
                phxi.dumpalldata(f_name='img/' + path + '_'.join([f_name, str(i)]) + '.pkl')
        cv2.imshow('grayimg', grayimg)
        cv2.waitKey(0)
        i += 1


def get_img_rbt(img_num, path='', jnt_range=(-np.pi, np.pi)):
    if not os.path.exists(os.path.join(config.ROOT, 'img', path)):
        os.mkdir(os.path.join(config.ROOT, 'img', path))
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    rbtx = xarmx.XArmShuidiX(ip='10.2.0.201')
    rbtx.arm_jaw_to(0)
    i = 0

    for a in np.linspace(jnt_range[0], jnt_range[1], img_num):
        jnts = rbtx.arm_get_jnt_values()
        jnts_new = copy.deepcopy(jnts)
        jnts_new[6] = a
        rbtx.arm_move_jspace_path([jnts, jnts_new])
        grayimg, _, _ = phxi.dumpalldata(f_name='img/' + path + str(i).zfill(3) + '.pkl')
        i += 1
        # cv2.imshow('grayimg', grayimg)
        # cv2.waitKey(0)


def get_img_rbt_opti(img_num, path='', jnt_range=(-np.pi, np.pi)):
    if not os.path.exists(os.path.join(config.ROOT, 'img', path)):
        os.mkdir(os.path.join(config.ROOT, 'img', path))
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    rbtx = xarmx.XArmShuidiX(ip='10.2.0.201')
    rbtx.arm_jaw_to(0)
    i = 0
    server = nc.NokovClient(server_ip='10.1.1.198')

    for a in np.linspace(jnt_range[0], jnt_range[1], img_num):
        jnts = rbtx.arm_get_jnt_values()
        jnts_new = copy.deepcopy(jnts)
        jnts_new[6] = a
        rbtx.arm_move_jspace_path([jnts, jnts_new])
        time.sleep(1)
        opti_data = server.get_rigidbody_set_frame()
        print(opti_data.rigidbody_set_dict)
        print(opti_data.rigidbody_set_dict[1].qx)

        grayimg, _, _ = phxi.dumpalldata(f_name=os.path.join('img', path, f'{str(i).zfill(3)}.pkl'))
        pickle.dump(opti_data, open(os.path.join(config.ROOT, 'img', path, f'{str(i).zfill(3)}_opti.pkl'), 'wb'))

        i += 1
        # cv2.imshow('grayimg', grayimg)
        # cv2.waitKey(0)


if __name__ == '__main__':
    folder_name = 'plate_a_quadratic'
    img_num = 30
    # get_img_rbt(img_num, path=f'phoxi/seq/{folder_name}/')
    get_img_rbt_opti(img_num, path=f'phoxi/opti/{folder_name}/', jnt_range=(-np.pi, 0))
