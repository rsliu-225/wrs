import os
import random
import math
import basis.robot_math as rm
import numpy as np
import open3d as o3d

import datagenerator.data_utils as utl
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd
from basis.trimesh.creation import icosphere


def show_ico():
    tm = icosphere(subdivisions=3)
    origin = gm.gen_sphere(radius=.00001)
    origin.attach_to(base)
    for i in tm.vertices:
        gm.gen_sphere(i, radius=.01).attach_to(origin)
    icosphere_cm = cm.CollisionModel(tm)
    icosphere_cm.set_rgba([1, 1, 1, .7])

    icosphere_cm.attach_to(base)
    icosphere_cm.show_cdmesh()

    selection_vector = gm.gen_arrow(np.array([0, 0, 0]), np.array([0, 0, 1]), thickness=.1)
    selection_vector.set_rgba(np.array([31 / 255, 191 / 255, 31 / 255, 1]))

    vertices, vertex_normals, faces = icosphere_cm.extract_rotated_vvnf()
    objwm = gm.WireFrameModel(cm.da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
    objwm.attach_to(base)
    icosphere_cm.set_scale([.995, .995, .995])


if __name__ == '__main__':
    cam_pos = np.asarray([0, 0, .5])
    base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])

    show_ico()

    icomats = rm.gen_icorotmats(rotation_interval=math.radians(360 / 60))
    # icos = trm.creation.icosphere(1)
    # icos_cm = cm.CollisionModel(icos)
    # icos_cm.attach_to(base)

    width = .005
    thickness = .0015
    fo = './tst'
    cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

    pseq = utl.poly_inp(pseq=np.asarray([[0, 0, 0], [.018, .02, .02], [.06, .04, 0], [.12, 0, 0]]))
    # pseq = utl.uni_length(pseq, goal_len=1.6)
    pseq = utl.uni_length(pseq, goal_len=.2)
    pseq, rotseq = utl.get_rotseq_by_pseq(pseq)

    objcm = utl.gen_swap(pseq, rotseq, cross_sec)
    objcm.set_rgba((1, 1, 0, 1))
    objcm.attach_to(base)

    # for matlist in icomats:
    #     np.random.shuffle(matlist)
    #     for j, rot in enumerate(matlist):
    #         gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001, rgba=(.7, .7, .7, .7)).attach_to(base)
    #     for j, rot in enumerate(matlist[:10]):
    #         gm.gen_sphere(pos=rot[:, 0] * .1, radius=.001).attach_to(base)
    #         # objcm_tmp = copy.deepcopy(objcm)
    #         # objcm_tmp.set_homomat(rm.homomat_from_posrot(rot=rot))
    #         # objcm_tmp.attach_to(base)
    # base.run()

    '''
    gen data
    '''
    cnt = 0
    obj_id = 0
    rot_center = (0, 0, 0)

    utl.get_objpcd_partial_o3d(objcm, objcm, np.eye(3), rot_center, path=fo,
                               f_name=f'{str(obj_id)}_{str(cnt).zfill(3)}',
                               visible_threshold=np.radians(30), noise_cnt=3,
                               occ_vt_ratio=random.uniform(.1, .5), noise_vt_ratio=random.uniform(.5, 1),
                               add_noise=False, add_occ=False, add_rnd_occ=True, add_noise_pts=False,
                               savemesh=False, savedepthimg=False, savergbimg=False, toggledebug=True)

    # '''
    # show data
    # '''
    # path = './tst/'
    # for f in sorted(os.listdir(path)):
    #     if f[-3:] == 'pcd':
    #         o3dpcd = o3d.io.read_point_cloud(f"{path}/{f}")
    #         gm.gen_pointcloud(o3dpcd.points).attach_to(base)
    # base.run()
    #
    # '''
    # show key point
    # '''
    # objpcd_narry = utl.get_objpcd_full_sample(objcm)
    # gm.gen_pointcloud(objpcd_narry, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # utl.get_kpts_gmm(objpcd_narry, rgba=(1, 0, 0, 1))
    #
    # o3dmesh = o3d.io.read_triangle_mesh(f"tst/0_0_0.ply")
    # o3dpcd = o3d.io.read_point_cloud(f"tst/0_0_0.pcd")
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 0, .5]]).attach_to(base)
    # utl.get_kpts_gmm(o3dpcd.points, rgba=(0, 1, 0, 1))
    #
    # base.run()
    #
    # o3dpcd = o3d.io.read_point_cloud(f"tst/0_0_0.pcd")
    # o3dpcd_occ = utl.add_random_occ(o3dpcd)
    # gm.gen_pointcloud(o3dpcd.points, pntsize=5, rgbas=[[1, 1, 1, .5]]).attach_to(base)
    # gm.gen_pointcloud(o3dpcd_occ.points, pntsize=5, rgbas=[[1, 0, 0, .5]]).attach_to(base)
    # base.run()
