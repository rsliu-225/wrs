import math
import pickle
import numpy as np
import basis.trimesh as trimesh

import basis.trimesh.sample as sample
import utils.pcd_utils as pcdu
import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.geometric_model as gm
import multiprocessing

TOGGLEDEBUG = False


class Item(object):
    def __init__(self, *args, **kwargs):
        self.__objcm = None
        if "reconstruct" in list(kwargs.keys()):
            self.__reconstruct = kwargs["reconstruct"]
        else:
            self.__reconstruct = False

        if "objmat4" in list(kwargs.keys()):
            self.__objmat4 = kwargs["objmat4"]
        else:
            self.__objmat4 = np.eye(4)

        if "objcm" in list(kwargs.keys()):
            self.__objcm = kwargs["objcm"]
            # if kwargs["filter_dir"] is not None:
            #     self.__filter_dir = kwargs["filter_dir"]
            #     pcdu.show_pcd(self.__objcm.trimesh.vertices)
            #     mask_range = self.__filter_by_range(self.__objcm,
            #                                         x_range=(-50, 50), y_range=(50, 500), z_range=(-50, 50))
            #     mask_ndir = self.__filer_by_dir(self.__objcm, self.__filter_dir)
            #     self.__reconstruct_by_mask(mask_range * mask_ndir)
            self.__pcd_std = pcdu.get_objpcd(kwargs["objcm"], objmat4=self.__objmat4)
            self.__w, self.__h = pcdu.get_pcd_w_h(self.__pcd_std)
        if "pcd" in list(kwargs.keys()):
            self.__pcd = kwargs["pcd"]
            self.__nrmls = pcdu.get_nrmls(kwargs["pcd"], camera_location=(800, -200, 1800), toggledebug=TOGGLEDEBUG)
            if self.__reconstruct:
                self.__objcm = pcdu.reconstruct_surface(self.__pcd, radii=[5])
        else:
            self.__pcd, self.__nrmls = pcdu.get_objpcd_withnrmls(self.__objcm, sample_num=kwargs["sample_num"],
                                                                 objmat4=self.__objmat4, toggledebug=TOGGLEDEBUG)
        # filtered_pcd = []
        # filtered_nrmls = []
        # dirc_v = np.asarray((0, 0, 1))
        # dirc_h = np.asarray((-1, -1, 0))
        # for i, n in enumerate(self.__nrmls):
        #     if n.dot(dirc_v) / (np.linalg.norm(n) * np.linalg.norm(dirc_v)) < 0.1 and \
        #             n.dot(dirc_h) / (np.linalg.norm(n) * np.linalg.norm(dirc_h)) < 0.1:
        #         filtered_pcd.append(self.__pcd[i])
        #         filtered_nrmls.append(self.__nrmls[i])
        # self.__pcd, self.__nrmls = np.asarray(filtered_pcd), np.asarray(filtered_nrmls)

        self.__pcdcenter = np.array((np.mean(self.__pcd[:, 0]),
                                     np.mean(self.__pcd[:, 1]),
                                     np.mean(self.__pcd[:, 2])))

        if "drawcenter" in list(kwargs.keys()):
            self.__drawcenter = kwargs["drawcenter"]
        else:
            self.__drawcenter = self.__pcdcenter
        # self.__nrmls = [-n for n in self.__nrmls]

    @property
    def objcm(self):
        return self.__objcm

    @property
    def pcd(self):
        return self.__pcd

    @property
    def nrmls(self):
        return self.__nrmls

    @property
    def pcd_std(self):
        return self.__pcd_std

    @property
    def objmat4(self):
        return self.__objmat4

    @property
    def pcdcenter(self):
        return self.__pcdcenter

    @property
    def drawcenter(self):
        return self.__drawcenter

    def set_drawcenter(self, posdiff):
        self.__drawcenter = self.__drawcenter + np.asarray(posdiff)

    def set_objmat4(self, objmat4):
        self.__objmat4 = objmat4
        self.objcm.sethomomat(objmat4)

    def reverse_nrmls(self):
        self.__nrmls = [-n for n in self.__nrmls]

    def gen_colps(self, radius=.03, max_smp=120, show=False):
        nsample = int(math.ceil(self.objcm.objtrm.area / (radius ** 2 / 3.0)))
        if nsample > max_smp:
            nsample = max_smp
        samples = self.objcm.sample_surface(nsample=nsample, toggle_option=None)
        samples = pcdu.trans_pcd(samples, self.objmat4)
        if show:
            for p in samples:
                gm.gen_sphere(pos=p, rgba=(1, 1, 0, .2), radius=radius)
        return samples

    def gen_colps_top(self, show=False):
        col_ps = []
        ps = self.pcd
        x_range = (min([x[0] for x in ps]), max([x[0] for x in ps]))
        y_range = (min([x[1] for x in ps]), max([x[1] for x in ps]))
        step = 10
        for i in range(int(x_range[0]), int(x_range[1]) + 1, step):
            for j in range(int(y_range[0]), int(y_range[1]) + 1, step):
                ps_temp = np.asarray([p for p in ps if i < p[0] < i + step and j < p[1] < j + step])
                if len(ps_temp) != 0:
                    p = ps_temp[ps_temp[:, 2] == max([x[2] for x in ps_temp])][0]
                    col_ps.append(p)
        col_ps = np.asarray(col_ps)
        if show:
            for p in col_ps:
                gm.gen_sphere(pos=p, rgba=(1, 0, 0, .2), radius=.01)
        return col_ps

    def __filer_by_dir(self, objcm, dir):
        # vs = objcm.trimesh.vertices
        # faces = objcm.trimesh.faces
        nrmls = objcm.objtrm.vertex_normals
        cos = np.dot(nrmls, dir) / (np.linalg.norm(nrmls) * np.linalg.norm(dir))
        mask = cos > 0
        return mask

    def __filter_by_range(self, objcm, x_range=None, y_range=None, z_range=None):
        vs = objcm.trimesh.vertices
        center = pcdu.get_pcd_center(vs)
        mask = [True] * len(vs)
        if x_range is not None:
            mask *= (vs[:, 0] < center[0] + x_range[1]) * (vs[:, 0] > center[0] + x_range[0])
        if y_range is not None:
            mask *= (vs[:, 1] < center[1] + y_range[1]) * (vs[:, 1] > center[1] + y_range[0])
        if z_range is not None:
            mask *= (vs[:, 2] < center[2] + z_range[1]) * (vs[:, 2] > center[2] + z_range[0])
        return mask

    def fff(self, v_ids, fs, start_idx, rl):
        for idx, f in enumerate(fs):
            if list(set(v_ids) & set(f)):
                rl.append(start_idx + idx)

    def __filer_fs_mp(self, faces, v_ids, slices_num=100):
        step = int(len(faces) / slices_num)
        ss = 0

        manager = multiprocessing.Manager()
        result_list = manager.list()
        processes = []

        for i in range(slices_num):
            se = ss + step
            if se > len(faces):
                se = len(faces)

            sub_faces = faces[ss:se]
            p = multiprocessing.Process(
                target=self.fff, args=(v_ids, sub_faces, ss, result_list)
            )
            processes.append(p)
            p.start()
            ss += step

        for p in processes:
            p.join()

        face_mask = []
        for i in range(0, len(faces)):
            if i in result_list:
                face_mask.append(False)
            else:
                face_mask.append(True)
        return face_mask

    def __filer_faces(self, faces, v_ids):
        face_mask = []
        for i, f in enumerate(faces):
            if list(set(v_ids) & set(f)):
                face_mask.append(False)
            else:
                face_mask.append(True)

    def __reconstruct_by_mask(self, mask):
        vs = self.__objcm.trimesh.vertices
        faces = self.__objcm.trimesh.faces
        nrmls = self.__objcm.trimesh.vertex_normals
        vs = vs[mask]
        nrmls = nrmls[mask]
        v_ids = np.where(np.asarray(mask))[0]
        print(len(v_ids))
        print(len(faces))
        face_mask = self.__filer_fs_mp(faces, v_ids)
        faces = faces[face_mask]
        self.__objcm = cm.CollisionModel(initor=trimesh.Trimesh(vertices=vs, faces=faces, vertex_normals=nrmls))
        pickle.dump([vs, nrmls, faces], open('skull.pkl', "wb"))
        print(len(vs))
        pcdu.show_pcd(vs)
        base.run()
        # self.__objcm = pcdu.reconstruct_surface(vs, radii=[5])

    def show_objcm(self, rgba=(1, 1, 1, 1), show_localframe=False):
        # import copy
        # objmat4 = copy.deepcopy(self.objmat4)
        # objmat4[:3, :3] = np.eye(3)
        # self.__objcm.sethomomat(objmat4)
        self.__objcm.sethomomat(self.objmat4)
        self.__objcm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        if show_localframe:
            self.__objcm.showlocalframe()
        self.__objcm.attach_to(base)

    def show_objpcd(self, rgba=(1, 1, 1, 1)):
        pcdu.show_pcd(self.__pcd, rgba=rgba)
