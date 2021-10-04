import cv2
import drivers.devices.kinect_azure.pykinectazure as pk
import visualization.panda.world as wd
import modeling.geometric_model as gm
import pickle
import basis.robot_math as rm
import os
import shutil
import config_LfD as config
import time
import detection_utils as du
import utils.pcd_utils as pcdu
import numpy as np
import open3d as o3d


class KinectAzura(object):
    def __init__(self,online=True):
        self.root = config.DATA_PATH
        if online:
            self.knt = pk.PyKinectAzure()
            calibration = self.knt.get_calibration()
            depth_param = calibration.depth_camera_calibration.intrinsics.parameters.param
            self.knt.device_get_capture()
            depth_image_handle = self.knt.capture_get_depth_image()
            self.intr = {'width': self.knt.image_get_width_pixels(depth_image_handle),
                         'height': self.knt.image_get_height_pixels(depth_image_handle),
                         'fx': depth_param.fx, 'fy': depth_param.fy,
                         'cx': depth_param.cx, 'cy': depth_param.cy}
            pickle.dump(self.intr, open(f'{config.ROOT}/local_vis/knt_azura/knt_azura_intr.pkl', 'wb'))
        else:
            self.intr = pickle.load(open(f'{config.ROOT}/local_vis/knt_azura/knt_azura_intr.pkl', 'rb'))

    def view(self):
        while True:
            self.knt.device_get_capture()
            color_image_handle = self.knt.capture_get_color_image()
            depth_image_handle = self.knt.capture_get_depth_image()
            if color_image_handle and depth_image_handle:
                depthimg = self.knt.image_convert_to_numpy(depth_image_handle)
                rgbimg_trans = self.knt.transform_color_to_depth(color_image_handle, depth_image_handle)
                self.knt.image_release(color_image_handle)
                self.knt.image_release(depth_image_handle)
                rgbimg_trans = cv2.cvtColor(rgbimg_trans, cv2.COLOR_BGRA2BGR)

                cv2.imshow("depth", du.scale_depth_img(depthimg))
                cv2.imshow("rgb", rgbimg_trans)
            self.knt.capture_release()
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        print('--------------------done--------------------')

    def dump_frameseq(self, folder_name, time_interval=1.0, rel_path=os.path.join('raw_img/k4a/seq/')):
        dump_path = os.path.join(self.root, rel_path, folder_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        else:
            shutil.rmtree(dump_path)
            os.makedirs(dump_path)
        print('--------------------start--------------------')
        i = 0
        while True:
            self.knt.device_get_capture()
            color_image_handle = self.knt.capture_get_color_image()
            depth_image_handle = self.knt.capture_get_depth_image()
            if color_image_handle and depth_image_handle:
                depthimg = self.knt.image_convert_to_numpy(depth_image_handle)
                rgbimg_trans = self.knt.transform_color_to_depth(color_image_handle, depth_image_handle)
                pcd = self.knt.transform_depth_image_to_point_cloud(depth_image_handle)
                self.knt.image_release(color_image_handle)
                self.knt.image_release(depth_image_handle)
                rgbimg_trans = cv2.cvtColor(rgbimg_trans, cv2.COLOR_BGRA2BGR)

                pickle.dump([depthimg, rgbimg_trans, pcd],
                            open(os.path.join(dump_path, f'{str(i).zfill(4)}.pkl'), 'wb'))
                print('Captured!', os.path.join(dump_path, f'{str(i).zfill(4)}.pkl'))
                i += 1
                cv2.imshow("depth", du.scale_depth_img(depthimg))
                cv2.imshow("rgb", rgbimg_trans)
            self.knt.capture_release()
            key = cv2.waitKey(1)
            if time_interval > 0:
                time.sleep(time_interval)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        print('--------------------done--------------------')

    def show_frameseq(self, depthimg_list, rgbimg_list):
        print(f'num of frames: {len(depthimg_list)}')
        for img_id in range(len(depthimg_list)):
            depthimg = depthimg_list[img_id]
            rgbimg = rgbimg_list[img_id]
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
            imgs = np.hstack((rgbimg, depth_colormap))
            cv2.imshow('', imgs)
            cv2.waitKey(0)

    def rgbd2pcd(self, depthimg, rgbimg, toggledebug=False):
        pinhole_camera_intrinsic = \
            o3d.camera.PinholeCameraIntrinsic(self.intr["width"], self.intr["height"],
                                              self.intr["fx"], self.intr["fy"], self.intr["cx"], self.intr["cy"])
        img_depth = o3d.geometry.Image(depthimg)
        img_color = o3d.geometry.Image(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def show_rgbdseq(self, depthimg_list, rgbimg_list,win_name=''):
        pointcloud = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window(win_name, width=1280, height=720)

        geom_added = False
        print(f"num of frames: {len(depthimg_list)}")
        i = 0
        while True:
            if i >= len(depthimg_list):
                i = 0
            pcd = self.rgbd2pcd(depthimg_list[i], rgbimg_list[i])
            pointcloud.points = pcd.points
            pointcloud.colors = pcd.colors

            if geom_added == False:
                vis.add_geometry(pointcloud)
                geom_added = True

            vis.update_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()

            # cv2.imshow('rgb', rgbimg_list[i])
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            i += 1
        # cv2.destroyAllWindows()
        vis.destroy_window()
        del vis


if __name__ == '__main__':
    folder_name = 'glue'
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    knt = KinectAzura()
    # knt.view()
    knt.dump_frameseq(folder_name, time_interval=0)
    # depthimg_list, rgbimg_list, pcd_list = \
    #     du.load_frame_seq(folder_name, root_path=os.path.join(config.DATA_PATH, 'raw_img/k4a/seq/'))
    # knt.show_frameseq(depthimg_list, rgbimg_list)
    # knt.show_rgbdseq(depthimg_list, rgbimg_list)
    # pcdu.show_pcdseq(pcd_list)
    # base.run()
