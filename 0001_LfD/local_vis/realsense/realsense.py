import os
import pickle
import time

import cv2
import numpy as np
import pyrealsense2 as rs2

import config_LfD as config
import shutil
import utils.vision_utils as vu
import visualization.panda.world as wd
import modeling.geometric_model as gm
import open3d as o3d


class RealSense(object):
    def __init__(self, color_frame_size=(640, 480), depth_frame_size=(640, 480),
                 color_frame_format=rs2.format.bgr8, depth_frame_format=rs2.format.z16,
                 color_frame_framerate=30, depth_frame_framerate=30):

        self.__pipeline = rs2.pipeline()
        self.__config = rs2.config()

        self.__config.enable_stream(rs2.stream.color, color_frame_size[0], color_frame_size[1],
                                    color_frame_format, color_frame_framerate)
        self.__config.enable_stream(rs2.stream.depth, depth_frame_size[0], depth_frame_size[1],
                                    depth_frame_format, depth_frame_framerate)

        self.__flag = False
        self.intr = pickle.load(open("./realsense_intr.pkl", "rb"))

    def start(self):
        profile = self.__pipeline.start(self.__config)
        intr = profile.get_stream(rs2.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                "ppx": intr.ppx, "ppy": intr.ppy}
        pickle.dump(intr, open("./realsense_intr.pkl", "wb"))
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        clipping_distance_in_meters = 1  # 1 meter
        self.__clipping_distance = clipping_distance_in_meters / depth_scale
        self.__flag = True
        if not self.__flag:
            raise Exception("Frame pipeline is not started, please call \"start()\" first")

    def stop(self):
        self.__pipeline.stop()
        self.__flag = False

    def refresh(self):
        self.stop()
        self.start()

    def view(self):
        self.start()
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        try:
            while True:
                # Get frameset of color and depth
                frames = self.__pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depthimg = np.asanyarray(aligned_depth_frame.get_data())
                rgbimg = np.asanyarray(color_frame.get_data())

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_img_3d = np.dstack(
                    (depthimg, depthimg, depthimg))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_img_3d > self.__clipping_distance) | (depth_img_3d <= 0), grey_color,
                                      rgbimg)

                # Render images
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
                imgs = np.hstack((rgbimg, depth_colormap))
                cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Align Example', imgs)
                cv2.imshow('Bg removed Example', bg_removed)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.stop()

    def load_frame_seq(self, folder_name, root_path=os.path.join(config.ROOT, "res/rs/seq/")):
        depthimg_list = []
        rgbimg_list = []
        for f in sorted(os.listdir(os.path.join(root_path, folder_name))):
            tmp = pickle.load(open(os.path.join(root_path, folder_name, f), 'rb'))
            if tmp[0].shape[-1] == 3:
                depthimg_list.append(tmp[1])
                rgbimg_list.append(tmp[0])
            else:
                depthimg_list.append(tmp[0])
                rgbimg_list.append(tmp[1])
        return [depthimg_list, rgbimg_list]

    def load_frame(self, f_name, root_path=os.path.join(config.ROOT, "/res/rs/sgl/")):
        return pickle.load(open(os.path.join(root_path, f_name), 'rb'))

    def depth2pcd(self, depthimg, toggledebug=False):
        pinhole_camera_intrinsic = \
            o3d.camera.PinholeCameraIntrinsic(self.intr["width"], self.intr["height"],
                                              self.intr["fx"], self.intr["fy"], self.intr["ppx"], self.intr["ppy"])
        depthimg = o3d.geometry.Image(depthimg)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic)
        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
            print(np.asarray(pcd.points))
        return np.asarray(pcd.points)

    def rgbd2pcd(self, depthimg, rgbimg, toggledebug=False):
        pinhole_camera_intrinsic = \
            o3d.camera.PinholeCameraIntrinsic(self.intr["width"], self.intr["height"],
                                              self.intr["fx"], self.intr["fy"], self.intr["ppx"], self.intr["ppy"])
        img_depth = o3d.geometry.Image(depthimg)
        img_color = o3d.geometry.Image(cv2.cvtColor(rgbimg, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if toggledebug:
            o3d.visualization.draw_geometries([pcd])
            print(np.asarray(pcd.points))
        return pcd

    def get_depth_colormap(self):
        self.start()
        frames = self.__pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_img = np.asanyarray(depth_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        self.stop()

        return depth_colormap

    def get_frame(self):
        self.start()
        frames = self.__pipeline.wait_for_frames()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.

        align_to = rs2.stream.color
        align = rs2.align(align_to)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        rgb_frame = aligned_frames.get_color_frame()

        depth_img = np.asanyarray(depth_frame.get_data())
        rgb_img = np.asanyarray(rgb_frame.get_data())
        self.stop()

        return depth_img, rgb_img

    def dump_frame(self, f_name, root_path=os.path.join(config.ROOT, "res/rs/sgl/")):
        depth_img, rgb_img = self.get_frame()
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        pickle.dump([depth_img, rgb_img], open(os.path.join(root_path, f_name), 'wb'))

    def dump_bag(self, f_name, root_path=os.path.join(config.ROOT, "res/rs/bag/")):
        if not os.path.exists(os.path.join(root_path)):
            os.makedirs(os.path.join(root_path))
        self.__config.enable_record_to_file(os.path.join(root_path, f_name))
        self.start()
        time.sleep(3)
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        print("--------------------start--------------------")
        while self.__flag:
            frames = self.__pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue
            # Convert images to numpy arrays
            depthimg = np.asanyarray(depth_frame.get_data())
            rgbimg = np.asanyarray(rgb_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
            imgs = np.hstack((rgbimg, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', imgs)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                self.stop()
                break

        print("--------------------done--------------------")
        self.stop()

    def dump_frameseq(self, folder_name, time_interval=1.0, root_path=os.path.join(config.ROOT, "res/rs/seq/")):
        dump_path = os.path.join(root_path, folder_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        else:
            shutil.rmtree(dump_path)
            os.makedirs(dump_path)

        self.start()
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        time.sleep(3)
        print("--------------------start--------------------")
        i = 0
        while self.__flag:
            frames = self.__pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue

            # Convert images to numpy arrays
            depthimg = np.asanyarray(depth_frame.get_data())
            rgbimg = np.asanyarray(rgb_frame.get_data())

            pickle.dump([depthimg, rgbimg], open(os.path.join(dump_path, f"{str(i).zfill(4)}.pkl"), 'wb'))
            print("Captured!", i)
            i += 1

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
            imgs = np.hstack((rgbimg, depth_colormap))
            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', imgs)
            key = cv2.waitKey(1)
            if time_interval > 0:
                time.sleep(time_interval)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                self.stop()
                break

        print("--------------------done--------------------")
        self.stop()

    def convert_bag2frameseq(self, f_name, time_interval=0.1, root_path=os.path.join(config.ROOT, "res/rs/bag/")):
        dump_path = os.path.join("/".join(root_path.split("/")[:-2]), "seq", f_name.split(".bag")[0])

        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        else:
            shutil.rmtree(dump_path)
            os.makedirs(dump_path)
        self.__config.enable_device_from_file(os.path.join(root_path, f_name), repeat_playback=False)
        self.start()
        i = 0
        while self.__flag:
            try:
                frames = self.__pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                rgb_frame = frames.get_color_frame()
                depthimg = np.asanyarray(depth_frame.get_data())
                rgbimg = np.asanyarray(rgb_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)
                imgs = np.hstack((rgbimg, depth_colormap))
                cv2.namedWindow(f_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f_name, imgs)
                pickle.dump([depthimg, rgbimg], open(os.path.join(dump_path, f"{str(i).zfill(4)}.pkl"), 'wb'))
                i += 1
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    self.stop()
                    break
                time.sleep(time_interval)
            except:
                break

    def show_frameseq(self, folder_name, root_path=os.path.join(config.ROOT, "res/rs/seq/")):
        """

        :param folder_name:
        :param inx: 0 - depth image; 1 - rgb image
        :param root_path:
        :return:
        """
        print(root_path)
        data = self.load_frame_seq(folder_name, root_path=root_path)
        print(f"num of frames: {len(data[0])}")
        for img_id in range(len(data[0])):
            depth_img = data[0][img_id]
            color_img = data[1][img_id]
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            imgs = np.hstack((color_img, depth_colormap))
            cv2.namedWindow(folder_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(folder_name, imgs)
            cv2.waitKey(0)

    def show_pcdseq(self, folder_name, root_path=os.path.join(config.ROOT, "res/rs/seq/")):
        def __update(pcldnp, counter, pcd_list, task):
            if counter[0] >= len(pcd_list):
                counter[0] = 0
            if counter[0] < len(pcd_list):
                if pcldnp[0] is not None:
                    pcldnp[0].detach()
                pcd = pcd_list[counter[0]]
                pcldnp[0] = gm.gen_pointcloud(pcd, pntsize=1)
                pcldnp[0].attach_to(base)
                counter[0] += 1
            else:
                counter[0] = 0
            return task.again

        counter = [0]
        pcldnp = [None]
        base = wd.World(cam_pos=[.2, .2, -1.8], lookat_pos=[.2, .2, 0])
        depthimg_list, rgbimg_list = self.load_frame_seq(folder_name, root_path=root_path)
        pcd_list = []
        for i in range(len(depthimg_list)):
            pcd_list.append(self.depth2pcd(depthimg_list[i]))
        print(f"num of frames: {len(pcd_list)}")
        taskMgr.doMethodLater(0.1, __update, "update", extraArgs=[pcldnp, counter, pcd_list], appendTask=True)

        base.run()

    def show_rgbdseq(self, folder_name, root_path=os.path.join(config.ROOT, "res/rs/seq/")):
        pointcloud = o3d.geometry.PointCloud()
        vis = o3d.visualization.Visualizer()
        vis.create_window('PCD', width=1280, height=720)
        depthimg_list, rgbimg_list = self.load_frame_seq(folder_name, root_path=root_path)
        geom_added = False
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

            cv2.imshow('rgb', rgbimg_list[i])
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            i += 1
        cv2.destroyAllWindows()
        vis.destroy_window()
        del vis


if __name__ == '__main__':
    realsense = RealSense()
    # realsense.dump_frameseq("tst", time_interval=0)
    # realsense.show_frameseq("osaka")
    # realsense.show_pcdseq("osaka")
    realsense.show_rgbdseq("osaka")
    # realsense.dump_bag(f_name="tst.bag", time_limit=5)
    # realsense.convert_bag2frameseq(f_name="tst.bag", time_interval=.1)
    # realsense.view()
