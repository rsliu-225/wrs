import os
import pickle
import time

import cv2
import numpy as np
import pyrealsense2 as rs2

import config


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

    def start(self):
        profile = self.__pipeline.start(self.__config)
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

                depth_img = np.asanyarray(aligned_depth_frame.get_data())
                color_img = np.asanyarray(color_frame.get_data())

                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_img_3d = np.dstack(
                    (depth_img, depth_img, depth_img))  # depth image is 1 channel, color is 3 channels
                bg_removed = np.where((depth_img_3d > self.__clipping_distance) | (depth_img_3d <= 0), grey_color,
                                      color_img)

                # Render images
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                imgs = np.hstack((color_img, depth_colormap))
                cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Align Example', imgs)
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
        finally:
            self.stop()

    def load_frame_seq(self, f_name, root_path=os.path.join(config.ROOT, "img/realsense/seq/")):
        return pickle.load(open(os.path.join(root_path, f_name), 'rb'))

    def load_frame(self, f_name, root_path=os.path.join(config.ROOT, "img/realsense/sgl/")):
        return pickle.load(open(os.path.join(root_path, f_name), 'rb'))

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

    def dump_frame(self, f_name, root_path=os.path.join(config.ROOT, "img/realsense/sgl/")):
        depth_img, rgb_img = self.get_frame()
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        pickle.dump([depth_img, rgb_img], open(os.path.join(root_path, f_name), 'wb'))

    def get_frame_seq(self, time_limit=5, time_interval=1.0):
        self.start()
        time.sleep(3)

        timeout = time.time() + time_limit
        depth_img_list = []
        rgb_img_list = []

        print("--------------------start--------------------")
        while time.time() < timeout:
            frames = self.__pipeline.wait_for_frames()
            align_to = rs2.stream.color
            align = rs2.align(align_to)
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()

            if not depth_frame or not rgb_frame:
                continue

            # Convert images to numpy arrays
            depth_img = np.asanyarray(depth_frame.get_data())
            rgb_img = np.asanyarray(rgb_frame.get_data())

            depth_img_list.append(depth_img)
            rgb_img_list.append(rgb_img)
            print("Captured!", len(depth_img_list))

            if time_interval > 0:
                time.sleep(time_interval)
        print("--------------------done--------------------")
        frames_list = [depth_img_list, rgb_img_list]
        self.stop()

        return frames_list

    def dump_bag(self, f_name, time_limit=30, root_path=os.path.join(config.ROOT, "img/realsense/bag/")):
        self.__config.enable_record_to_file(os.path.join(root_path, f_name))
        self.start()
        align_to = rs2.stream.color
        align = rs2.align(align_to)
        timeout = time.time() + time_limit

        print("--------------------start--------------------")
        while time.time() < timeout:
            frames = self.__pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            rgb_frame = aligned_frames.get_color_frame()
            if not depth_frame or not rgb_frame:
                continue

            depth_img = np.asanyarray(depth_frame.get_data())
            rgb_img = np.asanyarray(rgb_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

            # Show images
            imgs = np.hstack((rgb_img, depth_colormap))
            cv2.namedWindow(f_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f_name, imgs)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                self.stop()
                break

        print("--------------------done--------------------")
        self.stop()

    def dump_frameseq(self, f_name, root_path=os.path.join(config.ROOT, "img/realsense/seq/"), time_total=5,
                      time_interval=1.0):
        frames_list = self.get_frame_seq(time_limit=time_total, time_interval=time_interval)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        pickle.dump(frames_list, open(os.path.join(root_path, f_name), 'wb'))

    def convert_bag2frameseq(self, f_name, sleep=0.1, root_path=os.path.join(config.ROOT, "img/realsense/bag/")):
        self.__config.enable_device_from_file(os.path.join(root_path, f_name), repeat_playback=False)
        self.start()
        depth_img_list = []
        rgb_img_list = []
        while self.__flag:
            try:
                frames = self.__pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                rgb_frame = frames.get_color_frame()
                depth_img = np.asanyarray(depth_frame.get_data())
                rgb_img = np.asanyarray(rgb_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
                imgs = np.hstack((rgb_img, depth_colormap))
                cv2.namedWindow(f_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(f_name, imgs)
                depth_img_list.append(depth_img)
                rgb_img_list.append(rgb_img)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    self.stop()
                    break
                time.sleep(sleep)
            except RuntimeError:
                self.stop()

        pickle.dump([depth_img_list, rgb_img_list],
                    open(os.path.join(config.ROOT, "img/realsense/seq/", f_name.replace(".bag", ".pkl")), 'wb'))

    def show_frameseq(self, f_name, root_path=os.path.join(config.ROOT, "img/realsense/seq/")):
        """

        :param f_name:
        :param inx: 0 - depth image; 1 - rgb image
        :param root_path:
        :return:
        """
        data = self.load_frame_seq(f_name, root_path=root_path)
        print(f"num of frames: {len(data[0])}")
        for img_id in range(len(data[0])):
            depth_img = data[0][img_id]
            color_img = data[1][img_id]
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
            imgs = np.hstack((color_img, depth_colormap))
            cv2.namedWindow(f_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(f_name, imgs)
            cv2.waitKey(0)


if __name__ == '__main__':
    realsense = RealSense()
    # realsense.dump_frameseq("inhand_tst3.pkl", time_interval=.2, time_total=2)
    # realsense.show_frameseq("inhand_tst3.pkl")
    # realsense.dump_bag(f_name="inhand_tst.bag", time_limit=5)
    # realsense.convert_bag2frameseq(f_name="tst.bag", sleep=1)
    # realsense.show_frameseq("tst.pkl")
    realsense.view()