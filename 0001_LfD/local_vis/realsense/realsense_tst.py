import pickle

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

if __name__ == '__main__':
    # ストリーム(Depth/Color)の設定
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーミング開始
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Alignオブジェクト生成
    align_to = rs.stream.color
    align = rs.align(align_to)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intr)
    intr = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}
    pickle.dump(intr, open("./realsense_intr.pkl", "wb"))
    intr = pickle.load(open("./realsense_intr.pkl", "rb"))
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"],
                                                                 intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])

    moveflag = 0
    # とりあえずカメラ一つ分だけでやる（もし複数のカメラなら上の設定をappend）
    try:
        while True:

            # フレーム待ち(Color & Depth)
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # imageをnumpy arrayに
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # depth imageをカラーマップに変換
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

            # 画像表示
            cv2.namedWindow('RealSense_depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense_rgb', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense_depth', depth_colormap)
            cv2.imshow('RealSense_rgb', color_image)
            if moveflag == 0:
                cv2.moveWindow("RealSense_depth", 2800, 400)
                cv2.moveWindow("RealSense_rgb", 1800, 0)
                moveflag = 1

            if cv2.waitKey(1) & 0xff == 27:  # ESCで終了
                cv2.destroyAllWindows()
                break

        ###############get pcd################
        for col in color_image:
            for pix in col:
                pix[0], pix[2] = pix[2], pix[0]
        target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image,
                                                             pinhole_camera_intrinsic)
        # pcd.colors = o3d.utility.Vector3dVector(color_image.color)

        # 回転する
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # 法線計算
        pcd.estimate_normals()
        print(np.asarray(pcd.points))

        ###############get pcd################

        o3d.visualization.draw_geometries([pcd])

    finally:
        # ストリーミング停止
        pipeline.stop()
