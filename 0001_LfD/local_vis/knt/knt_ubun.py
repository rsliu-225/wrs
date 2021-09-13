import drivers.rpc.frtknt.frtknt_client as fc
import visualization.panda.world as wd
import modeling.geometric_model as gm

import cv2
import time
import pickle
import numpy as np
import ktb
import os
import config_LfD as config
import shutil
import utils.vision_utils as vu
import open3d as o3d


def dump_frameseq(folder_name, time_interval=1.0, root_path=os.path.join(config.ROOT, "res/knt/")):
    k = ktb.Kinect()
    dump_path = os.path.join(root_path, folder_name)
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    else:
        shutil.rmtree(dump_path)
        os.makedirs(dump_path)

    i = 0
    time.sleep(3)
    while True:
        # Specify as many types as you want here
        rgbimg = k.get_frame(ktb.COLOR)
        depthimg = k.get_frame(ktb.DEPTH)
        pcd = k.get_ptcld()
        cv2.imshow('frame', rgbimg)
        key = cv2.waitKey(1)
        pickle.dump([depthimg, rgbimg, pcd], open(os.path.join(dump_path, f"{str(i).zfill(4)}.pkl"), 'wb'))
        print("Captured!", i)
        i += 1
        if time_interval > 0:
            time.sleep(time_interval)
        if key & 0xFF == ord('q') or key == 27:
            break


def load_frame_seq(folder_name, root_path=os.path.join(config.ROOT, "res/knt/")):
    depthimg_list = []
    rgbaimg_list = []
    pcd_list = []
    for f in sorted(os.listdir(os.path.join(root_path, folder_name))):
        tmp = pickle.load(open(os.path.join(root_path, folder_name, f), 'rb'))
        pcd_list.append(tmp[2])
        if tmp[0].shape[-1] == 3:
            depthimg_list.append(tmp[1])
            rgbaimg_list.append(tmp[0])
        else:
            depthimg_list.append(tmp[0])
            rgbaimg_list.append(tmp[1])
    return [depthimg_list, rgbaimg_list, pcd_list]


def show_frameseq(folder_name, root_path=os.path.join(config.ROOT, "res/knt/")):
    """

    :param folder_name:
    :param inx: 0 - depth image; 1 - rgb image
    :param root_path:
    :return:
    """
    print(root_path)
    data = load_frame_seq(folder_name, root_path=root_path)
    print(f"num of frames: {len(data[0])}")
    for img_id in range(len(data[0])):
        depth_img = data[0][img_id]
        color_img = data[1][img_id]
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        imgs = np.hstack((color_img, depth_colormap))
        cv2.namedWindow(folder_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(folder_name, imgs)
        cv2.waitKey(0)


def show_pcdseq(folder_name, root_path=os.path.join(config.ROOT, "res/knt/")):
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
    depthimg_list, rgbaimg_list, pcd_list = load_frame_seq(folder_name, root_path=root_path)
    pcd_list = []
    for depthimg in depthimg_list:
        pcd_list.append(vu.convert_depth2pcd(depthimg))
    print(f"num of frames: {len(pcd_list)}")
    taskMgr.doMethodLater(0.1, __update, "update", extraArgs=[pcldnp, counter, pcd_list], appendTask=True)

    base.run()


def show_rgbdseq(folder_name, root_path=os.path.join(config.ROOT, "res/knt/")):
    depthimg_list, rgbaimg_list, pcd_list = load_frame_seq(folder_name, root_path=root_path)
    pointcloud = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window('PCD', width=1280, height=720)
    try:
        knt = ktb.Kinect()
        intr = knt.intrinsic_parameters
        intr = {"width": 512, "height": 424, "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}
        pickle.dump(intr, open("./knt_intr.pkl", "wb"))
        print(intr)
    except:
        intr = pickle.load(open("./knt_intr.pkl", "rb"))
    pinhole_camera_intrinsic = \
        o3d.camera.PinholeCameraIntrinsic(512, 424, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    geom_added = False
    print(f"num of frames: {len(pcd_list)}")
    i = 0
    while True:
        if i >= len(depthimg_list):
            i = 0
        img_depth = o3d.geometry.Image(depthimg_list[i])
        img_color = o3d.geometry.Image(cv2.cvtColor(rgbaimg_list[i], cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

        if geom_added == False:
            vis.add_geometry(pointcloud)
            geom_added = True

        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

        cv2.imshow('rgb', rgbaimg_list[i])
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        i += 1
    cv2.destroyAllWindows()
    vis.destroy_window()
    del vis


if __name__ == '__main__':
    folder_name = "sakura"
    dump_frameseq(folder_name, time_interval=0)
    # show_frameseq(folder_name)
    # show_pcdseq(folder_name)
    show_rgbdseq(folder_name)
