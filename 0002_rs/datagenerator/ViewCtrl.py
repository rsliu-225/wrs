import open3d as o3d
import numpy as np
import json


class ViewControl(object):
    def __init__(self, json_f=None):
        # if json_f is not None:
        self.width = 1920
        self.height = 1080
        self.foc = .9
        self.intrinsic_matrix = np.asarray([[self.foc * self.width, 0, self.width / 2],
                                            [0, self.foc * self.width, self.height / 2],
                                            [0, 0, 1]])
        self.extrinsic_traj = []
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(width=self.width, height=self.height)
        # self.vis.get_render_option().load_from_json("./param/renderoption.json")
        # self.ctr = self.vis.get_view_control()

    def init_vis(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height)
        vis.get_render_option().load_from_json("./param/renderoption.json")
        ctr = vis.get_view_control()
        return vis, ctr

    def load_json(self, f_path):
        return json.load(open(f_path, "r"))

    def write_traj(self, f_path):
        param_json = {"class_name": "PinholeCameraTrajectory", "parameters": []}
        for extrinsic in self.extrinsic_traj:
            param_json['paramters'].append({
                "class_name": "PinholeCameraParameters",
                "extrinsic": extrinsic,
                "intrinsic":
                    {
                        "height": self.height,
                        "intrinsic_matrix": self.intrinsic_matrix,
                        "width": self.width
                    },
                "version_major": 1,
                "version_minor": 0
            })
        json.dump(param_json, open(f_path, "w"))

    def write_param(self, f_path, extrinsic):
        extrinsic = np.asarray(extrinsic).flatten()
        param_json = {"class_name": "PinholeCameraParameters",
                      "extrinsic": extrinsic,
                      "intrinsic":
                          {
                              "height": self.height,
                              "intrinsic_matrix": self.intrinsic_matrix,
                              "width": self.width
                          },
                      "version_major": 1,
                      "version_minor": 0
                      }
        json.dump(param_json, open(f_path, "w"))

    # def get_extrinsic(self,ctr):


    def set_param(self, ctr, param_path="./param/camera_param.json"):
        param = o3d.io.read_pinhole_camera_parameters(param_path)
        ctr.convert_from_pinhole_camera_parameters(param)
        return False

    def capture_pcd(self, o3dmesh, f_path, param_path="./param/camera_param.json"):
        vis, ctr = self.init_vis()
        vis.create_window(width=self.width, height=self.height)
        self.set_param(ctr, param_path=param_path)
        vis.add_geometry(o3dmesh)
        vis.poll_events()
        vis.capture_depth_point_cloud(f_path, do_render=False, convert_to_world_coordinate=True)
        vis.run()
        vis.destroy_window()

    def show(self, geo, param_path=None):
        vis, ctr = self.init_vis()
        vis.create_window(width=self.width, height=self.height)
        if param_path is not None:
            self.set_param(ctr, param_path=param_path)
        vis.add_geometry(geo)
        vis.poll_events()
        vis.run()
        vis.destroy_window()


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.1, .2, .4], lookat_pos=[0, 0, 0])

    vctrl = ViewControl()

    o3dmesh = o3d.io.read_triangle_mesh('./tst/mesh/0_000.ply')
    # vctrl.capture_pcd(o3dmesh, './tst/tst.pcd', param_path="./param/camera_param.json")
    # vctrl.capture_pcd(o3dmesh, './tst/tst_new.pcd', param_path="./param/camera_param_new.json")

    # o3dpcd = o3d.io.read_point_cloud('./tst/tst.pcd')
    # print(np.asarray(o3dpcd.points).shape)
    vctrl.show(o3dmesh, param_path="./param/camera_param.json")
    vctrl.show(o3dmesh, param_path="./param/camera_param_new.json")

    # o3dpcd = o3d.io.read_point_cloud('./tst/tst.pcd')
    # gm.gen_pointcloud(np.asarray(o3dpcd.points)).attach_to(base)
    # base.run()
