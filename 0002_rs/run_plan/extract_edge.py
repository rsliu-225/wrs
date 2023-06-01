import utils.pcd_utils as pcdu
import numpy as np
import pickle
import open3d as o3d
import basis.o3dhelper as o3dh
import visualization.panda.world as wd
import modeling.collision_model as cm

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])

    img, dep, pcd = pickle.load(open('C:/Users/rsliu/Documents/GitHub/wrs/0002_rs/img/all.pkl', 'rb'))
    pcd = np.asarray(pcdu.remove_pcd_zeros(pcd))
    o3dpcd = o3dh.nparray2o3dpcd(pcd)
    plane_model, inliers = o3dpcd.segment_plane(distance_threshold=0.01,
                                                ransac_n=3,
                                                num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    plane_o3dpcd = o3dpcd.select_by_index(inliers)
    plane_o3dpcd.paint_uniform_color([1, 0, 0])
    o3dpcd = o3dpcd.select_by_index(inliers, invert=True)

    pcd = np.asarray(o3dpcd.points)
    print(pcd.shape)
    p_list, nrmls = pcdu.detect_edge(pcd, voxel_size=.01, toggledebug=True)

    pcdu.show_pcd(pcd)
    base.run()
