import os
import open3d as o3d

folder_name = 'plate_a_cubic'
for f in sorted(os.listdir(folder_name)):
    if f[-3:] != 'pcd':
        continue
    o3dpcd = o3d.io.read_point_cloud(os.path.join(folder_name, f))
    o3d.visualization.draw_geometries([o3dpcd])
