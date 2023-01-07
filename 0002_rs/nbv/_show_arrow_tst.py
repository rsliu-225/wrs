import numpy as np
import open3d as o3d
import basis.robot_math as rm


def gen_o3d_arrow(spos, epos):
    vec_len = np.linalg.norm(np.array(epos) - np.array(spos))
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06 * vec_len,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04 * vec_len
    )
    mesh_arrow.paint_uniform_color([1, 0, 1])
    mesh_arrow.compute_vertex_normals()

    rot_mat = rm.rotmat_between_vectors((0, 0, 1), np.array(epos) - np.array(spos))
    mesh_arrow.rotate(rot_mat, center=(0, 0, 0))
    mesh_arrow.translate(np.array(begin))

    return mesh_arrow


if __name__ == "__main__":
    z_unit_Arr = np.array([0, 0, 1])
    begin = [.5, 0, 0]
    end = [.5, 0.04, 0.08]
    vec_Arr = np.array(end) - np.array(begin)
    vec_len = np.linalg.norm(vec_Arr)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    mesh_arrow = gen_o3d_arrow(begin, end)
    o3d.visualization.draw_geometries(
        geometry_list=[mesh_frame, mesh_arrow],
        window_name="after rotate", width=800, height=600
    )
