import numpy as np
import open3d as o3d
import modeling.collision_model as cm

objcm = cm.CollisionModel('../obstacles/plate.stl')
mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(objcm.objtrm.vertices),
                                 triangles=o3d.utility.Vector3iVector(objcm.objtrm.faces))

vertices = np.asarray(mesh.vertices)
print(vertices)
static_ids = [idx for idx in np.where(vertices[:, 1] < 0)[0]]
static_pos = []
for id in static_ids:
    static_pos.append(vertices[id])
handle_ids = [500]
handle_pos = [vertices[500] + np.array((-.1, -.1, -.1))]
constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)

print('Original Mesh')
R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
o3d.visualization.draw_geometries([mesh.rotate(R, center=mesh.get_center())])
print('Deformed Mesh')
mesh_prime.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_prime.rotate(R, center=mesh_prime.get_center())])
