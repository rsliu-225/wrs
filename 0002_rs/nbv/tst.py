import open3d as o3d
import numpy as np


def get_meterial(rgba):
    material = o3d.visualization.rendering.Material()
    material.shader = 'defaultLitTransparency'
    # material.shader = 'defaultLitSSR'
    material.base_color = [0.467, 0.467, 0.467, alpha]
    material.base_roughness = 0.0
    material.base_reflectance = 0.0
    material.base_clearcoat = 1.0
    material.thickness = 1.0
    material.transmission = 1.0
    material.absorption_distance = 10
    material.absorption_color = [0.5, 0.5, 0.5]

    return material


sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
sphere.compute_vertex_normals()
sphere.translate(np.array([0, 0, -3.5]))

box = o3d.geometry.TriangleMesh.create_box(2, 4, 4)
box.translate(np.array([-1, -2, -2]))
box.compute_triangle_normals()

mat_sphere = o3d.visualization.rendering.Material()
mat_sphere.shader = 'defaultLit'
mat_sphere.base_color = [0.8, 0, 0, 1.0]

mat_box = get_meterial()

geoms = [{'name': 'sphere', 'geometry': sphere, 'material': mat_sphere},
         {'name': 'box', 'geometry': box, 'material': mat_box}]

o3d.visualization.draw(geoms)
