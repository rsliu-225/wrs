import random
import trimesh as trm
import numpy as np
from skimage import measure
import modeling.geometric_model as gm
import modeling.collision_model as cm


def create_voxel_grid(num_cubes, size, position_start):
    grid = np.zeros((num_cubes + 2, num_cubes + 2, num_cubes + 2), dtype=bool)
    current_position = np.array([1, 1, 1])

    directions = [
        np.array([size, 0, 0]),
        np.array([-size, 0, 0]),
        np.array([0, size, 0]),
        np.array([0, -size, 0]),
        np.array([0, 0, size]),
        np.array([0, 0, -size]),
    ]

    for _ in range(num_cubes):
        grid[tuple(current_position)] = True
        direction = random.choice(directions)
        current_position += direction

    return grid


def generate_smooth_mesh(voxel_grid, level):
    verts, faces, _, _ = measure.marching_cubes(voxel_grid, level)
    mesh = trm.Trimesh(vertices=np.asarray(verts)/10, faces=faces)
    return mesh


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])

    num_cubes = 10
    size = 1
    position_start = np.array([1, 1, 1])

    voxel_grid = create_voxel_grid(num_cubes, size, position_start)
    smooth_mesh = generate_smooth_mesh(voxel_grid, 0.5)

    objsgm = gm.StaticGeometricModel(initor=smooth_mesh, btwosided=True)
    objgm = gm.GeometricModel(initor=objsgm)
    objcm = cm.CollisionModel(initor=objgm)
    objcm.attach_to(base)
    base.run()
