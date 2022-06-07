from gen_dataset import *
import visualization.panda.world as wd

cam_pos = np.asarray([0, 0, .5])
base = wd.World(cam_pos=cam_pos, lookat_pos=[0, 0, 0])
width = .005
thickness = .0015

pseq = gen_sgl_curve(step=.001, pseq=np.asarray([[0, 0, 0], [.018, .03, 0], [.07, -0.01, 0], [.12, 0, 0]]))
rotseq = get_rotseq_by_pseq(pseq)
cross_sec = [[0, width / 2], [0, -width / 2], [-thickness / 2, -width / 2], [-thickness / 2, width / 2]]

objcm = gen_swap(pseq, rotseq, cross_sec)
objcm.set_rgba((1, 1, 1, 1))
objcm.attach_to(base)

rot_center = (0, 0, 0)
res = (1000, 900)
# res = (100, 90)
fact = 5.0


def init_gen(rng_x, rng_y, rng_z, identifier):
    partial_path = './partial' + identifier
    for x in rng_x:
        for y in rng_y:
            for z in rng_z:
                rot = rm.rotmat_from_axangle((1, 0, 0), x * np.pi / fact) \
                    .dot(rm.rotmat_from_axangle((0, 1, 0), y * np.pi / fact)) \
                    .dot(rm.rotmat_from_axangle((0, 0, 1), z * np.pi / fact))
                get_objpcd_partial_o3d(objcm, rot, rot_center, path=partial_path,
                                       f_name='_'.join([str(x), str(y), str(z)]),
                                       resolusion=res)


def display(rng_x, rng_y, rng_z):
    for x in rng_x:
        for y in rng_y:
            for z in rng_z:
                o3dmesh = o3d.io.read_triangle_mesh(f"./partial/{'_'.join([str(x), str(y), str(z)])}.ply")
                objcm = o3dmesh2cm(o3dmesh)
                o3dpcd = o3d.io.read_point_cloud(f"./partial/{'_'.join([str(x), str(y), str(z)])}.pcd")
                print(f"{'_'.join([str(x), str(y), str(z)])}", end=":")
                print(o3dpcd)
                gm.gen_pointcloud(o3dpcd.points, pntsize=5).attach_to(base)
                objcm.set_rgba((1, 1, 1, 1))
                objcm.attach_to(base)

    objpcd = get_objpcd_full_sample(objcm)
    gm.gen_pointcloud(objpcd, pntsize=5, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()


x_r, y_r, z_r = list(range(10)), list(range(10)), list(range(10))
# init_gen(x_r, y_r, z_r)

x_r, y_r, z_r = list(range(4)), list(range(4)), list(range(4))
display(x_r, y_r, z_r)
