import pickle
import modeling.geometric_model as gm
import visualization.panda.world as pc

base = pc.World(cam_pos=[0, 0, -1], lookat_pos=[0, 0, 0], w=1024, h=768)
pcd = pickle.load(open('tst.pkl', 'rb'))
gm.gen_pointcloud(pcd).attach_to(base)
base.run()
