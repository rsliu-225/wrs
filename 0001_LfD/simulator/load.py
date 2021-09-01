import pickle

import numpy as np

import modeling.geometric_model as gm
import visualization.panda.world as pc

pcd = pickle.load(open('tst.pkl', 'rb'))
center = np.mean(pcd, axis=0)
base = pc.World(cam_pos=center + [0, 0, -10], lookat_pos=center, w=1024, h=768)
gm.gen_pointcloud(pcd).attach_to(base)
base.run()
