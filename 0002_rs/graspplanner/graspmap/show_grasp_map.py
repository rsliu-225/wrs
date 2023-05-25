import pickle
import config
import motionplanner.motion_planner as m_planner
from collections import Counter

pen_f_name = "pentip_short"
graspmap = pickle.load(open(config.ROOT + "/graspplanner/graspmap/temp/" + pen_f_name + "_graspmap.pkl", "rb"))
for grasp_id, v in graspmap.items():
    print(grasp_id, Counter(v.values()))
