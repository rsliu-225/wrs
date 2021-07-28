from localenv import envloader as el
import numpy as np
import graspplanner.handover_planner as hop

if __name__ == "__main__":
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e(showrbt=True)
    ho_planner = hop.HandoverPlanner("pentip.stl", rbt, rbtball, retractdistance=60)
    ho_planner.genhvgpsgl(np.array([700, 0, 1100]), debug=False)
