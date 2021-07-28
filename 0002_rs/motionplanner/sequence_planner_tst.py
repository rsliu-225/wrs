from graspplanner.handover_planner import HandoverPlanner
from motionplanner.sequence_planner import SequencePlanner
import utiltools.robotmath as rm
from gui.core.inspector import InspectorPanel
from debuger.inspector import Inspector
from motionplanner.state import *
from localenv import envloader as el
import numpy as np
import copy
import config

STATE = config.STATE

if __name__ == "__main__":
    '''
    load env
    '''
    base, env = el.loadEnv_wrs()
    rbt, rbtmg, rbtball = el.loadUr3e()
    obscmlist = env.getstationaryobslist() + env.getchangableobslist()

    hmstr = HandoverPlanner("pentip.stl", rbt, rbtball, retractdistance=60)
    objcm = hmstr.objcm

    initilizestate(base=base, taskMgr=taskMgr, rbt=rbt, obscmlist=obscmlist, inspector=Inspector(), objcm=objcm)

    # start
    startmat4 = rm.homobuild(np.array([600, 250, 900]), rm.rodrigues([1, 0, 0, ], 0))
    objcmstart = copy.deepcopy(objcm)
    objcmstart.sethomomat(startmat4)
    objcmstart.reparentTo(base.render)
    objcmstart.setColor(1, 0, 0, 1)
    objcmstart.showlocalframe()
    # goal
    goalmat4 = rm.homobuild(np.array([700, -150, 900]), rm.rodrigues([0, 0, 1, ], 90))
    objcmgoal = copy.deepcopy(objcm)
    objcmgoal.sethomomat(goalmat4)
    objcmgoal.reparentTo(base.render)
    objcmgoal.setColor(0, 0, 1, 1)

    # base.run()
    sm = SequencePlanner(hndovermaster=hmstr, obstaclecmlist=obscmlist, gobacktoinitafterplanning=False, debug=False,
                         inspector=STATE["inspector"])

    # Sequence Master:
    #   __init__     1. Load Handover grasps
    #   addStartGoal 2. Add the grasps at the start and the goal.
    #                   Connect the grasps of the start and goal with the handover grasps
    #   planRegrasp  3. Find a shortest path that can grasp the object and move it from start to the goal

    sm.addStartGoal(
        startrotmat4=goalmat4,
        goalrotmat4=startmat4,
        choice="startrgtgoallft",
        # choice="startlftgoallft",
    )

    try:
        objmsmp, numikrmsmp, jawwidthmp, originalpathnidlist = sm.planRegrasp(objcm=objcm)

    except:
        print("plan failed, enter the debug mode")
        print(STATE)

    # print(objmsmp)
    # print(numikrmsmp)
    # print(jawwidthmp)
    # print(originalpathnidlist)
    registeration([
        [InspectorPanel, STATE["inspector"]]
    ])

    base.run()
