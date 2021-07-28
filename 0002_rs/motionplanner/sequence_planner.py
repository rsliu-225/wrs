import copy
import itertools
import math
import os
import pickle

import networkx as nx
import numpy as np
from matplotlib import collections as mc

import config
import motion.rrt.ddrrtconnect as ddrrtc
import motion.smoother as sm
from debuger.inspector import Error_info
from motion import checker as ck
from utiltools import robotmath as rm


class SequencePlanner(object):
    def __init__(self, hndovermaster, obstaclecmlist, rethanda=60, gobacktoinitafterplanning=False, graphdebug=False,
                 debug=False, inspector=None):
        self.rbt = hndovermaster.rbt
        self.rbtball = hndovermaster.rbtball
        self.inspector = inspector

        self.cdchecker = hndovermaster.cdchecker
        self.hmstr = hndovermaster
        self.obstaclecmlist = obstaclecmlist
        self.gobacktoinitafterplanning = gobacktoinitafterplanning

        # start and goal
        self.startrgtnodeids = []  # start nodes rgt arm
        self.startlftnodeids = []  # start nodes lft arm
        self.goalrgtnodeids = []  # goal nodes rgt arm
        self.goallftnodeids = []  # goal nodes lft arm
        self.shortestpaths = None  # shortest path

        # retract distance
        self.__rethanda = rethanda  # retract  distance
        self.retworlda = 60  # retact distance 2: make sure no collision

        # build the graph
        # load graph
        if not os.path.exists(os.path.join(config.ROOT, "motionplanner", "hndovr_graph")):
            os.mkdir(os.path.join(config.ROOT, "motionplanner", "hndovr_graph"))
        try:
            # when update the grasps and the handover, remember to delete the hndovr_graph
            self.loadhandovergrasp()
            if not graphdebug:
                raise Exception("data graph")
            with open(os.path.join(config.ROOT, "motionplanner", "hndovr_graph", self.hmstr.objname + "_graph.pkl"),
                      "rb") as f:
                self.regg = pickle.load(f)
        except:
            self.regg = nx.Graph()  # graph
            self.loadhandovergrasp()
            self.buildGraph(armname="rgt", fpglist=self.fpsnestedglist_rgt)  # add the handover grasps of the rgt arm
            self.buildGraph(armname="lft", fpglist=self.fpsnestedglist_lft)  # add the handover grasps of the left arm
            self.bridgeGraph()  # connect the handover grasps of the left arm and rgt arm
            with open(os.path.join(config.ROOT, "motionplanner", "hndovr_graph", self.hmstr.objname + "_graph.pkl"),
                      "wb") as f:
                pickle.dump(self.regg, f)

        self.reggbk = copy.deepcopy(self.regg)  # backup of the graph: restore the handover nodes

        # shortestpaths
        self.directshortestpaths_startrgtgoalrgt = []
        self.directshortestpaths_startrgtgoallft = []
        self.directshortestpaths_startlftgoalrgt = []
        self.directshortestpaths_startlftgoallft = []

        self.debug = debug

    def loadhandovergrasp(self):
        self.grasprgthnd, self.grasplfthnd, self.fpsnpmat4, self.identitygplist, \
        self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
        self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
        self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft = self.hmstr.gethandover()
        self.hndovrpos = [pos for pos in self.fpsnpmat4]
        self.feasiblefpgpairlist = {}
        for posid in range(len(self.fpsnpmat4)):
            if posid in self.ikfid_fpsnestedglist_rgt.keys() and posid in self.ikfid_fpsnestedglist_lft.keys():
                pass
            else:
                continue
            for i0, i1 in self.identitygplist:
                if i0 in self.ikfid_fpsnestedglist_rgt[posid] and i1 in self.ikfid_fpsnestedglist_lft[posid]:
                    if posid not in self.feasiblefpgpairlist:
                        self.feasiblefpgpairlist[posid] = []
                    self.feasiblefpgpairlist[posid].append([i0, i1])
                else:
                    continue
            # base.run()

    def reset(self):
        self.regg = copy.deepcopy(self.reggbk)
        # shortestpaths
        self.directshortestpaths_startrgtgoalrgt = []
        self.directshortestpaths_startrgtgoallft = []
        self.directshortestpaths_startlftgoalrgt = []
        self.directshortestpaths_startlftgoallft = []

        self.startrgtnodeids = []
        self.startlftnodeids = []
        self.goalrgtnodeids = []
        self.goallftnodeids = []
        self.shortestpaths = None

        self.startrotmat4 = None
        self.goalrotmat4 = None

    def buildGraph(self, armname, fpglist):
        if armname == "rgt":
            ik_feasible_jnts = self.ikjnts_fpsnestedglist_rgt
        else:
            ik_feasible_jnts = self.ikjnts_fpsnestedglist_lft

        globalidsedges = {}  # the global id of the edges?
        # handpairList: possible handpairList
        for posid in range(len(self.fpsnpmat4)):  # iterate the handover positions
            addednodelist = []
            # {posid: possible handover pair [rgt hand grasp id, left hand grasp id]}
            if posid not in self.feasiblefpgpairlist:
                continue
            for pairind, (rgtid, lftid) in enumerate(self.feasiblefpgpairlist[posid]):  # i0 is the rgt hand grasp id
                if armname == "rgt":
                    graspid = rgtid
                else:
                    graspid = lftid
                if graspid in addednodelist:
                    continue
                addednodelist.append(graspid)
                hndrotmat4 = fpglist[posid][graspid][2]  # grasp mat4
                fpgid = graspid  # floating pose grasp id
                handa = - fpglist[posid][graspid][3]  # negative of z (negative direction of the hand)
                fpgfgrcenter = fpglist[posid][graspid][1]  # center of the floating grasp
                fpgfgrcenterhanda = fpgfgrcenter + handa * self.hmstr.retractdistance
                # the place that is negative direction of along of the center of the floating grasp
                jawwidth = fpglist[posid][graspid][0]  # jawidth
                fpjnts = np.array(ik_feasible_jnts[posid][graspid][0])  # floating pose jnts
                fpjnts_handa = np.array(ik_feasible_jnts[posid][graspid][1])
                hndrotmat3 = hndrotmat4[:3, :3]
                # manipulability
                self.rbt.movearmfk(fpjnts_handa, armname=armname)
                manipulability = self.rbt.manipulability(armname=armname)
                self.regg.add_node('ho' + armname + str(fpgid) + 'pos' + str(posid), fgrcenter=fpgfgrcenter,
                                   fgrcenterhanda=fpgfgrcenterhanda, jawwidth=jawwidth,
                                   hndrotmat3np=hndrotmat3,
                                   armjnts=fpjnts,
                                   armjntshanda=fpjnts_handa,
                                   floatingposegrippairind=pairind,
                                   handoverposid=posid,
                                   identity=armname + str(fpgid),
                                   manipulability=manipulability
                                   )

                if armname + str(fpgid) not in globalidsedges:  # {armname+fpgid: }
                    globalidsedges[armname + str(fpgid)] = []
                globalidsedges[armname + str(fpgid)].append('ho' + armname + str(fpgid) + 'pos' + str(posid))
        for globalidedgesid in globalidsedges:
            if len(globalidsedges[globalidedgesid]) == 1:
                continue
            for edge in list(itertools.combinations(globalidsedges[globalidedgesid], 2)):
                self.regg.add_edge(*edge, weight=1, edgetype='transfer')

    def bridgeGraph(self):
        for posid, objrotmat4 in enumerate(self.fpsnpmat4):
            pass
            if posid not in self.feasiblefpgpairlist:
                continue
            for rgtid, lftid in self.feasiblefpgpairlist[posid]:
                rgtnode_name = 'horgt' + str(rgtid) + "pos" + str(posid)
                lftnode_name = 'holft' + str(lftid) + "pos" + str(posid)
                self.regg.add_edge(rgtnode_name, lftnode_name,
                                   weight=1, edgetype='handovertransit')

                if self.inspector is not None:
                    self.inspector.add_error("handover", Error_info(
                        # name='horgt' + str(rgtid) + "pos" + str(posid) + "---" + 'holft' + str(lftid) + "pos" + str(
                        #     posid),
                        name=f'horgt{str(rgtid)}pos{str(posid)}---holft{str(lftid)}pos{str(posid)}',
                        lftarmjnts=self.regg.nodes['holft' + str(lftid) + "pos" + str(posid)]['armjnts'],
                        rgtarmjnts=self.regg.nodes['horgt' + str(rgtid) + "pos" + str(posid)]['armjnts'],
                        lftjawwidth=self.regg.nodes['holft' + str(lftid) + "pos" + str(posid)]['jawwidth'],
                        rgtjawwidth=self.regg.nodes['horgt' + str(rgtid) + "pos" + str(posid)]['jawwidth']), )
                    print("error",self.inspector.error_detail)

    def addStartGoal(self, startrotmat4, goalrotmat4, choice, starttoolvec=None, goaltoolvec=None,
                     possiblegrasp=None, starttmpobstacle=[], goaltmpobstacle=[]):
        """
        add start and goal to the grasph
        if start/goalgrasppose is not None, the only pose will be used
        the pose is defined by a numpy 4x4 homomatrix

        :param startrotmat4: numpy matrix
        :param goalrotmat4: numpy matrix
        :param choice in "startrgtgoallft" "startrgtgoalrgt" "startlftgoalrgt" "startrgtgoallft"
        :param startgraspgid:
        :param goalgraspgid:
        :param starttoolvec
        :param goaltoolvec there are three choices for the tool vecs: None indicates global z, [0,0,0] indicates no tool vec
        :return:

        author: weiwei
        date: 20180925
        """
        # self.grasp = possiblegrasp

        if starttoolvec is not None:
            starttoolVec3 = np.array([starttoolvec[0], starttoolvec[1], starttoolvec[2]])
        else:
            starttoolVec3 = None
        if goaltoolvec is not None:
            goaltoolVec3 = np.array([goaltoolvec[0], goaltoolvec[1], goaltoolvec[2]])
        else:
            goaltoolVec3 = None

        self.startrotmat4 = startrotmat4
        self.goalrotmat4 = goalrotmat4

        self.choice = choice
        startchoice = choice[:8]
        goalchoice = choice[8:]

        print("startgraspgid is None, all grasps are candidates")
        self.__addend(startrotmat4, cond=startchoice, worldframevec3=starttoolVec3, tempobstacle=starttmpobstacle)

        print("goalgraspgid is None, all grasps are candidates")
        self.__addend(goalrotmat4, goalchoice, worldframevec3=goaltoolVec3, tempobstacle=goaltmpobstacle)

        # if self.debug:
        #     base.run()

        # add start to goal direct edges rgt-rgt
        for startnodeid in self.startrgtnodeids:
            for goalnodeid in self.goalrgtnodeids:
                # startnodeggid = start node global grip id
                startnodeggid = self.regg.node[startnodeid]['identity']
                goalnodeggid = self.regg.node[goalnodeid]['identity']
                print(startnodeggid, goalnodeggid)
                if startnodeggid == goalnodeggid:
                    self.regg.add_edge(startnodeid, goalnodeid, weight=1, edgetype='startgoalrgttransfer')

        # add start to goal direct edges lft-lft
        for startnodeid in self.startlftnodeids:
            for goalnodeid in self.goallftnodeids:
                # startnodeggid = start node global grip id
                startnodeggid = self.regg.node[startnodeid]['identity']
                goalnodeggid = self.regg.node[goalnodeid]['identity']
                if startnodeggid == goalnodeggid:
                    self.regg.add_edge(startnodeid, goalnodeid, weight=1, edgetype='startgoallfttransfer')

    def __addend(self, objposmat, cond="startrgt", worldframevec3=None, tempobstacle=[]):
        """
        add a start or a goal for the regg, using different hand

        :param objposmat:
        :param cond: the specification of the rotmat4: "startrgt", "startlft", "goalrgt", "goallft"
        :param ctvec, ctangle: the conditions of filtering, the candidate hand z must have a smaller angle with vec
        :param toolvec: the direction to move the tool in the last step, it is described in the local coordinate system of the object
        :return:

        author: weiwei
        date: 20180925
        """

        if worldframevec3 is None:
            worlda = np.array([0, 0, 1])
        else:
            worlda = worldframevec3

        feasiblegrasps = 0
        ikfailedgrasps = 0
        ikhndafailedgrasps = 0
        handcollidedgrasps = 0
        if self.debug:
            print(f"The hand is {self.hmstr.rgthndfa.name if cond[-3:] == 'rgt' else self.hmstr.lfthndfa.name}")

        # the nodeids is also for quick access
        if cond == "startrgt":
            self.startrgtnodeids = []
            nodeids = self.startrgtnodeids
        elif cond == "startlft":
            self.startlftnodeids = []
            nodeids = self.startlftnodeids
        elif cond == "goalrgt":
            self.goalrgtnodeids = []
            nodeids = self.goalrgtnodeids
        elif cond == "goallft":
            self.goallftnodeids = []
            nodeids = self.goallftnodeids
        else:
            raise Exception("Wrong conditions!")
        if "rgt" in cond:
            grasps = self.grasprgthnd
        else:
            grasps = self.grasplfthnd
        # the node id of a globalgripid
        nodeidofglobalid = {}
        for graspid, graspinfo in enumerate(grasps):
            rotmat = graspinfo[2]  # the mat of the grasp
            ttgsrotmat = np.dot(objposmat, rotmat)  # the grasp at the obj posision
            # filtering
            handa = -ttgsrotmat[:3, 2]
            # vector between object and robot ee
            # check if the hand collide with obstacles
            # set jawwidth to 50 to avoid collision with surrounding obstacles
            # set to gripping with is unnecessary
            ishndcollided = self.hmstr.checkhndenvcollision(ttgsrotmat, self.obstaclecmlist + tempobstacle,
                                                            armname=cond[-3:], debug=self.debug)
            if not ishndcollided:
                ttgsfgrcenternp = rm.homotransformpoint(objposmat, graspinfo[1])
                ttgsfgrcenternp_handa = ttgsfgrcenternp + handa * self.__rethanda
                ttgsfgrcenternp_worlda = ttgsfgrcenternp + worlda * self.retworlda
                ttgsjawwidth = graspinfo[0]
                ttgsrotmat3 = ttgsrotmat[:3, :3]
                ikr_handa = None
                ikr_worlda = None
                ikr = self.rbt.numik(ttgsfgrcenternp, ttgsrotmat3, armname=cond[-3:])
                if ikr is not None:
                    ikr_handa = self.rbt.numikmsc(ttgsfgrcenternp_handa, ttgsrotmat3, msc=ikr, armname=cond[-3:])
                    if ikr_handa is not None:
                        ikr_worlda = self.rbt.numikmsc(ttgsfgrcenternp_worlda, ttgsrotmat3, msc=ikr, armname=cond[-3:])
                # ikcaz = self.robot.numikr(ttgsfgrcenternp_worldaworldz, ttgsrotmat3np, armname = 'rgt')
                if (ikr is not None) and (ikr_handa is not None) and (ikr_worlda is not None):
                    feasiblegrasps += 1
                    # note the tabletopposition here is not the contact for the intermediate states
                    # it is the zero pos
                    objposmat_copy = copy.deepcopy(objposmat)
                    tabletopposition = objposmat_copy[:3, :3]
                    startrotmat4worlda = copy.deepcopy(objposmat_copy)
                    startrotmat4worlda[:3, 3] = objposmat_copy[:3, 3] + worlda * self.retworlda
                    # manipulability
                    self.rbt.movearmfk(ikr_worlda, armname=cond[-3:])
                    manipulability = self.rbt.manipulability(armname=cond[-3:])
                    self.regg.add_node(cond + str(graspid), fgrcenter=ttgsfgrcenternp,
                                       fgrcenterhanda=ttgsfgrcenternp_handa,
                                       fgrcenterworlda=ttgsfgrcenternp_worlda,
                                       jawwidth=ttgsjawwidth, hndrotmat3np=ttgsrotmat3,
                                       armjnts=ikr,
                                       armjntshanda=ikr_handa,
                                       armjntsworlda=ikr_worlda,
                                       tabletopplacementrotmat=objposmat_copy,
                                       tabletopposition=tabletopposition,
                                       tabletopplacementrotmathanda=objposmat_copy,
                                       tabletopplacementrotmatworlda=startrotmat4worlda,
                                       identity=cond[-3:] + str(graspid),
                                       manipulability=manipulability
                                       )
                    nodeidofglobalid[cond[-3:] + str(graspid)] = cond + str(graspid)
                    nodeids.append(cond + str(graspid))
                    # tmprtq85.reparentTo(base.render)
                else:
                    if ikr is None:
                        ikfailedgrasps += 1
                    else:
                        ikhndafailedgrasps += 1
            else:
                handcollidedgrasps += 1

        # base.run()
        print("IK failed grasps:", ikfailedgrasps)
        print("IK handa failed grasps", ikhndafailedgrasps)
        print("Hand collided grasps:", handcollidedgrasps)
        print("feasible grasps:", feasiblegrasps)

        if len(nodeids) == 0:
            print("No available " + cond[:-3] + " grip for " + cond[-3:] + " hand!")

        for edge in list(itertools.combinations(nodeids, 2)):
            self.regg.add_edge(*edge, weight=1, edgetype=cond + 'transit')

        # add transfer edge
        for reggnode, reggnodedata in self.regg.nodes(data=True):
            if reggnode.startswith(cond[-3:]) or reggnode.startswith('ho' + cond[-3:]):
                globalgripid = reggnodedata['identity']
                if globalgripid in nodeidofglobalid.keys():
                    nodeid = nodeidofglobalid[globalgripid]
                    self.regg.add_edge(nodeid, reggnode, weight=1, edgetype=cond + 'transfer')

    def updateshortestpath(self):
        """
        this function is assumed to be called after start and goal are set

        :return:
        """

        # startrgt goalrgt
        if len(self.startrgtnodeids) > 0 and len(self.goalrgtnodeids) > 0:
            print("Number of start grasps: ", len(self.startrgtnodeids), "; Number of goal grasps: ",
                  len(self.goalrgtnodeids))
            startgrip = self.startrgtnodeids[0]
            goalgrip = self.goalrgtnodeids[0]
            self.shortestpaths = nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip)
            self.directshortestpaths_startrgtgoalrgt = []
            try:
                for path in self.shortestpaths:
                    for i, pathnode in enumerate(path):
                        if pathnode.startswith('start') and i < len(path) - 1:
                            continue
                        else:
                            self.directshortestpaths_startrgtgoalrgt.append(path[i - 1:])
                            break
                    for i, pathnode in enumerate(self.directshortestpaths_startrgtgoalrgt[-1]):
                        if i > 0 and pathnode.startswith('goal'):
                            self.directshortestpaths_startrgtgoalrgt[-1] = self.directshortestpaths_startrgtgoalrgt[-1][
                                                                           :i + 1]
                            break
            except:
                # base.run()
                raise Exception('No startrgtgoalrgt')

        # startrgt goallft
        if len(self.startrgtnodeids) > 0 and len(self.goallftnodeids) > 0:
            print("Number of start grasps: ", len(self.startrgtnodeids), "; Number of goal grasps: ",
                  len(self.goallftnodeids))
            startgrip = self.startrgtnodeids[0]
            goalgrip = self.goallftnodeids[0]
            self.shortestpaths = nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip)
            self.directshortestpaths_startrgtgoallft = []
            try:
                for path in self.shortestpaths:
                    for i, pathnode in enumerate(path):
                        if pathnode.startswith('start') and i < len(path) - 1:
                            continue
                        else:
                            self.directshortestpaths_startrgtgoallft.append(path[i - 1:])
                            break
                    for i, pathnode in enumerate(self.directshortestpaths_startrgtgoallft[-1]):
                        if i > 0 and pathnode.startswith('goal'):
                            self.directshortestpaths_startrgtgoallft[-1] = self.directshortestpaths_startrgtgoallft[-1][
                                                                           :i + 1]
                            break
            except:
                raise Exception('No startrgtgoallft')

        # startlft goalrgt
        if len(self.startlftnodeids) > 0 and len(self.goalrgtnodeids) > 0:
            print("Number of start grasps: ", len(self.startlftnodeids), "; Number of goal grasps: ",
                  len(self.goalrgtnodeids))
            startgrip = self.startlftnodeids[0]
            goalgrip = self.goalrgtnodeids[0]
            self.shortestpaths = nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip)
            self.directshortestpaths_startlftgoalrgt = []
            # first n obj in self.shortestpaths
            maxiter = 20
            counter = 0
            tmpshortestpaths = []
            for node in self.shortestpaths:
                tmpshortestpaths.append(node)
                counter += 1
                if counter >= maxiter:
                    break
            self.shortestpaths = tmpshortestpaths
            self.shortestpaths.sort(
                key=lambda element: sum([self.regg.nodes[node]['manipulability'] for node in element]), reverse=True)
            try:
                for path in self.shortestpaths:
                    for i, pathnode in enumerate(path):
                        if pathnode.startswith('start') and i < len(path) - 1:
                            continue
                        else:
                            self.directshortestpaths_startlftgoalrgt.append(path[i - 1:])
                            break
                    for i, pathnode in enumerate(self.directshortestpaths_startlftgoalrgt[-1]):
                        if i > 0 and pathnode.startswith('goal'):
                            self.directshortestpaths_startlftgoalrgt[-1] = self.directshortestpaths_startlftgoalrgt[-1][
                                                                           :i + 1]
                            break
            except:
                raise Exception('No startlftgoalrgt')

        # startlft goallft
        if len(self.startlftnodeids) > 0 and len(self.goallftnodeids) > 0:
            print("Number of start grasps: ", len(self.startlftnodeids), "; Number of goal grasps: ",
                  len(self.goallftnodeids))
            startgrip = self.startlftnodeids[0]
            goalgrip = self.goallftnodeids[0]
            self.shortestpaths = nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip)
            self.directshortestpaths_startlftgoallft = []
            try:
                for path in self.shortestpaths:
                    for i, pathnode in enumerate(path):
                        if pathnode.startswith('start') and i < len(path) - 1:
                            continue
                        else:
                            self.directshortestpaths_startlftgoallft.append(path[i - 1:])
                            break
                    for i, pathnode in enumerate(self.directshortestpaths_startlftgoallft[-1]):
                        if i > 0 and pathnode.startswith('goal'):
                            self.directshortestpaths_startlftgoallft[-1] = self.directshortestpaths_startlftgoallft[-1][
                                                                           :i + 1]
                            break
                    # break
            except:
                raise Exception('No startlftgoallft')

    def getMotionSequence(self, id, type="OO", previous=[]):
        """
            generate motion sequence using the shortest path
            right arm
            this function is for simple pick and place with regrasp

            # 20190319 comment by weiwei
            five letters are attached to nids,
            they are "x", "w", "o", "c", "i"
            where "x" indicates handa,
            "w" indicates worlda,
            "o" and "c" are at grasping psoe, they indicate the open and close states of a hand
            "i" indicates initial pose
            these letter will be use to determine planning methods in the planner.py file
            e.g. an o->c motion will be simple finger motion, no rrt planners will be called
            a x->w will be planning with hold, x->c will be interplation, x->i will be planning without hold, etc.
            see the planning.py file for details

            #20190319 comment by weiwei
            OO means start from a hand open pose and stop at a hand open pose
            OC means start from a hand open pose and stop at a hand close pose
            CC means start from a hand close pose and stop at a hand close pose
            To generate multiple motion sequences, OC->CC->CC->...->CC->CO is the preferred type order choice


            :param: regrip an object of the regriptppfp.RegripTppfp class
            :param id: which path to plot
            :param choice: startrgtgoalrgt/startrgtgoallft/startlftgoalrgt/startlftgoallft
            :param type: one of "OO', "OC", "CC", "CO"
            :param previous: set it to [] if the motion is not a continuing one, or else, set it to [lastobjmat4, lastikr, lastjawwidth]

            :return: [[waist, lftbody, rgtbody],...]

            author: weiwei
            date: 20170302
            """

        if (self.choice not in ["startrgtgoalrgt", "startrgtgoallft", "startlftgoalrgt", "startlftgoallft"]):
            raise Exception("The choice parameter of getMotionSequence must be " +
                            "one of startrgtgoalrt, startrgtgoalft, startlftgoalrgt, startlftgoallft! " +
                            "Right now it is %s" % self.choice + ".")

        if (type not in ["OO", "OC", "CC", "CO"]):
            raise Exception("The choice parameter of type must be " +
                            "one of OO, OC, CC, CO! " + "Right now it is %s" % self.choice + ".")

        directshortestpaths = []
        if self.choice is 'startrgtgoalrgt':
            directshortestpaths = self.directshortestpaths_startrgtgoalrgt
        elif self.choice is 'startrgtgoallft':
            directshortestpaths = self.directshortestpaths_startrgtgoallft
        elif self.choice is 'startlftgoalrgt':
            directshortestpaths = self.directshortestpaths_startlftgoalrgt
        elif self.choice is 'startlftgoallft':
            directshortestpaths = self.directshortestpaths_startlftgoallft

        if len(directshortestpaths) == 0:
            print("No path found!")
            raise Exception("No path Found")

        pathnidlist = directshortestpaths[id]
        if len(previous) != 0:
            numikrlist = [previous[0]]
            jawwidth = [previous[1]]
            objmat4list = [previous[2]]
        else:
            numikrlist = [[0, self.rbt.initrgtjnts, self.rbt.initlftjnts]]
            jawwidth = [[self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen]]
            objmat4list = [self.startrotmat4]
        extendedpathnidlist = ['begin']
        print(pathnidlist)

        for i in range(len(pathnidlist) - 1):
            if i == 0 and len(pathnidlist) == 2:
                # two node path
                # they must be both rgt or both lft
                # they cannot be handover
                ## starting node
                nid = pathnidlist[i]
                gripjawwidth = self.regg.nodes[nid]['jawwidth']
                armjntsgrp = self.regg.nodes[nid]['armjnts']
                armjntsgrphanda = self.regg.nodes[nid]['armjntshanda']
                armjntsgrpworlda = self.regg.nodes[nid]['armjntsworlda']
                # choice
                if nid.startswith('startrgt'):
                    if ((type is "OC") or (type is "OO")):
                        numikrlist.append([0, armjntsgrphanda, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, jawwidth[-1][1]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, jawwidth[-1][1]])
                    numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                    numikrlist.append([0, armjntsgrpworlda, numikrlist[-1][2]])
                    jawwidth.append([gripjawwidth, jawwidth[-1][1]])
                    jawwidth.append([gripjawwidth, jawwidth[-1][1]])

                if nid.startswith('startlft'):
                    if ((type is "OC") or (type is "OO")):
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrphanda])
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                        jawwidth.append([jawwidth[-1][0], self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([jawwidth[-1][0], self.rbt.lfthnd.jawwidthopen])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrpworlda])
                    jawwidth.append([jawwidth[-1][0], gripjawwidth])
                    jawwidth.append([jawwidth[-1][0], gripjawwidth])
                objmat4handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                objmat4 = self.regg.nodes[nid]['tabletopplacementrotmat']
                objmat4worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']

                if ((type is "OC") or (type is "OO")):
                    objmat4list.append(objmat4handa)
                    objmat4list.append(objmat4)
                    extendedpathnidlist.append(nid + "x")
                    extendedpathnidlist.append(nid + "o")
                objmat4list.append(objmat4)
                objmat4list.append(objmat4worlda)
                extendedpathnidlist.append(nid + "c")
                extendedpathnidlist.append(nid + "w")
                ## goal node
                nid = pathnidlist[i + 1]
                gripjawwidth = self.regg.nodes[nid]['jawwidth']
                armjntsgrp = self.regg.nodes[nid]['armjnts']
                armjntsgrphanda = self.regg.nodes[nid]['armjntshanda']
                armjntsgrpworlda = self.regg.nodes[nid]['armjntsworlda']
                # initialize
                # choice
                if nid.startswith('goalrgt'):
                    numikrlist.append([0, armjntsgrpworlda, numikrlist[-1][2]])
                    numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                    jawwidth.append([gripjawwidth, self.rbt.rgthnd.jawwidthopen])
                    jawwidth.append([gripjawwidth, self.rbt.rgthnd.jawwidthopen])

                    if ((type is "CO") or (type is "OO")):
                        numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrphanda, numikrlist[-1][2]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])

                if nid.startswith('goallft'):
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrpworlda])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])

                if ((type is "CO") or (type is "OO")):
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrphanda])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])

                objmat4worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                objmat4 = self.regg.nodes[nid]['tabletopplacementrotmat']
                objmat4handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                objmat4list.append(objmat4worlda)
                objmat4list.append(objmat4)

                extendedpathnidlist.append(nid + "w")
                extendedpathnidlist.append(nid + "c")
                if ((type is "CO") or (type is "OO")):
                    objmat4list.append(objmat4)
                    objmat4list.append(objmat4handa)

                    extendedpathnidlist.append(nid + "o")
                    extendedpathnidlist.append(nid + "x")
            elif i == 0:
                # not two nodepath, starting node, transfer
                ## starting node
                nid = pathnidlist[i]
                gripjawwidth = self.regg.nodes[nid]['jawwidth']
                armjntsgrp = self.regg.nodes[nid]['armjnts']
                armjntsgrphanda = self.regg.nodes[nid]['armjntshanda']
                armjntsgrpworlda = self.regg.nodes[nid]['armjntsworlda']
                # choice
                if nid.startswith('startrgt'):
                    if ((type is "OC") or (type is "OO")):
                        numikrlist.append([0, armjntsgrphanda, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                    numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                    numikrlist.append([0, armjntsgrpworlda, numikrlist[-1][2]])
                    jawwidth.append([gripjawwidth, self.rbt.lfthnd.jawwidthopen])
                    jawwidth.append([gripjawwidth, self.rbt.lfthnd.jawwidthopen])
                if nid.startswith('startlft'):
                    if ((type is "OC") or (type is "OO")):
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrphanda])
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrpworlda])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])
                objmat4handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                objmat4 = self.regg.nodes[nid]['tabletopplacementrotmat']
                objmat4worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                if ((type is "OC") or (type is "OO")):
                    objmat4list.append(objmat4handa)
                    objmat4list.append(objmat4)
                    extendedpathnidlist.append(nid + "x")
                    extendedpathnidlist.append(nid + "o")
                objmat4list.append(objmat4)
                objmat4list.append(objmat4worlda)
                extendedpathnidlist.append(nid + "c")
                extendedpathnidlist.append(nid + "w")

            elif i + 1 != len(pathnidlist) - 1:
                # if handovertransit
                if self.regg.edges[pathnidlist[i], pathnidlist[i + 1]]['edgetype'] == "handovertransit":
                    nid0 = pathnidlist[i]
                    nid1 = pathnidlist[i + 1]
                    #### nid0 move to handover
                    grpjawwidth0 = self.regg.nodes[nid0]['jawwidth']
                    armjntsgrp0 = self.regg.nodes[nid0]['armjnts']
                    # initialize
                    # choice
                    if nid0.startswith('horgt'):
                        numikrlist.append([0, armjntsgrp0, numikrlist[-1][2]])
                        jawwidth.append([grpjawwidth0, self.rbt.rgthnd.jawwidthopen])
                    elif nid0.startswith('holft'):
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp0])
                        jawwidth.append([self.rbt.lfthnd.jawwidthopen, grpjawwidth0])
                    objmat40 = self.hndovrpos[self.regg.nodes[nid0]['handoverposid']]
                    objmat4list.append(objmat40)
                    extendedpathnidlist.append(nid0 + "c")
                    #### nid1 move to handover
                    grpjawwidth1 = self.regg.nodes[nid1]['jawwidth']
                    armjntsgrphanda1 = self.regg.nodes[nid1]['armjntshanda']
                    # initialize
                    armjntsgrp1 = self.regg.nodes[nid1]['armjnts']
                    # choice
                    if nid1.startswith('horgt'):
                        numikrlist.append([0, armjntsgrphanda1, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrp1, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrp1, numikrlist[-1][2]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth0])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth0])
                        jawwidth.append([grpjawwidth1, grpjawwidth0])
                    elif nid1.startswith('holft'):
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrphanda1])
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp1])
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp1])
                        jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([grpjawwidth1, grpjawwidth1])
                    objmat41 = self.hndovrpos[self.regg.nodes[nid1]['handoverposid']]
                    objmat4list.append(objmat41)
                    objmat4list.append(objmat41)
                    objmat4list.append(objmat41)
                    extendedpathnidlist.append(nid1 + "x")
                    extendedpathnidlist.append(nid1 + "o")
                    extendedpathnidlist.append(nid1 + "c")
                    #### nid0 move back
                    armjntsgrpb = self.regg.nodes[nid0]['armjntshanda']
                    # choice
                    if nid0.startswith('horgt'):
                        numikrlist.append([0, armjntsgrp0, armjntsgrp1])
                        numikrlist.append([0, armjntsgrpb, armjntsgrp1])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth1])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth1])
                    elif nid0.startswith('holft'):
                        numikrlist.append([0, armjntsgrp1, armjntsgrp0])
                        numikrlist.append([0, armjntsgrp1, armjntsgrpb])
                        jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                    objmat4b = self.hndovrpos[self.regg.nodes[nid0]['handoverposid']]
                    objmat4list.append(objmat4b)
                    objmat4list.append(objmat4b)
                    extendedpathnidlist.append(nid0 + "o")
                    extendedpathnidlist.append(nid0 + "x")
                else:
                    print(self.regg.edges[pathnidlist[i], pathnidlist[i + 1]]['edgetype'])
                    # not two node path, middle nodes, if transfer
                    ## middle first
                    nid = pathnidlist[i]
                    if nid.startswith('ho'):
                        pass
                    ## middle second
                    nid = pathnidlist[i + 1]
                    # could be ho
                    if nid.startswith('ho'):
                        pass
            else:

                ## last node
                nid = pathnidlist[i + 1]
                gripjawwidth = self.regg.nodes[nid]['jawwidth']
                armjntsgrp = self.regg.nodes[nid]['armjnts']
                armjntsgrphanda = self.regg.nodes[nid]['armjntshanda']
                armjntsgrpworlda = self.regg.nodes[nid]['armjntsworlda']
                # initialize
                # choice
                if nid.startswith('goalrgt'):
                    numikrlist.append([0, armjntsgrpworlda, numikrlist[-1][2]])
                    numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                    jawwidth.append([gripjawwidth, self.rbt.lfthnd.jawwidthopen])
                    jawwidth.append([gripjawwidth, self.rbt.lfthnd.jawwidthopen])
                    if ((type == "CO") or (type == "OO")):
                        numikrlist.append([0, armjntsgrp, numikrlist[-1][2]])
                        numikrlist.append([0, armjntsgrphanda, numikrlist[-1][2]])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                if nid.startswith('goallft'):
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrpworlda])
                    numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, gripjawwidth])
                    if ((type == "CO") or (type == "OO")):
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrp])
                        numikrlist.append([0, numikrlist[-1][1], armjntsgrphanda])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, self.rbt.lfthnd.jawwidthopen])
                objmat4worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                objmat4 = self.regg.nodes[nid]['tabletopplacementrotmat']
                objmat4handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                objmat4list.append(objmat4worlda)
                objmat4list.append(objmat4)
                extendedpathnidlist.append(nid + "w")
                extendedpathnidlist.append(nid + "c")
                if ((type == "CO") or (type == "OO")):
                    objmat4list.append(objmat4)
                    objmat4list.append(objmat4handa)
                    extendedpathnidlist.append(nid + "o")
                    extendedpathnidlist.append(nid + "x")

                if self.gobacktoinitafterplanning == True:
                    # # pre-place
                    nid = pathnidlist[i + 1]
                    # initilize
                    grpjawwidth1 = self.regg.nodes[nid]['jawwidth']
                    armjntsgrp1 = self.regg.nodes[nid]['armjnts']
                    objmat4b = self.hndovrpos[self.regg.nodes[nid0]['handoverposid']]
                    # move back to init pose
                    if nid.startswith('goalrgt'):
                        numikrlist.append([self.rbt.initlftjntsr[0], armjntsgrp1, self.rbt.initlftjntsr[1:]])
                        jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                    elif nid.startswith('goallft'):
                        numikrlist.append([self.rbt.initrgtjntsr[0], self.rbt.initrgtjntsr[1:], armjntsgrp1])
                        jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth1])
                    objmat4list.append(objmat4b)
                    fnid = 'goallft' + nid[nid.index('t') + 1:] if nid.startswith('goalrgt') else 'goalrgt' + nid[
                                                                                                              nid.index(
                                                                                                                  't') + 1:]
                    extendedpathnidlist.append(fnid + "i")
        extendedpathnidlist.append('end')

        return [objmat4list, numikrlist, jawwidth, extendedpathnidlist, pathnidlist]

    def removeBadNodes(self, nodelist):
        """
        remove the invalidated nodes to prepare for a new plan

        :param nodelist: a list of invalidated nodes
        :return:

        author: weiwei
        date: 20170920
        """

        print("Removing nodes ", nodelist)
        self.regg.remove_nodes_from(nodelist)
        for node in nodelist:
            if node.startswith('startrgt'):
                try:
                    self.startrgtnodeids.remove(node)
                except KeyError:
                    pass
            if node.startswith('startlft'):
                try:
                    self.startlftnodeids.remove(node)
                except KeyError:
                    pass
            if node.startswith('goalrgt'):
                try:
                    self.goalrgtnodeids.remove(node)
                except KeyError:
                    pass
            if node.startswith('goallft'):
                try:
                    self.goallftnodeids.remove(node)
                except KeyError:
                    pass

    def removeBadEdge(self, node0, node1):
        """
        remove an invalidated edge to prepare for a new plan

        :param node0, node1 two ends of an edge
        :return:

        author: weiwei
        date: 20190423
        """
        if node0 == node1:
            return
        print("Removing edge ", node0, node1)
        self.regg.remove_edge(node0, node1)

    def planRegrasp(self, objcm, obstaclecmlist=None, id=0, switch="OC", previous=[], end=False,
                    togglemp=True):
        """
        plan the regrasp sequences

        :param objpath:
        :param robot:
        :param hand:
        :param dbase:
        :param obstaclecmlist:
        :param id = 0
        :param switch in "OC" open-close "CC" close-close
        :param previous: set it to [] if the motion is not a continuing one, or else, set it to [lastikr, lastjawwidth,lastobjmat4]
        :param end: set it to True if it is the last one
        :param togglemp denotes whether the motion between the keyposes are planned or not, True by default
        :return:

        author: weiwei
        date: 20180924
        """

        robot = self.rbt
        cdchecker = self.cdchecker
        if obstaclecmlist == None:
            obstaclecmlist = self.obstaclecmlist
        else:
            obstaclecmlist = obstaclecmlist

        while True:
            print("--------------new search---------------")
            self.updateshortestpath()
            print("I Get Stuck updateshrtestpath")
            [objms, numikrms, jawwidth, pathnidlist, originalpathnidlist] = \
                self.getMotionSequence(id=id, type=switch, previous=previous)
            print("I Get Stuck getMotionSequence")
            if objms == None:
                return [None, None, None, None]
            bcdfree = True
            for i in range(len(numikrms)):
                rgtarmjnts = numikrms[i][1].tolist()
                lftarmjnts = numikrms[i][2].tolist()
                robot.movealljnts([numikrms[i][0], 0, 0] + rgtarmjnts + lftarmjnts)
                # skip the exact handover pose and only detect the cd between armhnd and body
                if pathnidlist[i].startswith('ho') and pathnidlist[i + 1].startswith('ho'):
                    abcd = cdchecker.isCollidedHO(robot, obstaclecmlist)
                    if abcd:
                        if self.inspector is not None:
                            self.inspector.add_error("handover collision",
                                                     Error_info(name=f"{pathnidlist[i]} -- {pathnidlist[i + 1]}",
                                                                objmat=objms[i], lftarmjnts=lftarmjnts,
                                                                rgtarmjnts=rgtarmjnts, lftjawwidth=jawwidth[i][0],
                                                                rgtjawwidth=jawwidth[i][1]), )
                        self.removeBadNodes([pathnidlist[i][:-1]])
                        print("Abcd collided at ho pose")
                        bcdfree = False
                        break
                else:
                    # NOTE: we ignore both arms here for conciseness
                    # This might be a potential bug
                    if cdchecker.isCollided(robot, obstaclecmlist, holdarmname="all"):
                        if self.inspector is not None:
                            self.inspector.add_error("non-ho pose collision",
                                                     Error_info(name=pathnidlist[i],
                                                                objmat=objms[i], lftarmjnts=lftarmjnts,
                                                                rgtarmjnts=rgtarmjnts, lftjawwidth=jawwidth[i][0],
                                                                rgtjawwidth=jawwidth[i][1]), )
                        self.removeBadNodes([pathnidlist[i][:-1]])
                        print("Robot collided at non-ho pose")
                        bcdfree = False
                        break
            robot.goinitpose()
            if bcdfree:
                objmsmp = []
                numikrmsmp = []
                jawwidthmp = []
                print(pathnidlist)
                if not togglemp:
                    for i, numikrm in enumerate(numikrms):
                        if i > 0:
                            startid = pathnidlist[i - 1]
                            endid = pathnidlist[i]
                            if (not end) and (endid is 'end'):
                                continue
                            if (len(previous) > 0) and (startid is 'begin'):
                                continue
                            numikrmsmp.append([numikrms[i - 1], numikrms[i]])
                            objmsmp.append([objms[i - 1], objms[i]])
                            jawwidthmp.append([jawwidth[i - 1], jawwidth[i]])
                    return objmsmp, numikrmsmp, jawwidthmp, originalpathnidlist

                # INNERLOOP motion planning
                smoother = sm.Smoother()
                ctcallback = ck.Checker(robot, cdchecker)
                breakflag = False
                for i, numikrm in enumerate(numikrms):
                    if i > 0:
                        # determine which arm to plan
                        # assume right
                        # assume redundant planning
                        robot.goinitpose()
                        startid = pathnidlist[i - 1]
                        endid = pathnidlist[i]
                        objmat = objms[i - 1]
                        objrot = objmat[:3, :3]
                        objpos = objmat[:3, 3]
                        if (not end) and (endid is 'end'):
                            continue
                        if (len(previous) > 0) and (startid is 'begin'):
                            continue
                        if (startid[-1] == "o" and endid[-1] == "c") or (startid[-1] == "c" and endid[-1] == "o"):
                            # open and close gripper
                            print("O/C hands, simply include ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            numikrmsmp.append([numikrms[i - 1], numikrms[i]])
                            objmsmp.append([objms[i - 1], objms[i]])
                            jawwidthmp.append([jawwidth[i - 1], jawwidth[i]])
                            continue
                        if (startid[:-1] == endid[:-1]):  # move to handover pose or linear interpolation
                            if (startid[-1] != "i") and (endid[-1] != "i"):
                                # linear interpolation
                                tempnumikrmsmp = []
                                tempjawwidthmp = []
                                tempobjmsmp = []
                                temparmname = "rgt"
                                startjntags = numikrms[i - 1][1].tolist()
                                goaljntags = numikrms[i][1].tolist()
                                if "lft" in startid:
                                    temparmname = "lft"
                                    startjntags = numikrms[i - 1][2].tolist()
                                    goaljntags = numikrms[i][2].tolist()
                                # TODO there is about 0.1 mm error in the final position
                                [interplatedjnts, interplatedobjposes] = \
                                    ctcallback.isLMAvailableJNTwithObj(startjntags, goaljntags,
                                                                       [objpos, objrot], armname=temparmname,
                                                                       type=startid[-1])
                                if len(interplatedjnts) == 0:
                                    print("Failed to interpolate motion primitive! restarting...")
                                    # always a single hand
                                    if self.inspector is not None:
                                        self.inspector. \
                                            add_error("interplation error",
                                                      Error_info(name=f"{startid}-{endid}", objmat=None,
                                                                 lftarmjnts=startjntags if temparmname == "lft" else
                                                                 numikrms[i - 1][2],
                                                                 rgtarmjnts=startjntags if temparmname == "rgt" else
                                                                 numikrms[i - 1][1],
                                                                 lftjawwidth=jawwidth[i - 1][0],
                                                                 rgtjawwidth=jawwidth[i - 1][1]), )
                                        self.inspector. \
                                            add_error("interplation error",
                                                      Error_info(name=f"{startid}-{endid}", objmat=None,
                                                                 lftarmjnts=goaljntags if temparmname == "lft" else
                                                                 numikrms[i - 1][2],
                                                                 rgtarmjnts=goaljntags if temparmname == "rgt" else
                                                                 numikrms[i - 1][1],
                                                                 lftjawwidth=jawwidth[i - 1][0],
                                                                 rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.removeBadNodes([pathnidlist[i - 1][:-1]])
                                    breakflag = True
                                    break
                                print("Motion primitives, interplate ", pathnidlist[i - 1], " and ", pathnidlist[i])
                                for eachitem in interplatedjnts:
                                    if temparmname == "rgt":
                                        tempnumikrmsmp.append(
                                            [numikrms[i - 1][0], np.array(eachitem), numikrms[i - 1][2]])
                                    else:
                                        tempnumikrmsmp.append(
                                            [numikrms[i - 1][0], numikrms[i - 1][1], np.array(eachitem)])
                                    tempjawwidthmp.append(jawwidth[i - 1])
                                for eachitem in interplatedobjposes:
                                    tempobjmsmp.append(rm.homobuild(eachitem[0], eachitem[1]))
                                numikrmsmp.append(tempnumikrmsmp)
                                jawwidthmp.append(tempjawwidthmp)
                                objmsmp.append(tempobjmsmp)
                                # update the keypose to avoid non-continuous linear motion: numikrms and objms
                                if temparmname == "rgt":
                                    numikrms[i][1] = tempnumikrmsmp[-1][1]
                                elif temparmname == "lft":
                                    numikrms[i][2] = tempnumikrmsmp[-1][2]
                                objms[i] = tempobjmsmp[-1]
                                continue
                        # init robot pose
                        rgtarmjnts = numikrms[i - 1][1].tolist()
                        lftarmjnts = numikrms[i - 1][2].tolist()
                        robot.movealljnts([numikrms[i - 1][0], 0, 0] + rgtarmjnts + lftarmjnts)
                        # assume rgt
                        armname = 'rgt'
                        start = numikrms[i - 1][1].tolist()
                        goal = numikrms[i][1].tolist()
                        startjawwidth = jawwidth[i - 1][0]
                        if "lft" in endid:
                            armname = 'lft'
                            start = numikrms[i - 1][2].tolist()
                            goal = numikrms[i][2].tolist()
                            startjawwidth = jawwidth[i - 1][1]
                        starttreesamplerate = 25
                        endtreesamplerate = 30
                        print(armname)
                        print(startjawwidth)
                        ctcallback.setarmname(armname)
                        planner = ddrrtc.DDRRTConnect(start=start, goal=goal, ctcallback=ctcallback,
                                                      starttreesamplerate=starttreesamplerate,
                                                      endtreesamplerate=endtreesamplerate, expanddis=20,
                                                      maxiter=200, maxtime=7.0)
                        tempnumikrmsmp = []
                        tempjawwidthmp = []
                        tempobjmsmp = []
                        if (endid[-1] == "c") or (endid[-1] == "w"):
                            print("Planning hold motion between ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            relpos, relrot = robot.getinhandpose(objpos, objrot, armname)

                            path, sampledpoints = planner.planninghold([objcm], [[relpos, relrot]], obstaclecmlist)
                            if path is False:
                                print("Motion planning with hold failed! restarting...")

                                # TODO remove bad edge?
                                # regrip.removeBadNodes([pathnidlist[i-1][:-1]])
                                self.removeBadNodes([pathnidlist[i][:-1]])
                                if self.inspector is not None:
                                    # inspector
                                    self.inspector.add_error("rrt-hold error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=None,
                                                                        lftarmjnts=start if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=start if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.inspector.add_error("rrt-hold error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=None,
                                                                        lftarmjnts=goal if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=goal if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                breakflag = True
                                break
                            path = smoother.pathsmoothinghold(path, planner, 30)
                            npath = len(path)
                            for j in range(npath):
                                if armname == 'rgt':
                                    tempnumikrmsmp.append([0.0, np.array(path[j]), numikrms[i - 1][2]])
                                else:
                                    tempnumikrmsmp.append([0.0, numikrms[i - 1][1], np.array(path[j])])
                                robot.movearmfk(np.array(path[j]), armname=armname)
                                tempjawwidthmp.append(jawwidth[i - 1])
                                objpos, objrot = robot.getworldpose(relpos, relrot, armname)
                                tempobjmsmp.append(rm.homobuild(objpos, objrot))
                        else:
                            # if the arm is not holding an object, the object will be treated as an obstacle
                            print("Planning motion ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            objcmcopy = copy.deepcopy(objcm)
                            objcmcopy.sethomomat(objms[i - 1])
                            obstaclecmlistnew = obstaclecmlist + [objcmcopy]

                            path, sampledpoints = planner.planning(obstaclecmlistnew)
                            if path is False:
                                print("Motion planning failed! restarting...")
                                if pathnidlist[i - 1] == "begin":
                                    self.removeBadNodes([pathnidlist[i][:-1]])
                                    breakflag = True
                                    break
                                if pathnidlist[i] == "end":
                                    self.removeBadNodes([pathnidlist[i - 1][:-1]])
                                    breakflag = True
                                    break
                                node0 = pathnidlist[i - 1][:-1]
                                node1 = pathnidlist[i][:-1]
                                self.removeBadEdge(node0, node1)
                                if self.inspector is not None:
                                    # inspector
                                    self.inspector.add_error("rrt error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=objms[i - 1],
                                                                        lftarmjnts=start if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=start if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.inspector.add_error("rrt error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=objms[i - 1],
                                                                        lftarmjnts=goal if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=goal if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )
                                breakflag = True
                                break
                            path = smoother.pathsmoothing(path, planner, 30)
                            npath = len(path)
                            for j in range(npath):
                                if armname == 'rgt':
                                    tempnumikrmsmp.append([0.0, np.array(path[j]), numikrms[i - 1][2]])
                                else:
                                    tempnumikrmsmp.append([0.0, numikrms[i - 1][1], np.array(path[j])])
                                tempjawwidthmp.append(jawwidth[i - 1])
                                tempobjmsmp.append(objms[i - 1])
                        numikrmsmp.append(tempnumikrmsmp)
                        jawwidthmp.append(tempjawwidthmp)
                        objmsmp.append(tempobjmsmp)
                print(i, len(numikrms) - 1)
                if breakflag is False:
                    # successfully finished!
                    return [objmsmp, numikrmsmp, jawwidthmp, originalpathnidlist]
                else:
                    # remov node and start new search
                    continue

    def plotgraph(self, pltfig):
        """
        plot the graph without start and goal

        :param pltfig: the matplotlib object
        :return:

        author: weiwei
        date: 20161217, sapporos
        """

        def add(num1, num2):
            try:
                total = float(num1) + float(num2)
            except ValueError:
                return None
            else:
                return total

        # biggest circle: grips; big circle: rotation; small circle: placements
        radiusplacement = 30
        radiusrot = 6
        radiusgrip = 1
        xyplacementspos = {}
        xydiscreterotspos = {}
        self.xyzglobalgrippos = {}
        self.fttpsids = []
        for i, ttpsid in enumerate(self.fttpsids):
            xydiscreterotspos[ttpsid] = {}
            self.xyzglobalgrippos[ttpsid] = {}
            xypos = [radiusplacement * math.cos(2 * math.pi / self.nfttps * i),
                     radiusplacement * math.sin(2 * math.pi / self.nfttps * i)]
            xyplacementspos[ttpsid] = xypos
            for j, anglevalue in enumerate(self.angles):
                self.xyzglobalgrippos[ttpsid][anglevalue] = {}
                xypos = [radiusrot * math.cos(math.radians(anglevalue)), radiusrot * math.sin(math.radians(anglevalue))]
                xydiscreterotspos[ttpsid][anglevalue] = \
                    [xyplacementspos[ttpsid][0] + xypos[0], xyplacementspos[ttpsid][1] + xypos[1]]
                for k, globalgripid in enumerate(self.globalgripids):
                    xypos = [radiusgrip * math.cos(2 * math.pi / len(self.globalgripids) * k),
                             radiusgrip * math.sin(2 * math.pi / len(self.globalgripids) * k)]
                    self.xyzglobalgrippos[ttpsid][anglevalue][globalgripid] = \
                        [xydiscreterotspos[ttpsid][anglevalue][0] + xypos[0],
                         xydiscreterotspos[ttpsid][anglevalue][1] + xypos[1], 0]

        # for start and goal grasps poses:

        self.xyzglobalgrippos_startgoal = {}
        for k, globalgripid in enumerate(self.grasp):
            xypos = [radiusgrip * math.cos(2 * math.pi / len(self.grasp) * k),
                     radiusgrip * math.sin(2 * math.pi / len(self.grasp) * k)]
            self.xyzglobalgrippos_startgoal[k] = [xypos[0], xypos[1], 0]

        # self.grasp, self.gridsfloatingposemat4np, self.fpgpairlist, self.fpglist, \
        # self.IKfeasibleHndover_rgt, self.IKfeasibleHndover_lft, self.jnts_rgt, self.jnts_lft

        # for handover
        nfp = len(self.fpsnpmat4)
        xdist = 10
        x = range(300, 501, xdist)
        y = range(-50, 50, int(100 * xdist / nfp))

        transitedges = []
        transferedges = []
        hotransitedges = []
        hotransferedges = []
        startrgttransferedges = []
        startlfttransferedges = []
        goalrgttransferedges = []
        goallfttransferedges = []
        startgoalrgttransferedges = []
        startgoallfttransferedges = []
        startrgttransitedges = []
        goalrgttransitedges = []
        startlfttransitedges = []
        goallfttransitedges = []
        counter = 0
        for nid0, nid1, reggedgedata in self.regg.edges(data=True):
            counter = counter + 1
            if counter > 100000:
                break
            xyzpos0 = [0, 0, 0]
            xyzpos1 = [0, 0, 0]
            if (reggedgedata['edgetype'] is 'transit') or (reggedgedata['edgetype'] is 'transfer'):
                if nid0.startswith('ho'):
                    fpind0 = int(nid0[nid0.index("s") + 1:])
                    fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                    nfpgp = len(self.feasiblefpgpairlist[fpind0])
                    xpos = x[fpind0 % len(x)]
                    ypos = y[int(fpind0 / len(x))]
                    xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                    if nid0.startswith('horgt'):
                        xyzpos0[1] = xyzpos0[1] - 100
                    if nid0.startswith('holft'):
                        xyzpos0[1] = xyzpos0[1] + 100
                else:
                    fttpid0 = self.regg.node[nid0]['freetabletopplacementid']
                    anglevalue0 = self.regg.node[nid0]['angle']
                    ggid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = list(map(add, self.xyzglobalgrippos[fttpid0][anglevalue0][ggid0],
                                       [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                    if nid0.startswith('rgt'):
                        xyzpos0[1] = xyzpos0[1] - 800
                    if nid0.startswith('lft'):
                        xyzpos0[1] = xyzpos0[1] + 800
                if nid1.startswith('ho'):
                    fpind1 = int(nid1[nid1.index("s") + 1:])
                    fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                    nfpgp = len(self.feasiblefpgpairlist[fpind1])
                    xpos = x[fpind1 % len(x)]
                    ypos = y[int(fpind1 / len(x))]
                    xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                    if nid1.startswith('horgt'):
                        xyzpos1[1] = xyzpos1[1] - 100
                    if nid1.startswith('holft'):
                        xyzpos1[1] = xyzpos1[1] + 100
                else:
                    fttpid1 = self.regg.node[nid1]['freetabletopplacementid']
                    anglevalue1 = self.regg.node[nid1]['angle']
                    ggid1 = self.regg.node[nid1]['globalgripid']
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = map(add, self.xyzglobalgrippos[fttpid1][anglevalue1][ggid1],
                                  [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]])
                    if nid1.startswith('rgt'):
                        xyzpos1[1] = xyzpos1[1] - 800
                    if nid1.startswith('lft'):
                        xyzpos1[1] = xyzpos1[1] + 800
                # 3d
                # if reggedgedata['edgetype'] is 'transit':
                #     transitedges.append([xyzpos0, xyzpos1])
                # if reggedgedata['edgetype'] is 'transfer':
                #     transferedges.append([xyzpos0, xyzpos1])
                # 2d
                # move the basic graph to x+600
                xyzpos0[0] = xyzpos0[0] + 600
                xyzpos1[0] = xyzpos1[0] + 600
                if reggedgedata['edgetype'] is 'transit':
                    transitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] is 'transfer':
                    if nid0.startswith('ho') or nid1.startswith('ho'):
                        hotransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                    else:
                        transferedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif (reggedgedata['edgetype'] is 'handovertransit'):
                fpind0 = int(nid0[nid0.index("s") + 1:])
                fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                nfpgp = len(self.fpsnpmat4[fpind0])
                xpos = x[int(fpind0 % len(x))]
                ypos = y[int(fpind0 / len(x))]
                xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                           radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                if nid0.startswith('horgt'):
                    xyzpos0[1] = xyzpos0[1] - 100
                if nid0.startswith('holft'):
                    xyzpos0[1] = xyzpos0[1] + 100
                fpind1 = int(nid1[nid1.index("s") + 1:])
                fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                nfpgp = len(self.fpsnpmat4[fpind1])
                xpos = x[int(fpind1 % len(x))]
                ypos = y[int(fpind1 / len(x))]
                xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                           radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                if nid1.startswith('horgt'):
                    xyzpos1[1] = xyzpos1[1] - 100
                if nid1.startswith('holft'):
                    xyzpos1[1] = xyzpos1[1] + 100
                # move the basic graph to x+600
                xyzpos0[0] = xyzpos0[0] + 600
                xyzpos1[0] = xyzpos1[0] + 600
                hotransitedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif reggedgedata['edgetype'].endswith('transit'):
                gid0 = int(nid0[-nid0[::-1].index("t"):])
                gid1 = int(nid1[-nid1[::-1].index("t"):])
                tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                xyzpos0 = list(map(add, self.xyzglobalgrippos_startgoal[gid0],
                                   [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                xyzpos1 = list(map(add, self.xyzglobalgrippos_startgoal[gid1],
                                   [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                if reggedgedata['edgetype'] is 'startrgttransit':
                    startrgttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] is 'goalrgttransit':
                    goalrgttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] is 'startlfttransit':
                    startlfttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] is 'goallfttransit':
                    goallfttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif reggedgedata['edgetype'].endswith('transfer'):
                if nid0.startswith('ho'):
                    fpind0 = int(nid0[nid0.index("s") + 1:])
                    fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                    nfpgp = len(self.fpsnpmat4[fpind0])
                    xpos = x[int(fpind0 % len(x))]
                    ypos = y[int(fpind0 / len(x))]
                    xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                    if nid0.startswith('horgt'):
                        xyzpos0[1] = xyzpos0[1] - 100
                    if nid0.startswith('holft'):
                        xyzpos0[1] = xyzpos0[1] + 100
                    xyzpos0[0] = xyzpos0[0] + 600
                elif nid0.startswith('rgt') or nid0.startswith('lft'):
                    fttpid0 = self.regg.node[nid0]['freetabletopplacementid']
                    anglevalue0 = self.regg.node[nid0]['angle']
                    ggid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = map(add, self.xyzglobalgrippos[fttpid0][anglevalue0][ggid0],
                                  [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]])
                    if nid0.startswith('rgt'):
                        xyzpos0[1] = xyzpos0[1] - 800
                    if nid0.startswith('lft'):
                        xyzpos0[1] = xyzpos0[1] + 800
                    xyzpos0[0] = xyzpos0[0] + 600
                else:
                    gid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = list(map(add, self.xyzglobalgrippos_startgoal[gid0],
                                       [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                if nid1.startswith('ho'):
                    fpind1 = int(nid1[nid1.index("s") + 1:])
                    fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                    nfpgp = len(self.fpsnpmat4[fpind1])
                    xpos = x[int(fpind1 % len(x))]
                    ypos = y[int(fpind1 / len(x))]
                    xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                    if nid1.startswith('horgt'):
                        xyzpos1[1] = xyzpos1[1] - 100
                    if nid1.startswith('holft'):
                        xyzpos1[1] = xyzpos1[1] + 100
                    xyzpos1[0] = xyzpos1[0] + 600
                elif nid1.startswith('lft') or nid1.startswith('rgt'):
                    fttpid1 = self.regg.node[nid1]['freetabletopplacementid']
                    anglevalue1 = self.regg.node[nid1]['angle']
                    ggid1 = self.regg.node[nid1]['globalgripid']
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = list(map(add, self.xyzglobalgrippos[fttpid1][anglevalue1][ggid1],
                                       [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                    if nid1.startswith('rgt'):
                        xyzpos1[1] = xyzpos1[1] - 800
                    if nid1.startswith('lft'):
                        xyzpos1[1] = xyzpos1[1] + 800
                    xyzpos1[0] = xyzpos1[0] + 600
                else:
                    ggid1 = int(nid1[-nid1[::-1].index("t"):])
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = list(map(add, self.xyzglobalgrippos_startgoal[ggid1],
                                       [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                if reggedgedata['edgetype'].startswith('startgoalrgt'):
                    startgoalrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startgoallft'):
                    startgoallfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startrgt'):
                    startrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startlft'):
                    startlfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('goalrgt'):
                    goalrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('goallft'):
                    goallfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])

            # self.gnodesplotpos[nid0] = xyzpos0[:2]
            # self.gnodesplotpos[nid1] = xyzpos1[:2]
        # 3d
        # transitec = mc3d.Line3DCollection(transitedges, colors=[0,1,1,1], linewidths=1)
        # transferec = mc3d.Line3DCollection(transferedges, colors=[0,0,0,.1], linewidths=1)
        # 2d
        transitec = mc.LineCollection(transitedges, colors=[0, 1, 1, 1], linewidths=1)
        transferec = mc.LineCollection(transferedges, colors=[0, 0, 0, .1], linewidths=1)
        hotransitec = mc.LineCollection(hotransitedges, colors=[1, 0, 1, .1], linewidths=1)
        hotransferec = mc.LineCollection(hotransferedges, colors=[.5, .5, 0, .03], linewidths=1)
        # transfer
        startrgttransferec = mc.LineCollection(startrgttransferedges, colors=[.7, 0, 0, .3], linewidths=1)
        startlfttransferec = mc.LineCollection(startlfttransferedges, colors=[.3, 0, 0, .3], linewidths=1)
        goalrgttransferec = mc.LineCollection(goalrgttransferedges, colors=[0, 0, .7, .3], linewidths=1)
        goallfttransferec = mc.LineCollection(goallfttransferedges, colors=[0, 0, .3, .3], linewidths=1)
        startgoalrgttransferec = mc.LineCollection(startgoalrgttransferedges, colors=[0, 0, .7, .3], linewidths=1)
        startgoallfttransferec = mc.LineCollection(startgoallfttransferedges, colors=[0, 0, .3, .3], linewidths=1)
        # transit
        startrgttransitec = mc.LineCollection(startrgttransitedges, colors=[0, .5, 1, .3], linewidths=1)
        startlfttransitec = mc.LineCollection(startlfttransitedges, colors=[0, .2, .4, .3], linewidths=1)
        goalrgttransitec = mc.LineCollection(goalrgttransitedges, colors=[0, .5, 1, .3], linewidths=1)
        goallfttransitec = mc.LineCollection(goallfttransitedges, colors=[0, .2, .4, .3], linewidths=1)

        ax = pltfig.add_subplot(111)
        ax.add_collection(transferec)
        ax.add_collection(transitec)
        ax.add_collection(hotransferec)
        ax.add_collection(hotransitec)
        ax.add_collection(startrgttransferec)
        ax.add_collection(startlfttransferec)
        ax.add_collection(goalrgttransferec)
        ax.add_collection(goallfttransferec)
        ax.add_collection(startgoalrgttransferec)
        ax.add_collection(startgoallfttransferec)

    # def plausiblegrasp(self, objcmlist):
    #     for grasp in self.grasp:
    #         self.hmstr.hanfa
