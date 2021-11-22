import time
import os
import numpy as np
import robot_con.ur.program_builder as pb
import basis.robot_math as rm
import time
import pickle
import config
import socket
import struct
import math
import motion.trajectory.piecewisepoly as pwp

import matplotlib.pyplot as plt

class ForceController(object):
    def __init__(self, rbt, rbtx, armname="lft"):
        self.rbtx = rbtx
        self.rbt = rbt
        self.armname = armname
        self.arm = self.rbtx.rgt_arm_hnd.arm if self.armname == "rgt" else self.rbtx.lft_arm_hnd.arm
        self.__scriptpath = os.path.dirname(__file__)
        self.__programbuilder = pb.ProgramBuilder()
        self.__rrx = 90
        self.__rrz = 90
        self.__jointscaler = 1000000
        self.__timestep = 0.005
        self.pwp = pwp.PiecewisePoly('linear')

    def write_force(self, force_log):
        time.sleep(.5)
        print("rbt is running", self.arm.is_program_running())
        while self.arm.is_program_running():
            with open(force_log, "wb") as f:
                f.writelines([self.arm.get_tcp_force(), time.time()])
            print("force:", self.arm.get_tcp_force(), time.time())

    def get_force(self):
        force = None
        # print("rbt is running", self.arm.is_program_running())
        try:
            force = self.arm.get_tcp_force()
            print(force)
        except:
            print("fail to get force")

        return force

    def linearandspiralsearch(self, direction=np.array([0, 0, -1]), forcethreshold=8, armname="rgt"):

        jntangles = self.rbtx.getjnts(armname)
        self.rbt.movearmfk(jntangles, armname=armname)

        # get tcp
        pos, rot = self.rbt.gettcp(armname)
        vector = direction

        # the vector in the tip coordinate
        vector_tip = np.dot(np.linalg.inv(rot), vector)
        vector_tip = vector_tip / np.linalg.norm(vector_tip)

        print("The value of vector_tip is ", vector_tip)

        # the vector z-axis of tcp in the frame of tcp coodinate [0;0;1]
        vector_z = np.array([0, 0, 1])

        # the rot-axis from z-axis to vector in the frame of obj coodinate
        # the former one is target, with right hand rule
        vector_rotaxis = np.cross(vector, np.array([0, 0, 1]))

        if np.linalg.norm(vector_rotaxis) == 0:
            vector_rotaxis = np.array([0, 0, 1])

        # angle between vector_z and vector_tip
        vector_ztip = vector_z[0] * vector_tip[0] + vector_z[1] * vector_tip[1] + vector_z[2] * vector_tip[2]
        vector_ztip_norm = np.linalg.norm(vector_z) * np.linalg.norm(vector_tip)
        rad = np.arccos(vector_ztip / vector_ztip_norm)
        angle = rad * 180 / np.pi

        rotationmatrix = rm.rotmat_from_axangle(vector_rotaxis, angle)

        self.__programbuilder.load_prog(self.__scriptpath + "/urscripts/spiralsearch.script")
        prog = self.__programbuilder.get_program_to_run()
        prog = prog.replace("parameter_direction",
                            "[%f,%f,%f]" % (vector_tip[0], vector_tip[1], vector_tip[2]))
        prog = prog.replace("parameter_forcethreshold",
                            "%f" % forcethreshold)
        prog = prog.replace("parameter_rotationmatrix", "[%f,%f,%f,%f,%f,%f,%f,%f,%f]" % (
            rotationmatrix[0][0], rotationmatrix[0][1], rotationmatrix[0][2],
            rotationmatrix[1][0], rotationmatrix[1][1], rotationmatrix[1][2],
            rotationmatrix[2][0], rotationmatrix[2][1], rotationmatrix[2][2]
        ))
        self.arm.send_program(prog)
        time.sleep(.5)
        while self.arm.is_program_running():
            pass

    def impedance_control(self, toolframe, direction=np.array([0, 0, -1]), distance=0.01, force=7):
        self.__programbuilder.load_prog(self.__scriptpath + "/urscripts/impctl.script")
        prog = self.__programbuilder.get_program_to_run()
        # direction p[x,x,x,x,x,x]
        prog = prog.replace("parameter_toolframe",
                            f"p[{toolframe[0]},{toolframe[1]},{toolframe[2]},0,0,0]")
        prog = prog.replace("parameter_distance",
                            f"{distance}")
        prog = prog.replace("parameter_force",
                            f"[{direction[0] * force},{direction[1] * force},{direction[2] * force},0,0,0]")
        self.arm.send_program(prog)
        time.sleep(.5)
        while self.arm.is_program_running():
            time.sleep(1)
        time.sleep(.5)

    def attachfirm(self, direction=np.array([0, 0, -1]), forcethreshold=10):
        print(self.arm.is_running())
        armjnts = self.rbtx.getjnts(self.armname)
        self.rbt.fk(armjnts, armname=self.armname)
        tcppos = self.rbt.gettcp(armname=self.armname)[1]

        # the vector z-axis of obj in the world coordinate [:3,2]
        vector = direction

        # the vector in the tip coordinate
        vector_tip = np.dot(np.linalg.inv(tcppos), vector)
        vector_tip = vector_tip / np.linalg.norm(vector_tip)
        self.__programbuilder.load_prog(self.__scriptpath + "/urscripts/attachfirm.script")
        prog = self.__programbuilder.get_program_to_run()
        prog = prog.replace("parameter_direction",
                            "[%f,%f,%f]" % (vector_tip[0], vector_tip[1], vector_tip[2]))
        prog = prog.replace("parameter_forcethreshold",
                            "%f" % forcethreshold)
        print("The value of vector_tip is ", vector_tip)
        self.arm.send_program(prog)
        time.sleep(.5)
        while self.arm.is_program_running():
            pass
        time.sleep(1)
        return vector_tip

    def passive_move(self, jointspath, toolrelpose, timepathstep=1.0, inpfunc="cubic"):
        urx_urmdsocket_ipad = (config.IPURX, 60011)
        urx_urmdsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        urx_urmdsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        urx_urmdsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        urx_urmdsocket.bind(urx_urmdsocket_ipad)
        urx_urmdsocket.listen(5)
        jointspath = np.radians(np.asarray(jointspath)).tolist()
        jointsradlisttimestep, _, _, x = self.pwp.interpolate_by_max_jntspeed(path=jointspath, control_frequency=0.002)
        print(len(jointsradlisttimestep))
        # plt.plot(x, jointsradlisttimestep, 'o')
        # plt.show()
        # return jointsradlisttimestep
        # def cubic(t, timestamp, q0array, v0array, q1array, v1array):
        #     a0 = q0array
        #     a1 = v0array
        #     a2 = (-3 * (q0array - q1array) - (2 * v0array + v1array) * timestamp) / (timestamp ** 2)
        #     a3 = (2 * (q0array - q1array) + (v0array + v1array) * timestamp) / (timestamp ** 3)
        #     qt = a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)
        #     vt = a1 + 2 * a2 * t + 3 * a3 * (t ** 2)
        #     return qt.tolist(), vt.tolist()
        #
        # def quintic(t, timestamp, q0array, v0array,
        #             q1array, v1array, a0array=np.array([0.0] * 6), a1array=np.array([0.0] * 6)):
        #     a0 = q0array
        #     a1 = v0array
        #     a2 = a0array / 2.0
        #     a3 = (20 * (q1array - q0array) - (8 * v1array + 12 * v0array) * timestamp -
        #           (3 * a1array - a0array) * (timestamp ** 2)) / (2 * (timestamp ** 3))
        #     a4 = (30 * (q0array - q1array) + (14 * v1array + 16 * v0array) * timestamp +
        #           (3 * a1array - 2 * a0array) * (timestamp ** 2)) / (2 * (timestamp ** 4))
        #     a5 = (12 * (q1array - q0array) - 6 * (v1array + v0array) * timestamp -
        #           (a1array - a0array) * (timestamp ** 2)) / (2 * (timestamp ** 5))
        #     qt = a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3) + a4 * (t ** 4) + a5 * (t ** 5)
        #     vt = a1 + 2 * a2 * t + 3 * a3 * (t ** 2) + 4 * a4 * (t ** 3) + 5 * a5 * (t ** 4)
        #     return qt.tolist(), vt.tolist()
        #
        # if inpfunc != "cubic" and inpfunc != "quintic":
        #     raise ValueError("Interpolation functions must be cubic or quintic")
        # inpfunccallback = cubic
        # if inpfunc == "quintic":
        #     inpfunccallback = quintic
        #
        # timesstamplist = []
        # speedsradlist = []
        # jointsradlist = []
        # for id, joints in enumerate(jointspath):
        #     jointsrad = [math.radians(angdeg) for angdeg in joints[0:6]]
        #     jointsradlist.append(jointsrad)
        #     if id == 0:
        #         timesstamplist.append([0.0] * 6)
        #     else:
        #         timesstamplist.append([timepathstep] * 6)
        #     if id == 0 or id == len(jointspath) - 1:
        #         speedsradlist.append([0.0] * 6)
        #     else:
        #         thisjointsrad = jointsrad
        #         prejointsrad = [math.radians(angdeg) for angdeg in jointspath[id - 1][0:6]]
        #         nxtjointsrad = [math.radians(angdeg) for angdeg in jointspath[id + 1][0:6]]
        #         presarray = (np.array(thisjointsrad) - np.array(prejointsrad)) / timepathstep
        #         nxtsarray = (np.array(nxtjointsrad) - np.array(thisjointsrad)) / timepathstep
        #         # set to 0 if signs are different
        #         selectid = np.where((np.sign(presarray) + np.sign(nxtsarray)) == 0)
        #         sarray = (presarray + nxtsarray) / 2.0
        #         sarray[selectid] = 0.0
        #         speedsradlist.append(sarray.tolist())
        # t = 0
        # timestep = self.__timestep
        # jointsradlisttimestep = []
        # speedsradlisttimestep = []
        # for idlist, timesstamp in enumerate(timesstamplist):
        #     if idlist == 0:
        #         continue
        #     timesstampnp = np.array(timesstamp)
        #     jointsradprenp = np.array(jointsradlist[idlist - 1])
        #     speedsradprenp = np.array(speedsradlist[idlist - 1])
        #     jointsradnp = np.array(jointsradlist[idlist])
        #     speedsradnp = np.array(speedsradlist[idlist])
        #     # reduce timestep in the last step to avoid overfitting
        #     if idlist == len(timesstamplist) - 1:
        #         while t <= timesstampnp.max():
        #             jsrad, vsrad = inpfunccallback(t, timesstampnp,
        #                                            jointsradprenp, speedsradprenp,
        #                                            jointsradnp, speedsradnp)
        #             jointsradlisttimestep.append(jsrad)
        #             speedsradlisttimestep.append(vsrad)
        #             t = t + timestep / 3
        #     else:
        #         while t <= timesstampnp.max():
        #             jsrad, vsrad = inpfunccallback(t, timesstampnp,
        #                                            jointsradprenp, speedsradprenp,
        #                                            jointsradnp, speedsradnp)
        #             jointsradlisttimestep.append(jsrad)
        #             speedsradlisttimestep.append(vsrad)
        #             t = t + timestep
        #         t = 0
        print("--------------passive move---------------")
        print("path length:", len(jointspath))
        print("path length inp:", len(jointsradlisttimestep))
        self.__programbuilder.load_prog(self.__scriptpath + "/urscripts/forcemode.script")
        prog = self.__programbuilder.get_program_to_run()
        prog = prog.replace("parameter_ip", urx_urmdsocket_ipad[0])
        prog = prog.replace("parameter_port", str(urx_urmdsocket_ipad[1]))
        prog = prog.replace("parameter_jointscaler", str(self.__jointscaler))
        prog = prog.replace("parameter_toolrelpose", str(toolrelpose))

        print(toolrelpose)
        self.arm.send_program(prog)

        # accept arm socket
        urmdsocket, urmdsocket_addr = urx_urmdsocket.accept()
        print("Connected by ", urmdsocket_addr)
        keepalive = 1
        buf = bytes()
        print(len(jointsradlisttimestep))
        for id, jointsrad in enumerate(jointsradlisttimestep):
            if id == len(jointsradlisttimestep) - 1:
                keepalive = 0
            jointsradint = [int(jointrad * self.__jointscaler) for jointrad in jointsrad]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                               jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        urmdsocket.send(buf)

        time.sleep(.5)
        while self.arm.is_program_running():
            pass


if __name__ == "__main__":
    import os
    import robotsim.robots.dualarm.ur3edual.ur3edual as ur3e
    import pandaplotutils.pandactrl as pandactrl
    import robotcon.ur3edual as ur3ex

    base = pandactrl.World(camp=[3, 0, 3], lookatpos=[0, 0, .7])
    rbt = ur3e.Ur3EDualRobot()
    rbtx = ur3ex.Ur3EDualUrx()

    fc = ForceController(rbt, rbtx)
    fc.attachfirm(forcethreshold=1)

    ForceDataSet = []
    movementData = []

    fc.attachfirm(forcethreshold=1)
    time.sleep(.5)
    print(rbtx.lft_arm_hnd.is_program_running())
    while rbtx.lft_arm_hnd.is_program_running():
        print(rbtx.lft_arm_hnd.get_tcp_force())
    print("==========================================")
    time.sleep(.5)
    print(rbtx.lft_arm_hnd.get_tcp_force())
