import math
import pickle
import re
import time
import os
import config
import copy
from socket import socket, AF_INET, SOCK_STREAM

import matplotlib.pyplot as plt
from direct.stdpy import threading
import numpy as np
import utils.run_script_utils as rsu
import utiltools.robotmath as rm
import localenv.envloader as el
import pickle


class UR3eServer(object):
    def __init__(self, host, port, backlog=5, buffer_size=8192, armname='lft'):
        self.rbt, _, _ = el.loadUr3e()
        self.host = host
        self.port = port
        self.backlog = backlog
        self.buffer_size = buffer_size
        self.exp_name = "raft"
        self.exp_id = "down_2"
        self.show_penfarme = False
        self.objrelpos, self.objrelrot, _ = rsu.load_motion_sgl("draw_circle", f"exp_{self.exp_name}/",
                                                                config.ID_DICT[self.exp_name])
        # self.objrelpos, self.objrelrot, _ = pickle.load(
        #     open(config.MOTIONSCRIPT_REL_PATH + 'exp_force/draw_L.pkl', 'rb'))
        print(self.objrelpos)
        self.armname = armname
        self.__socket = socket(AF_INET, SOCK_STREAM)
        self.__socket.bind((self.host, self.port))
        self.__socket.listen(self.backlog)
        self.__current_index = 0
        try:
            self.__state = pickle.load(open(f"{config.ROOT}/log/exp/{self.exp_name}_state.pkl", "rb"))
            self.__state = [int(s) for s in self.__state]
            print('load state:', self.__state)
            info = pickle.load(open(f"{config.ROOT}/log/exp/{self.exp_name + '_' + self.exp_id}.pkl", "rb"))
            self.__force = info[0]
            self.__armjnts = info[1]
            self.__speed = info[2]
            self.__diff = info[3]
            self.__tcppose = info[4]
        except:
            self.__state = []
            self.__force = []
            self.__armjnts = []
            self.__speed = []
            self.__diff = []
            self.__tcppose = []

        self.__force_obj = []
        self.__torque_obj = []
        self.__speedl_obj = []
        self.__speedr_obj = []

        self.__flag = True
        self.__ploton = True
        self.__plotflag = False

    def start(self):
        def plot():
            print("plot start")
            fig = plt.figure(1, figsize=(16, 9))
            plt.ion()
            plt.show()
            plt.ylim((-10, 10))

            while self.__ploton:
                if self.__plotflag:
                    plt.clf()
                    # force = [l[:3] for l in self.__force]
                    # torque = [l[3:] for l in self.__force]
                    # speedl = [l[:3] for l in self.__speed]
                    # speedr = [l[3:] for l in self.__speed]
                    force = copy.deepcopy(self.__force_obj)
                    torque = copy.deepcopy(self.__torque_obj)
                    speedl = copy.deepcopy(self.__speedl_obj)
                    speedr = copy.deepcopy(self.__speedr_obj)
                    tcprot = [l[3:] for l in self.__tcppose]
                    diff = [d for d in self.__diff]

                    if any(len(lst) != len(force) for lst in [diff, force, torque, speedr, speedl, tcprot]):
                        continue
                    x = [i for i in range(len(force))]

                    plt.subplot(231)
                    plt.plot(x, force, label=["Fx", "Fy", "Fz"])
                    plt.title("Force")
                    plt.legend(("Fx", "Fy", "Fz"), loc='upper left')

                    plt.subplot(232)
                    plt.plot(x, torque, label=["Rx", "Ry", "Rz"])
                    plt.title("Torque")
                    plt.legend(("Rx", "Ry", "Rz"), loc='upper left')

                    plt.subplot(233)
                    plt.plot(x, diff, label="diff")
                    plt.title("TCP deviation")
                    plt.legend(loc='upper left')

                    plt.subplot(234)
                    plt.plot(x, speedl, label=["x", "y", "z"])
                    plt.title("Speed")
                    plt.legend(("x", "y", "z"), loc='upper left')

                    plt.subplot(235)
                    plt.plot(x, speedr, label=["Rx", "Ry", "Rz"])
                    plt.title("Speed")
                    plt.legend(("Rx", "Ry", "Rz"), loc='upper left')

                    plt.subplot(236)
                    plt.plot(x, tcprot, label=["Rx", "Ry", "Rz"])
                    plt.title("TCP rotation")
                    plt.legend(("Rx", "Ry", "Rz"), loc='upper left')

                    plt.pause(0.005)
                time.sleep(.1)
            plt.savefig(f"{config.ROOT}/log/exp/{self.exp_name}_{self.exp_id}.png")
            plt.close(fig)

        self.__thread_plot = threading.Thread(target=plot, name="plot")
        self.__thread_plot.start()
        while self.__flag:
            try:
                conn, address = self.__socket.accept()
                # print('Got connection from {}'.format(address))
                while True:
                    msg = conn.recv(self.buffer_size)
                    if not msg:
                        break
                    msg = re.findall(r"b\'(.+)\'", str(msg))[0]
                    if msg == "stop":
                        print("stop message received!")
                        # self.__stop_msg_handler()
                        conn.close()
                        self.__socket.close()
                        break
                    else:
                        self.__msg_handler(msg)
                        self.__plotflag = True
            except:
                self.__stop_msg_handler()
                self.__socket.close()
                self.__flag = False
                self.__ploton = False
                print("plot off")
                time.sleep(.05)
                self.__plotflag = False
                self.__thread_plot.join()
                self.__thread_plot = None

    def __stop_msg_handler(self):
        print("robot stop!")
        pickle.dump([self.__force, self.__armjnts, self.__speed, self.__diff, self.__tcppose,
                     [self.objrelpos, self.objrelrot]],
                    open(f"{config.ROOT}/log/exp/{self.exp_name + '_' + self.exp_id}.pkl", "wb"))
        self.__state.append(self.__current_index + 1)
        pickle.dump(self.__state, open(f"{config.ROOT}/log/exp/{self.exp_name}_state.pkl", "wb"))
        print(".pkl saved!")

    def __msg_handler(self, msg):
        T = np.eye(3)
        if self.show_penfarme:
            if len(self.__tcppose) > 0 and len(self.__armjnts) > 0:
                self.rbt.movearmfk(self.__armjnts[-1], self.armname)
                _, eerot = self.rbt.getee(armname=self.armname)
                baserot = np.dot(rm.rodrigues((1, 0, 0), -90), rm.rodrigues((0, 1, 0), 90))
                T = np.linalg.inv(np.dot(eerot, np.linalg.inv(self.objrelrot))).dot(baserot)
                # tcprot = rm.rotmat_from_euler(self.__tcppose[-1][3], self.__tcppose[-1][4], self.__tcppose[-1][5])
                # T = self.objrelrot.dot(np.linalg.inv(tcprot))

        if msg[0] == "f":
            self.__force.append(eval(msg[2:]))
            self.__force_obj.append(np.dot(T, self.__force[-1][:3]))
            self.__torque_obj.append(np.dot(T, self.__force[-1][3:]))
            print("force:", self.__force[-1])
        if msg[0] == "s":
            self.__speed.append(eval(msg[2:]))
            self.__speedl_obj.append(np.dot(T, self.__speed[-1][:3]))
            self.__speedr_obj.append(np.dot(T, self.__speed[-1][3:]))
            print("speed:", self.__speed[-1])
        if msg[0] == "t":
            self.__tcppose.append(eval(msg[2:]))
            print("tcp rotation:", self.__tcppose[-1])
        if msg[0] == "a":
            self.__armjnts.append([a * 180 / math.pi for a in eval(msg[1:])])
            print("armjnts:", self.__armjnts[-1])
        if msg[0] == "d":
            self.__diff.append(eval(msg[1:]))
            print("diff:", msg[1:])
        if msg[0] == "i":
            self.__current_index = int(msg[1:])
            print("index:", self.__current_index)


if __name__ == '__main__':
    ur3e_server = UR3eServer('0.0.0.0', 8000)
    ur3e_server.start()
