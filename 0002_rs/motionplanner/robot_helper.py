import copy

import matplotlib.pyplot as plt
import numpy as np
import modeling.geometric_model as gm
import basis.robot_math as rm


class RobotHelper(object):
    def __init__(self, env, rbt, armname="lft_arm"):
        self.rbt = rbt
        self.env = env
        self.obscmlist = env.getstationaryobslist() + env.getchangableobslist()
        self.armname = armname
        self.initjnts = self.rbt.get_jnt_values(self.armname)
        # self.armlj = self.rbt.get_jnt_values

    def jacobian(self, releemat4=np.eye(4), scale=1.0):
        """
        compute the jacobian matrix of the targetjoints of rgt or lft arm

        :param scale:
        :param releemat4:
        :return: armjac a 6-by-x matrix

        """

        # TODO: fix releemat4 error
        armjac = np.zeros((6, len(self.rbt.targetjoints)))
        counter = 0

        endmat4 = np.dot(rm.homomat_from_posrot(self.armlj[self.rbt.targetjoints[-1]]["linkpos"],
                                                self.armlj[self.rbt.targetjoints[-1]]["rotmat"]), releemat4)
        endpos = endmat4[:3, 3]
        for i in self.rbt.targetjoints:
            if i != self.rbt.targetjoints[-1]:
                a = np.dot(self.armlj[i]["rotmat"], self.armlj[i]["rotax"])
            else:
                a = np.dot(endmat4[:3, :3], self.armlj[i]["rotax"])
            armjac[:, counter] = np.append(np.cross(a, endpos - self.armlj[i]["linkpos"]) * scale, a)
            counter += 1

        # for i in self.rbt.targetjoints:
        #     a = np.dot(self.armlj[i]["rotmat"], self.armlj[i]["rotax"])
        #     armjac[:, counter] = np.append(
        #         np.cross(a, self.armlj[self.rbt.targetjoints[-1]]["linkpos"] - self.armlj[i]["linkpos"]) * scale, a)
        #     counter += 1

        return armjac

    def manipulability(self, releemat4=np.eye(4), armjnts=None):
        if armjnts is not None:
            self.goto_armjnts(armjnts)
        armjac = self.jacobian(releemat4=releemat4, scale=1.0)
        return np.sqrt(np.linalg.det(np.dot(armjac, armjac.transpose())))

    def manipulability_axmat(self, releemat4=np.eye(4), armjnts=None):
        if armjnts is not None:
            self.goto_armjnts(armjnts)
        armjac = self.jacobian(releemat4=releemat4, scale=1.0)
        jjt = np.dot(armjac, armjac.T)
        pcv, pcaxmat = np.linalg.eig(jjt)
        # only keep translation
        axmat = np.eye(3)
        axmat[:, 0] = np.sqrt(pcv[0]) * pcaxmat[:3, 0]
        axmat[:, 1] = np.sqrt(pcv[1]) * pcaxmat[:3, 1]
        axmat[:, 2] = np.sqrt(pcv[2]) * pcaxmat[:3, 2]
        return axmat

    def tcperror(self, tgtpos, tgtrot, releemat4=np.eye(4), scale=1.0):
        """
        compute the error of a specified (rgt or lft) tool point center to its goal

        :param tgtpos: the position of the goal
        :param tgtrot: the rotation of the goal
        :param releemat4:
        :param scale:
        :return: a 1-by-6 vector where the first three indicates the displacement in pos,
                    the second three indictes the displacement in rot
        """
        eepos = copy.deepcopy(self.armlj[self.rbt.targetjoints[-1]]["linkend"])
        eerot = copy.deepcopy(self.armlj[self.rbt.targetjoints[-1]]["rotmat"])
        eemat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), releemat4)
        eepos = eemat4[:3, 3]
        eerot = eemat4[:3, :3]
        deltapos = (tgtpos - eepos) * scale
        deltaw = rm.deltaw_between_rotmat(tgtrot, eerot.T)
        return np.append(deltapos, deltaw)

    def is_selfcollided(self, armjnts=None):
        if armjnts is not None:
            self.rbt.fk(self.armname, armjnts)
        return self.rbt.is_collided(obstacle_list=self.obscmlist)

    def is_objcollided(self, obj, armjnts=None):
        if armjnts is not None:
            self.rbt.fk(self.armname, armjnts)
        return self.rbt.is_collided(obstacle_list=self.obscmlist + [obj])

    def goto_armjnts(self, armjnts):
        self.rbt.fk(self.armname, armjnts)

    def goto_initarmjnts(self):
        if self.armname == "rgt_arm":
            self.goto_armjnts(self.rbt.initrgtjnts)
        if self.armname == "lft_arm":
            self.goto_armjnts(self.rbt.initlftjnts)

    def get_ee(self, armjnts=None, releemat4=np.eye(4)):
        if armjnts is not None:
            self.goto_armjnts(armjnts)
        eepos, eerot = self.rbt.getee(armname=self.armname)
        eemat4 = np.dot(rm.homomat_from_posrot(eepos, eerot), releemat4)
        return eemat4[:3, 3], eemat4[:3, :3]

    def get_tcp(self, armjnts=None):
        if armjnts is not None:
            self.goto_armjnts(armjnts)
        return self.rbt.get_gl_tcp(manipulator_name=self.armname)

    def draw_axis(self, pos, rot, rgba=None, length=50, thickness=5):
        gm.gen_frame(pos=pos, rotmat=rot, length=length, thickness=thickness)
        gm.gen_sphere(pos, radius=10, rgba=(1, 1, 0, 1))

    def draw_axis_uneven(self, pos, rot, scale=1, thickness=.005):
        gm.gen_arrow(spos=pos, epos=pos + scale * rot[:3, 0], rgba=(1, 0, 0, 1), thickness=thickness)
        gm.gen_arrow(spos=pos, epos=pos + scale * rot[:3, 1], rgba=(0, 1, 0, 1), thickness=thickness)
        gm.gen_arrow(spos=pos, epos=pos + scale * rot[:3, 2], rgba=(0, 0, 1, 1), thickness=thickness)

    def plot_armjnts(self, path, scatter=False, title="armjnts", show=True):
        path = np.array(path)
        x = range(len(path))
        if scatter:
            plt.scatter(x, [p for p in path[:, 0]])
            plt.scatter(x, [p for p in path[:, 1]])
            plt.scatter(x, [p for p in path[:, 2]])
            plt.scatter(x, [p for p in path[:, 3]])
            plt.scatter(x, [p for p in path[:, 4]])
            plt.scatter(x, [p for p in path[:, 5]])
        else:
            plt.plot(x, [p for p in path[:, 0]])
            plt.plot(x, [p for p in path[:, 1]])
            plt.plot(x, [p for p in path[:, 2]])
            plt.plot(x, [p for p in path[:, 3]])
            plt.plot(x, [p for p in path[:, 4]])
            plt.plot(x, [p for p in path[:, 5]])
        plt.title(title)
        # plt.legend(range(6))
        if show:
            plt.show()

    def plot_nodepath(self, node_path, label=None, title="node path"):
        x = range(len(node_path))
        plt.plot(x, [int(node.split("_")[1]) for node in node_path], label=label)
        if title is not None:
            plt.title(title)
        # plt.show()

    def plot_vlist(self, vlist, label=None, title=None):
        plt.plot(range(len(vlist)), vlist, label=label)
        if title is not None:
            plt.title(title)
        # plt.show()

    def plot_rot_diff(self, mat4_list, label=""):
        angle_list = []
        for i in range(1, len(mat4_list)):
            angle = rm.degree_betweenvector(mat4_list[i - 1][:3, 0], mat4_list[i][:3, 0])
            angle_list.append(angle)
        plt.plot(np.asarray(angle_list), label=label)
        plt.legend()
        plt.title("rotation difference")
        plt.xlabel("id")
        plt.ylabel("degree")
        plt.show()

    def show_armjnts(self, rgba=None, armjnts=None, toggleendcoord=False, jawwidth=50, genmnp=True):
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, self.armname)
        if genmnp:
            self.__genmnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord, jawwidth=jawwidth)
        else:
            self.__gensnp_by_armname(rgba=rgba, toggleendcoord=toggleendcoord)

    def __genmnp_by_armname(self, rgba, toggleendcoord=False, jawwidth=50):
        self.rbt.closegripper(armname=self.armname, jawwidth=jawwidth)
        if self.armname == "lft_arm":
            self.rbt.gen_meshmodel(toggle_tcpcs=toggleendcoord, rgba=rgba).attach_to(base)
        else:
            self.rbt.gen_meshmodel(toggle_tcpcs=toggleendcoord, rgba=rgba).attach_to(base)

    def __gensnp_by_armname(self, rgba, toggleendcoord=False):
        self.rbt.closegripper(armname=self.armname)
        if self.armname == "lft_arm":
            self.rbt.gen_stickmodel(toggle_jntscs=False, toggle_tcpcs=toggleendcoord).attach_to(base)
        else:
            self.rbt.gen_stickmodel(toggle_jntscs=False, toggle_tcpcs=toggleendcoord).attach_to(base)


if __name__ == '__main__':
    '''
    set up env and param
    '''
    from localenv import envloader as el

    base, env = el.loadEnv_wrs()

    rbt = el.loadUr3e(showrbt=False)
    rbth = RobotHelper(env, rbt, "lft_arm")
    print(rbth.manipulability())
