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

    def is_selfcollided(self, armjnts=None):
        if armjnts is not None:
            self.rbt.fk(self.armname, armjnts)
        flag, cps = self.rbt.is_collided(obstacle_list=self.obscmlist, toggle_contact_points=True)
        if len(cps) > 0:
            for p in cps:
                gm.gen_sphere(p, radius=.01).attach_to(base)
            if min([p[2] for p in cps]) > 7:
                return False
        return flag

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
        eepos, eerot = self.rbt.get_gl_tcp(armname=self.armname)
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

    def plot_nodepath(self, node_path, label=None, title="node path", show=True):
        x = range(len(node_path))
        plt.plot(x, [int(node.split("_")[1]) for node in node_path], label=label)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    def plot_vlist(self, vlist, label=None, title=None, show=True):
        plt.plot(range(len(vlist)), vlist, label=label)
        if title is not None:
            plt.title(title)
        if show:
            plt.show()

    def plot_rot_diff(self, mat4_list, label="", show=True):
        angle_list = []
        for i in range(1, len(mat4_list)):
            angle = rm.angle_between_vectors(mat4_list[i - 1][:3, 0], mat4_list[i][:3, 0])
            angle_list.append(angle)
        plt.plot(np.asarray(angle_list), label=label)
        plt.legend()
        plt.title("rotation difference")
        plt.xlabel("id")
        plt.ylabel("degree")
        if show:
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
