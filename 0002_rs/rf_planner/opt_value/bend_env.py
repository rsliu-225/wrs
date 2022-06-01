import numpy as np
import visualization.panda.world as wd

import bendplanner.BendSim as bs
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig
import modeling.geometric_model as gm
import open3d as o3d
import basis.o3dhelper as o3dh
from collections import Counter


class BendEnv(object):
    def __init__(self, goal_pseq, pseq=None, rotseq=None, show=True, granularity=np.pi / 90, cm_type='stick'):
        self._pseq, self._rotseq = pseq, rotseq

        self._goal_pseq = goal_pseq
        self._goal_rotseq = bu.get_rotseq_by_pseq(self._goal_pseq)

        base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])

        self._sim = bs.BendSim(pseq, rotseq, show, granularity, cm_type)
        self._sim.reset(self._pseq, self._rotseq)
        self._sim.move_to_org(bconfig.INIT_L)

        # fit_pseq, _ = bu.decimate_pseq(goal_pseq, tor=.0002, toggledebug=False)

        self._goal_voxel = bu.voxelize(self._goal_pseq, self._goal_rotseq, bconfig.THICKNESS)
        self._goal_voxel_one_hot = bu.onehot_voxel(self._goal_voxel)

        self._reward_thres = 0.7
        self._reward_bonus = 5

        self._action_space_low = [-np.pi / 2, 0, -np.pi, bconfig.INIT_L]
        self._action_space_high = [np.pi / 2, 0, np.pi, bu.cal_length(self._goal_pseq)]

        self._init_rot = np.eye(3)

    def get_observation(self, one_hot=True):
        self._sim.move_to_org()
        voxel = self._sim.voxelize()
        if one_hot:
            return bu.onehot_voxel(voxel)
        else:
            return voxel

    def visualize_observation(self, vis_goal=True):
        if vis_goal:
            grids = [self.get_observation(False), self._goal_voxel]
            colors = ['g', 'r']
        else:
            grids = [self.get_observation(False)]
            colors = ['g']
        o3d.visualization.draw_geometries(grids)
        # bu.visualize_voxel(grids, colors)

    def sample_action(self):
        # TODO: implement random action sampling from action space here
        return [0.0, 0.0, 0.0, 0.0]

    def step(self, action):
        """
        Step the environment.

        Here, action is [bend_angle_radian, lift_angle_radian, rot_angle_radian, bend_position]

        Args:
        ----
            action (numpy.ndarray): the action array, including [bend_angle_radian, lift_angle_radian,
            , bend_position]


        Returns:
        -------
            numpy.ndarray, float, boolean, dict: observation, reward, done and info
        """
        # print(f"action before clip {action}")
        for i in range(4):
            action[i] = np.clip(action[i], self._action_space_low[i], self._action_space_high[i])
        print(f"action after clip {action}")
        is_success, _, _ = self._sim.gen_by_bendseq([action], cc=False)
        self._init_rot = bu.get_init_rot(self._sim.pseq)
        goal_pseq_tmp, goal_rotseq_tmp = bu.align_with_init(
            self._sim, self._goal_pseq, self._init_rot, self._goal_rotseq,
        )

        self._goal_voxel = bu.voxelize(goal_pseq_tmp, goal_rotseq_tmp, bconfig.THICKNESS)
        self._goal_voxel_one_hot = bu.onehot_voxel(self._goal_voxel)
        if not is_success[0]:
            done = True
        else:
            done = False

        observation = self.get_observation()
        reward = self._get_reward_per_step()
        print('reward:', reward)
        info = {}
        self._sim.show(rgba=(1, 1, 1, .5))

        return observation, reward, done, info

    def render(self):
        self._sim.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    def reset(self):
        self._sim.reset(self._pseq, self._rotseq)
        self._sim.move_to_org(bconfig.INIT_L)

    def _get_reward_per_step(self):
        """
        Calculate the per step reward based on current point set and the target points set(self._goal_set)

        """
        # try:
        #     err, _ = bu.mindist_err(self._sim.pseq, self._goal_pseq)
        #     print(.1 / err)
        # except:
        #     print(self._sim.pseq)
        #     print(self._goal_pseq)
        querier = np.asarray(self._sim.objcm.sample_surface(radius=.0005)[0])
        overlap_mask = self._goal_voxel.check_if_included(o3d.utility.Vector3dVector(querier))
        overlap_o3dpcd = o3dh.nparray2o3dpcd(np.asarray(querier[overlap_mask]))
        overlap_o3dpcd.paint_uniform_color([1, 0.706, 0])
        try:
            recall = overlap_mask.count(True) / len(overlap_mask)
        except:
            recall = 0
        if recall >= self._reward_thres:
            return recall + self._reward_bonus
        else:
            return recall
