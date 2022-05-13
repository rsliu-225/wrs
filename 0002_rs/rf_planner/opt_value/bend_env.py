import numpy as np
import visualization.panda.world as wd

import bendplanner.BendSim as bs
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig


class BendEnv(object):
    def __init__(self, goal_pseq, pseq=None, rotseq=None, show=False, granularity=np.pi / 90, cm_type='stick'):
        self._pseq, self._rotseq = pseq, rotseq

        self._goal_pseq = goal_pseq
        self._goal_rotseq = bu.get_rotseq_by_pseq(self._goal_pseq)

        self._init_rot = np.eye(3)

        base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])

        self._sim = bs.BendSim(pseq, rotseq, show, granularity, cm_type)
        self._sim.reset(self._pseq, self._rotseq)
        self._sim.move_to_org(bconfig.INIT_L)

        # Align the goal with initialization
        self._goal_pseq, self._goal_rotseq = bu.align_with_init(
            self._sim, self._goal_pseq, self._init_rot, self._goal_rotseq,
        )
        self._goal_voxel = bu.voxelize(self._goal_pseq, self._goal_rotseq, bconfig.THICKNESS)
        self._goal_voxel_one_hot = bu.onehot_voxel(self._goal_voxel)

        self._reward_thres = 0.7
        self._reward_bonus = 5

    def get_observation(self, one_hot=True):
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
        bu.visualize_voxel(grids, colors)

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
        is_success, _, _ = self._sim.gen_by_bendseq([action])
        self._sim.move_to_org(bconfig.INIT_L)

        if not is_success[0]:
            done = True
        else:
            done = False

        # observation = bu.avg_polylines_dist_err(self._sim.pseq, self._goal_pseq)
        observation = self.get_observation()
        reward = self._get_reward_per_step(observation)
        info = {}

        return observation, reward, done, info

    def render(self):
        self._sim.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    def reset(self):
        self._sim.reset(self._pseq, self._rotseq)
        self._sim.move_to_org(bconfig.INIT_L)

    def _get_reward_per_step(self, observation):
        """
        Calculate the per step reward based on current point set and the target points set(self._goal_set)

        """
        # err, _ = bu.avg_polylines_dist_err(self._goal_pseq, self._sim.pseq)

        matched = np.count_nonzero(self._goal_voxel_one_hot.astype(np.bool).flatten() & observation.astype(np.bool).flatten())
        recall = matched / np.count_nonzero(self._goal_voxel_one_hot.astype(np.bool).flatten())

        if recall >= self._reward_thres:
            return recall + self._reward_bonus
        else:
            return recall






