import random

import numpy as np
import visualization.panda.world as wd

import bendplanner.BendSim as bs
import bendplanner.bend_utils as bu
import bendplanner.bender_config as bconfig


class BendEnv(object):
    def __init__(self, goal_pseq, pseq=None, rotseq=None, show=False, granularity=np.pi / 90, cm_type='stick'):
        self._goal_pseq = goal_pseq
        self._pseq = pseq
        self._rotseq = rotseq
        self._show = show
        self._granularity = granularity
        self._cm_type = cm_type

        fit_pseq = bu.decimate_pseq(self._goal_pseq, r=bconfig.R_CENTER, tor=.0002, toggledebug=False)
        self._bendset, _ = bu.pseq2bendset(fit_pseq, toggledebug=False)
        self._res_action = list(range(len(self._bendset)))

        self._sim = bs.BendSim(self._pseq, self._rotseq, self._show, self._granularity, self._cm_type)

    def get_observation(self):
        return np.asarray(self._sim.pseq).flatten()

    def sample_action(self):
        return random.choice(self._res_action)

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
        print('action:', self._bendset[action])

        is_success, _, _ = self._sim.gen_by_bendseq([self._bendset[action]])
        if is_success[0]:
            reward = 1
            self._res_action.remove(action)
        else:
            reward = -10

        # reward = self._get_reward_per_step()
        info = {}

        if len(self._res_action) == 0:
            done = True
        else:
            done = False

        return np.asarray(self._sim.pseq).flatten(), reward, done, info

    def render(self):
        self._sim.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    def reset(self):
        self._sim.reset(self._pseq, self._rotseq)
        return np.asarray(self._sim.pseq).flatten()

    def _get_reward_per_step(self):
        """
        Calculate the per step reward based on current point set and the target points set(self._goal_set)

        """
        err, _ = bu.avg_distance_between_polylines(self._goal_pseq, self._sim.pseq)
        return 1
