import numpy as np
import visualization.panda.world as wd

from .BendSim import BendSim


class BendEnv():
    def __init__(self, goal_set, pseq=None, rotseq=None, show=False, granularity=np.pi / 90, cm_type='stick'):
        self._goal_set = goal_set

        self._pseq = pseq
        self._rotseq = rotseq
        self._show = show
        self._granularity = granularity
        self._cm_type = cm_type

        base = wd.World(cam_pos=[0, 0, .2], lookat_pos=[0, 0, 0])
        self._sim = BendSim(self._pseq, self._rotseq, self._show, self._granularity, self._cm_type)
        base.run()

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

        if not is_success:
            done = True
        else:
            done = False

        observation = self._sim.objcm

        reward = self._calculate_reward_per_step()

        info = {}

        return observation, reward, done, info

    def render(self):
        self._sim.show(rgba=(.7, .7, .7, .7), show_frame=True, show_pseq=False)

    def reset(self):
        self._sim.reset(self._pseq, self._rotseq)

    def _calculate_reward_per_step(self):
        """
        Calculate the per step reward based on current point set and the target points set(self._goal_set)

        # TODO: implement how to calculate reward here
        """
        return 1
