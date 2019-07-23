"""Run some Gym environments end to end."""
import unittest

import gym

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


class RunGymTest(unittest.TestCase):

  @staticmethod
  def _RunEnv(gym_env):
    env = environment_impl.GymEnvironment(gym_env)
    env.Reset()
    for _ in range(10):
      env.TakeRandomAction()

  def test_CartPoleV0(self):
    self._RunEnv(gym.make('CartPole-v0'))

  def test_MountainCarV0(self):
    self._RunEnv(gym.make('MountainCar-v0'))

  def test_AcrobotV1(self):
    self._RunEnv(gym.make('Acrobot-v1'))

  def test_MsPacmanV4(self):
    self._RunEnv(gym.make('MsPacman-v4'))


class RunGymWithFullSetupTest(unittest.TestCase):

  @staticmethod
  def _RunEnv(gym_env):
    env = environment_impl.GymEnvironment(gym_env)
    qfunc = qfunc_impl.RandomValueQFunction(
      action_space_size=env.GetActionSpaceSize())
    env.Reset()
    policy = policy_impl.GreedyPolicyWithRandomness(epsilon=1.0)

    runner_impl.SimpleRunner().Run(
      env=env, qfunc=qfunc, policy=policy, num_of_episodes=10)

  def test_CartPole(self):
    self._RunEnv(gym.make('CartPole-v0'))

  def test_MountainCar(self):
    self._RunEnv(gym.make('MountainCar-v0'))

  def test_Acrobot(self):
    self._RunEnv(gym.make('Acrobot-v1'))

  def test_MsPacman(self):
    self._RunEnv(gym.make('MsPacman-v4'))
