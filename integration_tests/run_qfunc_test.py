"""Run DQN with some Gym environments end to end."""
import unittest

import gym

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


class E2E_MemoizationQFunctionTest(unittest.TestCase):
  _multiprocess_can_split_ = True

  @staticmethod
  def _RunEnv(gym_env):
    env = environment_impl.GymEnvironment(gym_env)
    env.SetGymEnvMaxEpisodeSteps(10)
    qfunc = qfunc_impl.MemoizationQFunction(
      action_space_size=env.GetActionSpaceSize())
    env.Reset()
    policy = policy_impl.GreedyPolicyWithRandomness(epsilon=1.0)

    runner_impl.SimpleRunner().Run(
      env=env, qfunc=qfunc, policy=policy, num_of_episodes=1)

  def test_CartPole(self):
    self._RunEnv(gym.make('CartPole-v0'))

  def test_MountainCar(self):
    self._RunEnv(gym.make('MountainCar-v0'))

  def test_Acrobot(self):
    self._RunEnv(gym.make('Acrobot-v1'))

  def test_MsPacman(self):
    self._RunEnv(gym.make('MsPacman-v4'))

  def test_SpaceInvaders(self):
    self._RunEnv(gym.make('SpaceInvaders-v4'))


class E2E_DQNTest(unittest.TestCase):
  _multiprocess_can_split_ = True

  @staticmethod
  def _RunEnv(gym_env):
    env = environment_impl.GymEnvironment(gym_env)
    env.SetGymEnvMaxEpisodeSteps(10)
    qfunc = qfunc_impl.DQN(
      state_space_dim=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(4,),
      training_batch_size=4,
    )
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

  def test_SpaceInvaders(self):
    self._RunEnv(gym.make('SpaceInvaders-v4'))
