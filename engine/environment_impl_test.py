"""Unit tests for environment_impl.py."""
import unittest

import gym
import numpy

from deep_learning.engine import environment_impl
from qpylib import numpy_util_test


class SingleStateEnvironmentTest(unittest.TestCase):

  def test_stepLimit(self):
    env = environment_impl.SingleStateEnvironment(
      action_space_size=1, step_limit=10)
    env.Reset()

    transition = None
    for _ in range(10):
      transition = env.TakeAction(numpy.array([[0]]))
    self.assertIsNotNone(transition.sp)

    transition = env.TakeAction(numpy.array([[0]]))
    self.assertIsNone(transition.sp)


class GymEnvironment(numpy_util_test.NumpyTestCase):

  def test_getStateShape(self):
    env = environment_impl.GymEnvironment(gym.make('SpaceInvaders-v4'))
    self.assertEqual((210, 160, 3), env.GetStateShape())

  def test_takeAction(self):
    env = environment_impl.GymEnvironment(gym.make('CartPole-v1'))
    s = env.Reset()
    self.assertEqual((1, 4), s.shape)

    transition = env.TakeRandomAction()
    self.assertEqual((1, 4), transition.s.shape)
    self.assertEqual((1, 4), transition.sp.shape)

    old_s = transition.sp
    transition = env.TakeRandomAction()
    self.assertArrayEq(old_s, transition.s)
