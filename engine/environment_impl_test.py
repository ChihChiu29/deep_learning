"""Unit tests for environment_impl.py."""
import unittest

import numpy

from deep_learning.engine import environment_impl


class SingleStateEnvironmentTest(unittest.TestCase):

  def test_stepLimit(self):
    env = environment_impl.SingleStateEnvironment(step_limit=10)
    env.Reset()

    transition = None
    for _ in range(10):
      transition = env.TakeAction(numpy.array([[0]]))
    self.assertIsNotNone(transition.sp)

    transition = env.TakeAction(numpy.array([[0]]))
    self.assertIsNone(transition.sp)
