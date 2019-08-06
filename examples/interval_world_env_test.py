"""Unit tests for circular_world_env.py."""
import unittest

import numpy

from deep_learning.examples import circular_world_env
from deep_learning.examples import interval_world_env
from qpylib import numpy_util


class IntervalWorldTest(unittest.TestCase):

  def setUp(self) -> None:
    self.env = interval_world_env.IntervalWorld(size=5)
    self.env._current_state = 0

    self.left = numpy.array([[1, 0, 0]])
    self.stay = numpy.array([[0, 1, 0]])
    self.right = numpy.array([[0, 0, 1]])

  def test_left(self):
    tran = self.env.TakeAction(self.left)
    numpy_util.TestUtil.AssertArrayEqual(numpy.array([[-1]]), tran.sp)
    self.assertEqual(-1, tran.r)

  def test_stay(self):
    tran = self.env.TakeAction(self.stay)
    numpy_util.TestUtil.AssertArrayEqual(numpy.array([[0]]), tran.sp)
    self.assertEqual(0, tran.r)

  def test_right(self):
    tran = self.env.TakeAction(self.right)
    numpy_util.TestUtil.AssertArrayEqual(numpy.array([[1]]), tran.sp)
    self.assertEqual(-1, tran.r)

  def test_reward(self):
    self.env._current_state = 1
    tran = self.env.TakeAction(self.right)
    self.assertEqual(-1, tran.r)

    self.env._current_state = 1
    tran = self.env.TakeAction(self.left)
    self.assertEqual(1, tran.r)

    self.env._current_state = 1
    tran = self.env.TakeAction(self.stay)
    self.assertEqual(0, tran.r)

    self.env._current_state = -1
    tran = self.env.TakeAction(self.right)
    self.assertEqual(1, tran.r)

  def test_boundaryMoves(self):
    self.env._current_state = 5
    tran = self.env.TakeAction(self.right)
    self.assertIsNone(tran.sp)

    self.env._current_state = -5
    tran = self.env.TakeAction(self.left)
    self.assertIsNone(tran.sp)

  def test_environmentDone(self):
    self.env._num_actions_taken = circular_world_env.STEP_LIMIT
    tran = self.env.TakeAction(self.right)
    self.assertIsNone(tran.sp)
