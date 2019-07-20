"""Unit tests for circular_world_env.py."""

import numpy

from deep_learning.example import circular_world_env
from qpylib import numpy_util_test


class CircularWorldTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    self.env = circular_world_env.CircularWorld(size=5)

    self.left = numpy.array([[1, 0, 0]])
    self.stay = numpy.array([[0, 1, 0]])
    self.right = numpy.array([[0, 0, 1]])

  def test_left(self):
    tran = self.env.TakeAction(self.left)
    self.assertArrayEq(numpy.array([[-1]]), tran.sp)
    self.assertEqual(-1, tran.r)

  def test_stay(self):
    tran = self.env.TakeAction(self.stay)
    self.assertArrayEq(numpy.array([[0]]), tran.sp)
    self.assertEqual(0, tran.r)

  def test_right(self):
    tran = self.env.TakeAction(self.right)
    self.assertArrayEq(numpy.array([[1]]), tran.sp)
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
    self.assertEqual(-5, tran.sp)

    self.env._current_state = -5
    tran = self.env.TakeAction(self.left)
    self.assertEqual(5, tran.sp)

  def test_environmentDone(self):
    self.env._num_actions_taken = circular_world_env.STEP_LIMIT
    tran = self.env.TakeAction(self.right)
    self.assertIsNone(tran.sp)
