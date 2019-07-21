"""Unit tests for qfunc_impl.py."""

import numpy

from deep_learning.engine import qfunc_impl
from qpylib import numpy_util_test


class MemoizationQFunctionTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.MemoizationQFunction(action_space_size=2)

    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_GetSetValues(self):
    self.qfunc._SetValues(self.states, self.values)
    self.assertArrayEq(self.values, self.qfunc.GetValues(self.states))


class DQNTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.DQN(
      state_space_dim=3,
      action_space_size=2,
      hidden_layer_sizes=(6, 4),
      training_batch_size=1,
    )
    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_GetSetValues_convergence(self):
    for _ in range(100):
      self.qfunc._SetValues(self.states, self.values)
    diff1 = numpy.sum(
      numpy.abs(self.values - self.qfunc.GetValues(self.states)))
    for _ in range(1000):
      self.qfunc._SetValues(self.states, self.values)
    diff2 = numpy.sum(
      numpy.abs(self.values - self.qfunc.GetValues(self.states)))

    self.assertLess(diff2, diff1 / 5.0)
