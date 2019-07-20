"""Unit tests for qfunc_impl.py."""

import numpy

from deep_learning.engine import qfunc_impl
from qpylib import numpy_util_test


class MemoizationQFunctionTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.MemoizationQFunction(action_space_size=2)

    self.action1 = numpy.array([[1, 0]])
    self.action2 = numpy.array([[0, 1]])

    self.actions = numpy.array([
      [1, 0],
      [0, 1],
    ])

    self.state1 = numpy.array([[1, 1, 1]])
    self.state2 = numpy.array([[2, 3, 4]])

    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.value1 = numpy.array([[0.5, 0.5]])
    self.value2 = numpy.array([[0.3, 0.7]])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_GetSetValues(self):
    self.qfunc._protected_SetValues(self.states, self.values)
    self.assertArrayEq(self.values, self.qfunc.GetValues(self.states))
