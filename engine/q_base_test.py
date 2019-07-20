"""Unit tests for q_base.py."""

import numpy

from deep_learning.engine import qfunc_impl, q_base
from qpylib import numpy_util_test, logging


class QFunctionTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.MemoizationQFunction(state_space_size=3)

    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.actions = numpy.array([
      [1, 0],
      [0, 1],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_GetActionValues(self):
    self.qfunc._protected_SetValues(self.states, self.values)
    self.assertArrayEq(
      numpy.array([0.5, 0.7]),
      self.qfunc.GetActionValues(
        self.states, self.actions))

  def test_SetActionValues(self):
    self.qfunc._protected_SetValues(self.states, self.values)
    self.qfunc._SetActionValues(
      self.states, self.actions, numpy.array([0.2, 0.8]))
    logging.info(self.states)
    self.assertArrayEq(
      numpy.array([[0.2, 0.5], [0.3, 0.8]]),
      self.qfunc.GetValues(self.states))

  def test_UpdateValues_singleTransition(self):
    self.qfunc._protected_SetValues(
      numpy.array([
        [1, 2, 3],
        [4, 5, 6],
        [2, 2, 2],
      ]),
      numpy.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.8, 0.9],
      ]))

    self.qfunc.UpdateValues([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[0, 1]]),
      r=1.0,
      sp=numpy.array([[2, 2, 2]]),
    )], discount_factor=0.5)

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 1.0 = 1.45
    self.assertArrayEq(
      numpy.array([[0.5, 1.45]]),
      self.qfunc.GetValues(numpy.array([[1, 2, 3]])))

  def test_UpdateValues_multipleTransitions(self):
    self.qfunc._protected_SetValues(
      numpy.array([
        [1, 2, 3],
        [4, 5, 6],
        [2, 2, 2],
      ]),
      numpy.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.8, 0.9],
      ]))

    self.qfunc.UpdateValues([
      q_base.Transition(
        s=numpy.array([[1, 2, 3]]),
        a=numpy.array([[0, 1]]),
        r=1.0,
        sp=numpy.array([[2, 2, 2]]),
      ),
      q_base.Transition(
        s=numpy.array([[4, 5, 6]]),
        a=numpy.array([[0, 1]]),
        r=0.7,
        sp=numpy.array([[2, 2, 2]]),
      )],
      discount_factor=0.5)

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 1.0 = 1.45
    # The new values for state (4,5,6) should be:
    # - action (1,0): 0.3, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 0.7 = 1.15
    self.assertArrayEq(
      numpy.array([[0.5, 1.45], [0.3, 1.15]]),
      self.qfunc.GetValues(numpy.array([[1, 2, 3], [4, 5, 6]])))

  def test_UpdateValues_environmentDone(self):
    self.qfunc._protected_SetValues(
      numpy.array([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      numpy.array([
        [0.5, 0.5],
        [0.3, 0.7],
      ]))

    self.qfunc.UpdateValues([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[0, 1]]),
      r=1.0,
      sp=None,
    )], discount_factor=0.5)

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): 1.0, since environment is done, only reward is used.
    self.assertArrayEq(
      numpy.array([[0.5, 1.0]]),
      self.qfunc.GetValues(numpy.array([[1, 2, 3]])))
