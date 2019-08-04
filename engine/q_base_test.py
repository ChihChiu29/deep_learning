"""Unit tests for q_base.py."""
import unittest
from unittest import mock

import numpy

from deep_learning.engine import brain_impl
from deep_learning.engine import q_base
from deep_learning.engine import runner_impl
from qpylib import logging
from qpylib import numpy_util
from qpylib import numpy_util_test


class BrainTest(unittest.TestCase):

  def testGetActionValues(self):
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([2, 6]),
      q_base.Brain.GetActionValues(
        numpy.array([[1, 2, 3], [4, 5, 6]]),
        numpy.array([[0, 1, 0], [0, 0, 1]]),
      ))

  def testCombineTransitions(self):
    states, actions, rewards, new_states, reward_mask = (
      q_base.Brain.CombineTransitions([
        q_base.Transition(
          s=numpy.array([[1, 2, 3]]),
          a=numpy.array([[0, 1, 0]]),
          r=1.0,
          sp=numpy.array([[4, 5, 6]]),
        ),
        q_base.Transition(
          s=numpy.array([[4, 5, 6]]),
          a=numpy.array([[0, 0, 1]]),
          r=-1.0,
          sp=None,
        ),
      ]))
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([[1, 2, 3], [4, 5, 6]]), states)
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([[0, 1, 0], [0, 0, 1]]), actions)
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([1.0, -1.0]), rewards)
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([[4, 5, 6], [4, 5, 6]]), new_states)
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([1, 0]), reward_mask)


class QFunctionTest(unittest.TestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    # Learning from old values is disabled in the majority of tests.
    self.qfunc = brain_impl.MemoizationBrain(
      action_space_size=2,
      discount_factor=0.5,
      learning_rate=1.0,
    )

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
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([0.5, 0.7]),
      self.qfunc.GetActionValues(
        self.qfunc.GetValues(self.states), self.actions))

  def test_SetActionValues(self):
    self.qfunc._protected_SetValues(self.states, self.values)
    self.qfunc._SetActionValues(
      self.states, self.actions, numpy.array([0.2, 0.8]))
    logging.info(self.states)
    numpy_util.TestUtil.AssertArrayEqual(
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

    self.qfunc.UpdateFromTransitions([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[0, 1]]),
      r=1.0,
      sp=numpy.array([[2, 2, 2]]),
    )])

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 1.0 = 1.45
    numpy_util.TestUtil.AssertArrayEqual(
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

    self.qfunc.UpdateFromTransitions([
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
      )])

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 1.0 = 1.45
    # The new values for state (4,5,6) should be:
    # - action (1,0): 0.3, since it's not changed.
    # - action (0,1): max(0.8, 0.9) * 0.5 + 0.7 = 1.15
    numpy_util.TestUtil.AssertArrayEqual(
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

    self.qfunc.UpdateFromTransitions([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[0, 1]]),
      r=1.0,
      sp=None,
    )])

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): 1.0, since environment is done, only reward is used.
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([[0.5, 1.0]]),
      self.qfunc.GetValues(numpy.array([[1, 2, 3]])))

  def test_learningRate(self):
    # Disables learning from Q* to simplifies testing.
    qfunc = brain_impl.MemoizationBrain(
      action_space_size=2,
      discount_factor=0.0,
      learning_rate=0.9,
    )
    qfunc._protected_SetValues(
      numpy.array([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      numpy.array([
        [0.5, 0.6],
        [0.3, 0.7],
      ]))
    qfunc.UpdateFromTransitions([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[0, 1]]),
      r=1.0,
      sp=numpy.array([[2, 2, 2]]),
    )])

    # The new values for state (1,2,3) should be:
    # - action (1,0): 0.5, since it's not changed.
    # - action (0,1): (1-0.9) * 0.6 + 0.9 * 1.0 = 0.96.
    numpy_util.TestUtil.AssertArrayEqual(
      numpy.array([[0.5, 0.96]]),
      qfunc.GetValues(numpy.array([[1, 2, 3]])))


class RunnerTest(numpy_util_test.NumpyTestCase):

  def setUp(self) -> None:
    self.env = mock.MagicMock()
    self.qfunc = mock.MagicMock()
    self.policy = mock.MagicMock()

    self.runner = runner_impl.NoOpRunner()

  def test_runUsesNewStateAfterIteration(self):
    self.env.TakeAction.side_effect = [
      q_base.Transition(
        s=numpy.array([[0]]),
        a=numpy.array([[0]]),
        r=1.0,
        sp=numpy.array([[1]]),
      ),
      q_base.Transition(
        s=numpy.array([[1]]),
        a=numpy.array([[0]]),
        r=1.0,
        sp=None,
      )
    ]

    self.runner.Run(
      env=self.env,
      qfunc=self.qfunc,
      policy=self.policy,
      num_of_episodes=1,
    )

    # Tests that the second call is with the new state 1.
    self.policy.Decide.assert_called_with(
      env=mock.ANY, brain=mock.ANY, state=numpy.array([[1]]),
      episode_idx=0, num_of_episodes=1)
