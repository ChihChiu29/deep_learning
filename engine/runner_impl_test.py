"""Unit tests for runner_impl.py."""
import unittest
from unittest import mock

import numpy

from deep_learning.engine import base
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


class ExperienceReplayRunnerTest(unittest.TestCase):

  def test_memoryManagement(self):
    qfunc = qfunc_impl.RandomQFunction(action_space_size=2)
    runner = runner_impl.ExperienceReplayRunner(
      experience_capacity=1,
      experience_sample_batch_size=1,
      train_every_n_steps=1)

    tran1 = base.Transition(
      s=numpy.array([[1, 2]]),
      a=numpy.array([[1, 0]]),
      r=1,
      sp=numpy.array([[3, 4]]))

    tran2 = base.Transition(
      s=numpy.array([[3, 4]]),
      a=numpy.array([[0, 1]]),
      r=1,
      sp=numpy.array([[5, 6]]))

    runner._protected_ProcessTransition(qfunc, tran1, 0)
    runner._protected_ProcessTransition(qfunc, tran2, 1)

    hist = runner._experience._history
    self.assertEqual(1, len(hist))
    self.assertEqual(tran2, hist[0])


class NStepExperienceRunnerTest(unittest.TestCase):

  def setUp(self) -> None:
    self.brain = mock.MagicMock()
    self.runner = runner_impl.NStepExperienceRunner(
      discount_factor=0.5,
      n_step_return=5,
    )
    self.tran = base.Transition(
      s=numpy.array([[0]]),
      a=numpy.array([[1]]),
      r=1.0,
      sp=numpy.array([[0]]),
    )

  def testCalculateNStepReward_beforeMemoryFull(self):
    for _ in range(3):
      self.runner._protected_ProcessTransition(self.brain, self.tran, 0)
    self.assertAlmostEqual(
      1 + 0.5 + 0.5 ** 2, self.runner._GetNStepTransition().r)

  def testCalculateNStepReward_afterMemoryFull(self):
    for _ in range(10):
      self.runner._protected_ProcessTransition(self.brain, self.tran, 0)
    # ProcessTransition pops one transition when it's at capacity.
    self.assertAlmostEqual(
      1 + 0.5 + 0.5 ** 2 + 0.5 ** 3,
      self.runner._GetNStepTransition().r)

  def testCalculateNStepReward_whenDone(self):
    for _ in range(4):
      self.runner._protected_ProcessTransition(self.brain, self.tran, 0)
    self.assertFalse(self.brain.called)
    tran = base.Transition(
      s=numpy.array([[0]]),
      a=numpy.array([[1]]),
      r=1.0,
      sp=None,
    )
    self.runner._protected_ProcessTransition(self.brain, tran, 0)

    rewards = []
    for tran in self.brain.UpdateFromTransitions.call_args[0][0]:
      rewards.append(tran.r)
    self.assertCountEqual(
      [1.0, 1.0 + 0.5, 1.0 + 0.5 + 0.5 ** 2, 1.0 + 0.5 + 0.5 ** 2 + 0.5 ** 3,
       1.0 + 0.5 + 0.5 ** 2 + 0.5 ** 3 + 0.5 ** 4],
      rewards
    )
