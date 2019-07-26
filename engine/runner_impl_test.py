"""Unit tests for runner_impl.py."""
import unittest

import numpy

from deep_learning.engine import q_base
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


class ExperienceReplayRunnerTest(unittest.TestCase):

  def test_memoryManagement(self):
    qfunc = qfunc_impl.RandomValueQFunction(action_space_size=2)
    runner = runner_impl.ExperienceReplayRunner(
      experience_capacity=1,
      experience_sample_batch_size=1,
      train_every_n_steps=1)

    tran1 = q_base.Transition(
      s=numpy.array([[1, 2]]),
      a=numpy.array([[1, 0]]),
      r=1,
      sp=numpy.array([[3, 4]]))

    tran2 = q_base.Transition(
      s=numpy.array([[3, 4]]),
      a=numpy.array([[0, 1]]),
      r=1,
      sp=numpy.array([[5, 6]]))

    runner._protected_ProcessTransition(qfunc, tran1, 0)
    runner._protected_ProcessTransition(qfunc, tran2, 1)

    hist = runner.GetHistory()
    self.assertEqual(1, len(hist))
    self.assertEqual(tran2, hist[0])
