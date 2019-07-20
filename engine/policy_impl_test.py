"""Unit tests for policy_impl.py."""
import unittest
from unittest import mock

import numpy

from deep_learning.engine import policy_impl, q_base


class PolicyTest(unittest.TestCase):

  def test_GreedyPolicyWithRandomness_choosesNonOptimalAction(self):
    mock_qfunc = mock.MagicMock()
    mock_qfunc.GetValues.return_value = numpy.array([[0.3, 0.7]])

    policy = policy_impl.GreedyPolicyWithRandomness(epsilon=0.5)
    for _ in range(500):
      policy.Decide(
        env=q_base.Environment(action_space_size=2, state_space_size=1),
        qfunc=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=0,
        num_of_episodes=500)

    # Tests that roughly half of the time qfunc is not used to make the
    # decision.
    self.assertGreater(mock_qfunc.GetValues.call_count, 200)
    self.assertLess(mock_qfunc.GetValues.call_count, 300)
