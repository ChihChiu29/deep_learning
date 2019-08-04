"""Unit tests for policy_impl.py."""
from unittest import mock

import numpy

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from qpylib import numpy_util_test


class PolicyTest(numpy_util_test.NumpyTestCase):

  def test_GreedyPolicy_choosesOptimalAction(self):
    mock_qfunc = mock.MagicMock()
    mock_qfunc.GetValues.return_value = numpy.array([[0.3, 0.7]])

    policy = policy_impl.GreedyPolicy()
    self.assertArrayEq(
      numpy.array([[0, 1]]),
      policy.Decide(
        env=environment_impl.SingleStateEnvironment(
          action_space_size=2, step_limit=10),
        brain=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=0,
        num_of_episodes=500))

  def test_GreedyPolicyWithRandomness_choosesNonOptimalAction(self):
    mock_qfunc = mock.MagicMock()
    mock_qfunc.GetValues.return_value = numpy.array([[0.3, 0.7]])

    policy = policy_impl.GreedyPolicyWithRandomness(epsilon=0.5)
    for _ in range(500):
      policy.Decide(
        env=environment_impl.SingleStateEnvironment(
          action_space_size=2, step_limit=10),
        brain=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=0,
        num_of_episodes=500)

    # Tests that roughly half of the time qfunc is not used to make the
    # decision.
    self.assertGreater(mock_qfunc.GetValues.call_count, 200)
    self.assertLess(mock_qfunc.GetValues.call_count, 300)

  def test_GreedyPolicyWithRandomness_considerAllActions(self):
    mock_qfunc = mock.MagicMock()
    mock_qfunc.GetValues.return_value = numpy.array([[0.3, 0.7]])

    env = environment_impl.SingleStateEnvironment(
      action_space_size=2, step_limit=10)
    policy = policy_impl.GreedyPolicyWithRandomness(epsilon=1.0)
    choices = []
    for _ in range(500):
      choices.append(env.GetChoiceFromAction(policy.Decide(
        env=env,
        brain=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=0,
        num_of_episodes=500)))

    # Tests that roughly half of the time action 0 is chosen.
    num_of_0s = len([c for c in choices if c == 0])
    self.assertGreater(num_of_0s, 200)
    self.assertLess(num_of_0s, 300)

  def test_GreedyPolicyWithDecreasingRandomness_epsilonDecays(self):
    mock_qfunc = mock.MagicMock()
    mock_qfunc.GetValues.return_value = numpy.array([[0.3, 0.7]])

    env = environment_impl.SingleStateEnvironment(
      action_space_size=2, step_limit=10)
    policy = policy_impl.GreedyPolicyWithDecreasingRandomness(
      initial_epsilon=1.0,
      final_epsilon=0.0,
      decay_by_half_after_num_of_episodes=50)

    # When episode_idx is 0, the action should be purely random.
    choices = []
    for _ in range(500):
      choices.append(env.GetChoiceFromAction(policy.Decide(
        env=env,
        brain=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=0,
        num_of_episodes=500)))

    # Tests that roughly half of the time action 0 is chosen.
    num_of_0s = len([c for c in choices if c == 0])
    self.assertGreater(num_of_0s, 200)
    self.assertLess(num_of_0s, 300)

    # When episode_idx is really large, the action should be close to optimal.
    choices = []
    for _ in range(500):
      choices.append(env.GetChoiceFromAction(policy.Decide(
        env=env,
        brain=mock_qfunc,
        state=numpy.array([[0]]),
        episode_idx=500,
        num_of_episodes=500)))

    # Tests that almost all the time the optimal action 1 is chosen.
    num_of_0s = len([c for c in choices if c == 0])
    self.assertLess(num_of_0s, 5)
