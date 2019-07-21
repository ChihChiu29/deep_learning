"""Tests that implementations in engine package are runnable."""

import unittest

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


class RunnerTest(unittest.TestCase):

  def setUp(self) -> None:
    self.env = environment_impl.SingleStateEnvironment(
      action_space_size=1, step_limit=10)
    self.qfunc = qfunc_impl.MemoizationQFunction(
      action_space_size=3,
      discount_factor=0.9,
      learning_rate=0.9)
    self.policy = policy_impl.GreedyPolicy()

  def test_simpleRunner(self):
    # Tests that it can run; quality if not important for this test.
    runner_impl.SimpleRunner().Run(
      env=self.env, qfunc=self.qfunc, policy=self.policy, num_of_episodes=1)

  def test_experienceReplayRunner(self):
    # Tests that it can run; quality if not important for this test.
    runner_impl.ExperienceReplayRunner(
      experience_capacity=100,
      experience_sample_batch_size=10).Run(
      env=self.env, qfunc=self.qfunc, policy=self.policy, num_of_episodes=1)


class PolicyTest(unittest.TestCase):

  def setUp(self) -> None:
    self.env = environment_impl.SingleStateEnvironment(
      action_space_size=1, step_limit=10)
    self.qfunc = qfunc_impl.MemoizationQFunction(
      action_space_size=3,
      discount_factor=0.9,
      learning_rate=0.9)
    self.runner = runner_impl.SimpleRunner()

  def test_GreedyPolicy(self):
    # Tests that it can run; quality if not important for this test.
    self.runner.Run(
      env=self.env, qfunc=self.qfunc,
      policy=policy_impl.GreedyPolicy(), num_of_episodes=1)

  def test_GreedyPolicyWithRandomness(self):
    # Tests that it can run; quality if not important for this test.
    self.runner.Run(
      env=self.env, qfunc=self.qfunc,
      policy=policy_impl.GreedyPolicyWithRandomness(epsilon=0.1),
      num_of_episodes=1)
