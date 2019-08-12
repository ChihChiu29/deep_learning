"""Provides some policy implementations."""
import numpy

from deep_learning.engine import base
from qpylib import logging


class GreedyPolicy(base.Policy):
  """A policy that always picks the action that gives the max value."""

  def Decide(
      self,
      env: base.Environment,
      brain: base.Brain,
      state: base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> base.Action:
    values = brain.GetValues(state)
    choice = int(numpy.argmax(values))
    logging.vlog(
      20, 'making greedy decision for state %s using values: %s; choice: %d',
      state, values, choice)
    return env.GetActionFromChoice(choice)


class PolicyWithRandomness(base.Policy):
  """A policy that almost performs like another policy."""

  def __init__(
      self,
      base_policy: base.Policy,
      epsilon: float,
  ):
    """Constructor.

    Args:
      base_policy: the policy that decides the deterministic behavior.
      epsilon: a number between 0 and 1. It gives the probability that a
        random action is chosen instead of using the one decided by base_policy.
    """
    self._base_policy = base_policy
    self._e = epsilon

  def Decide(
      self,
      env: base.Environment,
      brain: base.Brain,
      state: base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> base.Action:
    if numpy.random.uniform(0, 1) < self._e:
      choice = env.GetRandomChoice()
      logging.vlog(
        20, 'making random decision for state %s choice: %d', state, choice)
      return env.GetActionFromChoice(choice)
    else:
      return self._base_policy.Decide(
        env=env,
        brain=brain,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
      )


class PolicyWithDecreasingRandomness(base.Policy):
  """A policy that has decaying of randomness that does not use base policy.

  The possibility that it does not perform the best action decreases
  exponentially over episodes.
  """

  def __init__(
      self,
      base_policy: base.Policy,
      initial_epsilon: float,
      final_epsilon: float,
      decay_by_half_after_num_of_episodes: int,
  ):
    """Constructor.

    Args:
      base_policy: the policy that decides the deterministic behavior.
      initial_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen initially.
      final_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen eventually.
      decay_by_half_after_num_of_episodes: e always decays by half from
        initial_e to final_e after this number of episodes.
    """
    self._base_policy = base_policy
    self._init_e = initial_epsilon
    self._final_e = final_epsilon
    self._decay_half_life = decay_by_half_after_num_of_episodes

  def Decide(
      self,
      env: base.Environment,
      brain: base.Brain,
      state: base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> base.Action:
    factor = numpy.log(0.5) / self._decay_half_life
    e = (self._final_e + (self._init_e - self._final_e) * numpy.exp(
      factor * episode_idx))
    if numpy.random.uniform(0, 1) < e:
      choice = env.GetRandomChoice()
      logging.vlog(
        20, 'making random decision (current e: %f) for state %s choice: %d',
        e, state, choice)
      return env.GetActionFromChoice(choice)
    else:
      return self._base_policy.Decide(
        env=env,
        brain=brain,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
      )


class GreedyPolicyWithRandomness(PolicyWithRandomness):
  """A policy that almost always the action that gives the max value."""

  def __init__(
      self,
      epsilon: float,
  ):
    """Constructor.

    Args:
      epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen.
    """
    super().__init__(GreedyPolicy(), epsilon)


class GreedyPolicyWithDecreasingRandomness(PolicyWithDecreasingRandomness):
  """A policy that has decaying of randomness that's not greedy.

  The possibility that it does not perform the best action decreases
  exponentially over episodes.
  """

  def __init__(
      self,
      initial_epsilon: float,
      final_epsilon: float,
      decay_by_half_after_num_of_episodes: int,
  ):
    """Constructor.

    Args:
      initial_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen initially.
      final_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen eventually.
      decay_by_half_after_num_of_episodes: e always decays by half from
        initial_e to final_e after this number of episodes.
    """
    super().__init__(
      GreedyPolicy(), initial_epsilon, final_epsilon,
      decay_by_half_after_num_of_episodes)
