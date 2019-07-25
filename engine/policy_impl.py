"""Provides some policy implementations."""
import numpy

from deep_learning.engine import q_base
from qpylib import logging


class GreedyPolicy(q_base.Policy):
  """A policy that always picks the action that gives the max Q value."""

  def Decide(
      self,
      env: q_base.Environment,
      qfunc: q_base.QFunction,
      state: q_base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> q_base.Action:
    values = qfunc.GetValues(state)
    choice = int(numpy.argmax(values))
    logging.vlog(
      15, 'making greedy decision for state %s using values: %s; choice: %d',
      state, values, choice)
    return env.GetActionFromChoice(choice)


class GreedyPolicyWithRandomness(q_base.Policy):
  """A policy that almost always the action that gives the max Q value."""

  def __init__(
      self,
      epsilon: float,
  ):
    """Constructor.

    Args:
      epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen.
    """
    self._e = epsilon

    self._greedy_policy = GreedyPolicy()

  def Decide(
      self,
      env: q_base.Environment,
      qfunc: q_base.QFunction,
      state: q_base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> q_base.Action:
    if numpy.random.uniform(0, 1) < self._e:
      choice = env.GetRandomChoice()
      logging.vlog(
        7, 'making random decision for state %s choice: %d', state, choice)
      return env.GetActionFromChoice(choice)
    else:
      return self._greedy_policy.Decide(
        env=env,
        qfunc=qfunc,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
      )


class GreedyPolicyWithDecreasingRandomness(q_base.Policy):
  """A policy that almost always the action that gives the max Q value.

  The possibility that it does not perform the best action decreases over steps.
  """

  def __init__(
      self,
      initial_epsilon: float,
      final_epsilon: float,
      decay_factor: float,
  ):
    """Constructor.

    Args:
      initial_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen initially.
      final_epsilon: a number between 0 and 1. It gives the probability that a
        random non-greedy action is chosen eventually.
      decay_factor: any float number. The epsilon that is actually used
        is calculated as:
          e = final_e + (initial_e - final_e) * exp(-decay_factor * episodes)
    """
    self._initial_epsilon = initial_epsilon
    self._final_epsilon = final_epsilon
    self._decay_factor = decay_factor

    self._num_of_steps = 0
    self._greedy_policy = GreedyPolicy()

  def Decide(
      self,
      env: q_base.Environment,
      qfunc: q_base.QFunction,
      state: q_base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> q_base.Action:
    # Calculating e using exp instead of calculating it incrementally because
    # there is no guarantee if the policy is used in a new run.
    e = (
        self._final_epsilon +
        (self._initial_epsilon - self._final_epsilon) * numpy.exp(
      - self._decay_factor * episode_idx))
    if numpy.random.uniform(0, 1) < e:
      choice = env.GetRandomChoice()
      logging.vlog(
        7, 'making random decision (current e: %f) for state %s choice: %d',
        e, state, choice)
      return env.GetActionFromChoice(choice)
    else:
      return self._greedy_policy.Decide(
        env=env,
        qfunc=qfunc,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
      )
