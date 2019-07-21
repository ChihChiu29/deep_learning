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
      7, 'making greedy decision for state %s using values: %s; choice: %d',
      state, values, choice)
    return env.GetAction(choice)


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
      choice = numpy.random.randint(0, env.GetActionSpaceSize())
      logging.vlog(
        7, 'making random decision for state %s choice: %d', state, choice)
      return env.GetAction(choice)
    else:
      return self._greedy_policy.Decide(
        env=env,
        qfunc=qfunc,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
      )
