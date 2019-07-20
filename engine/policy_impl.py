"""Provides some policy implementations."""
import numpy

from deep_learning.engine import q_base


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
    return env.GetAction(int(numpy.argmax(qfunc.GetValues(state))))


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

  def Decide(
      self,
      env: q_base.Environment,
      qfunc: q_base.QFunction,
      state: q_base.State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> q_base.Action:
    if numpy.random.uniform(0, 1) < self._e:
      return env.GetAction(numpy.random.randint(0, env.GetStateArraySize()))
    else:
      return env.GetAction(int(numpy.argmax(qfunc.GetValues(state))))
