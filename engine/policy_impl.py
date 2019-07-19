"""Provides some policy implementations."""
import numpy

from deep_learning.engine import q_base


class GreedyPolicy(q_base.Policy):

  def Decide(
      self,
      env: q_base.Environment,
      q_function: q_base.QFunction,
      state: q_base.State,
  ) -> q_base.Action:
    return env.GetAction(numpy.argmax(q_function.GetValues(state)))
