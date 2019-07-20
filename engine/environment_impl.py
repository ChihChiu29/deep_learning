"""Provides some environment implementations."""
import numpy

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Action, Transition


class SingleStateEnvironment(q_base.Environment):
  """An environment of a single state."""

  def __init__(
      self,
      step_limit: int,
  ):
    """Constructor.

    Args:
      step_limit: the number of actions allowed to take before environment
        is "done".
    """
    super().__init__(1, 1)
    self._step_limit = step_limit

    self._action_count = None
    self.Reset()

    self._state = numpy.array([[0]])

  def TakeAction(self, action: Action) -> Transition:
    if self._action_count >= self._step_limit:
      sp = None
    else:
      sp = self._state

    self._action_count += 1
    return q_base.Transition(s=self._state, a=action, r=0.0, sp=sp)

  def Reset(self):
    self._action_count = 0
