"""QFunction implementations."""
import numpy

from deep_learning.engine import q_base
from qpylib import t


class MemoizationQFunction(q_base.QFunction):
  """QFunction that uses memoization."""

  def __init__(self, state_space_size: int):
    self._state_space_size = state_space_size

    self._storage = {}  # {state: value}.

  def _Key(self, state: q_base.State) -> t.Text:
    return str(state)

  def GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    return numpy.vstack([
      self._storage.get(self._Key(s), numpy.zeros((1, self._state_space_size)))
      for s in states])

  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    for s, v in zip(states, values):
      self._storage[self._Key(s)] = v
