"""QFunction implementations."""
import numpy

from deep_learning.engine import q_base
from qpylib import t, parameters


class MemoizationQFunction(q_base.QFunction):
  """QFunction that uses memoization."""

  def __init__(self, action_space_size: int):
    self._action_space_size = action_space_size

    self._storage = {}  # {state: value}.

  def _Key(self, state: q_base.State) -> t.Text:
    return str(state)

  def GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    qvalues = numpy.vstack([
      self._storage.get(self._Key(s), numpy.zeros((1, self._action_space_size)))
      for s in states])
    if parameters.ENV.debug_verbosity > 7:
      print('GET: (%s) -> %s' % (states, qvalues))
    return qvalues

  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    for s, v in zip(states, values):
      self._storage[self._Key(s)] = v
      if parameters.ENV.debug_verbosity > 7:
        print('SET: (%s) <- %s' % (s, v))
