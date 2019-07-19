"""QFunction implementations."""
from deep_learning.engine import q_base


class MemoizationQFunction(q_base.QFunction):
  """QFunction that uses memoization."""

  def __init__(self):
    self._storage = {}  # {state: value}.
    
  def GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    pass

  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    pass
