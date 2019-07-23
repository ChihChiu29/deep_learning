"""QFunction implementations with models that have a head for each action."""
from deep_learning.engine import q_base
from deep_learning.engine.q_base import States, QValues


# TODO: Continue after DQN base class is implemented, after state refactoring
# is done.
class MultiHeadQFunction(q_base.QFunction):

  def _protected_GetValues(self, states: States) -> QValues:
    pass

  def _protected_SetValues(self, states: States, values: QValues) -> None:
    pass
