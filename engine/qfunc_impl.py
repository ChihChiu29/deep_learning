"""QFunction implementations."""
import numpy

from deep_learning.engine import q_base
from qpylib import t

DEFAULT_RMSPROP_LEARNING_RATE = 0.00025


class MemoizationQFunction(q_base.QFunction):
  """QFunction that uses memoization."""

  def __init__(
      self,
      action_space_size: int,
      discount_factor: float = None,
      learning_rate: float = None,
  ):
    super().__init__(discount_factor, learning_rate)
    self._action_space_size = action_space_size

    self._storage = {}  # {state: value}.

  def _Key(self, state: numpy.ndarray) -> t.Hashable:
    """Gets a key for a "state".

    Note that the state is a 1d array instead of the 2d array as declared in
    q_base.py.
    """
    return tuple(v for v in state)

  def _protected_GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    qvalues = numpy.vstack([
      self._storage.get(self._Key(s), numpy.zeros((1, self._action_space_size)))
      for s in states])
    return qvalues

  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    for s, v in zip(states, values):
      self._storage[self._Key(s)] = v

#
# class DQN(q_base.QFunction):
#
#   def __init__(
#       self,
#       state_space_dim: int,
#       action_space_size: int,
#       rmsprop_learning_rate: float = None,
#       discount_factor: float = None,
#   ):
#     """Constructor.
#
#     Args:
#       state_space_dim: the dimension of the state space.
#       action_space_size: the size of the action space.
#       discount_factor: gamma.
#       rmsprop_learning_rate: the learning rate used by the RMSprop optimizer.
#     """
#     super().__init__(discount_factor, 1.0)
#     self._rmsprop_learning_rate = (
#       rmsprop_learning_rate if rmsprop_learning_rate is not None
#       else DEFAULT_RMSPROP_LEARNING_RATE)
#
#   def _protected_GetValues(self, states: States) -> QValues:
#     pass
#
#   def _protected_SetValues(self, states: States, values: QValues) -> None:
#     pass
#
#   def _BuildModel(self):
#     def _BuildModelIntAction(
#         state_array_size: int,
#         num_nodes_in_layers: Iterable[int],
# ) -> models.Model:
#     """Builds a model with the given info; using an int for action."""
#     input_size = state_array_size + 1
#     model = models.Sequential()
#     model.add(
#         layers.Dense(input_size, activation='relu', input_dim=input_size))
#     for num_nodes in num_nodes_in_layers:
#         model.add(layers.Dense(num_nodes, activation='relu'))
#     model.add(layers.Dense(1))
#
#     model.compile(optimizer='sgd', loss='mse')
#
#     return model
