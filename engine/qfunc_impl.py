"""QFunction implementations."""
import numpy
from keras import models, layers, optimizers

from deep_learning.engine import q_base
from qpylib import t


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

  # @Override
  def _protected_GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    qvalues = numpy.vstack([
      self._storage.get(self._Key(s), numpy.zeros((1, self._action_space_size)))
      for s in states])
    return qvalues

  # @Override
  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    for s, v in zip(states, values):
      self._storage[self._Key(s)] = v


class DQN(q_base.QFunction):
  # Ref:
  #   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
  #   https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a
  DEFAULT_RMSPROP_LEARNING_RATE = 0.00025

  def __init__(
      self,
      state_space_dim: int,
      action_space_size: int,
      hidden_layer_sizes: t.Iterable[int],
      training_batch_size: int,
      rmsprop_learning_rate: float = None,
      discount_factor: float = None,
  ):
    """Constructor.

    Args:
      state_space_dim: the dimension of the state space.
      action_space_size: the size of the action space.
      hidden_layer_sizes: a list of number of nodes in the hidden layers,
        staring with the input layer.
      training_batch_size: the batch size used in training. When using
        experience replay runner, this size can be chosen to be the same
        as the experience sample size.
      discount_factor: gamma.
      rmsprop_learning_rate: the learning rate used by the RMSprop optimizer.
    """
    super().__init__(discount_factor, 1.0)
    self._state_space_dim = state_space_dim
    self._action_space_size = action_space_size
    self._hidden_layer_sizes = tuple(hidden_layer_sizes)
    self._training_batch_size = training_batch_size
    self._rmsprop_learning_rate = (
      rmsprop_learning_rate if rmsprop_learning_rate is not None
      else self.DEFAULT_RMSPROP_LEARNING_RATE)

    self._model = self._BuildModel()

  # @Override
  def _protected_GetValues(
      self,
      states: q_base.States,
  ) -> q_base.QValues:
    return self._model.predict(states)

  # @Override
  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.QValues,
  ) -> None:
    return self._model.fit(
      states, values, batch_size=self._training_batch_size, verbose=0)

  def _BuildModel(self) -> models.Model:
    """Builds a model that stores the Q values.

    Number of nodes in each layer is specified by hidden_layer_sizes in the
    constructor. The first layer has the input_dim equals to the state space
    dimension, and the last layer has the output dimension equals to the
    action space dimension.
    """
    activation = 'relu'
    model = models.Sequential()
    model.add(
      layers.Dense(
        units=self._hidden_layer_sizes[0],
        activation=activation,
        input_dim=self._state_space_dim))
    for num_nodes in self._hidden_layer_sizes[1:]:
      model.add(layers.Dense(units=num_nodes, activation=activation))
    model.add(layers.Dense(
      units=self._action_space_size, activation='linear'))

    opt = optimizers.RMSprop(lr=self._rmsprop_learning_rate)
    model.compile(loss='mse', optimizer=opt)
    return model

  def SaveModel(self, file_path: t.Text) -> None:
    """Saves the model and other stats to a file.

    See:
      https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    """
    self._model.save(file_path)

  def LoadModel(self, file_path: t.Text) -> None:
    """Loads a model from a file saved by SaveModel.

    Note that arguments to the constructor of this class is not saved. The
    saved model can only be loaded by an instance whose models has the
    identical structure.

    See:
      https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    """
    self._model = models.load_model(file_path)
