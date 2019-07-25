"""QFunction implementations."""
import keras
import numpy
from keras import layers
from keras import models
from keras import optimizers

from deep_learning.engine import q_base
from deep_learning.engine.q_base import QValues
from deep_learning.engine.q_base import States
from qpylib import t

_DEFAULT_ACTIVATION = 'relu'

_DEFAULT_TRAINING_BATCH_SIZE = 64


class RandomValueQFunction(q_base.QFunction):
  """QFunction that returns random value upon read, and write is no-op."""

  def __init__(
      self,
      action_space_size: int,
  ):
    """Constructor.

    Args:
      action_space_size: the size of the action space.

    """
    super().__init__(None, None)
    self._action_space_size = action_space_size

  def _protected_GetValues(self, states: States) -> QValues:
    return numpy.random.randint(
      0, self._action_space_size - 1,
      size=(len(states), self._action_space_size))

  def _protected_SetValues(self, states: States, values: QValues) -> None:
    """Writes has no effect."""
    pass


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
    return tuple(v.item() for v in numpy.nditer(state))

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
  """DQN implemented using Keras model."""

  def __init__(
      self,
      model: keras.Model,
      training_batch_size: object = _DEFAULT_TRAINING_BATCH_SIZE,
      discount_factor: object = None,
  ):
    """Constructor.

    Args:
      model: a compiled Keras model that powers the DQN interface. Its first
        layer's input_shape indicates the state shape, and its last layer's
        shape indicates the action space size. To be consistent with other
        classes in this package, the action space needs to be 1-dimensional.
      training_batch_size: the batch size used in training. When using
        experience replay runner, this size can be chosen to be the same
        as the experience sample size.
      discount_factor: gamma.
    """
    # DQN's learning is done via optimizer; no need to use a learning rate
    # at the Q-value iteration level.
    super().__init__(discount_factor, 1.0)
    self._model = model
    self._training_batch_size = training_batch_size

    self._state_shape = self._model.layers[0].input_shape[1:]
    output_shape = self._model.layers[-1].output_shape[1:]  # type: t.Tuple[int]
    if len(output_shape) != 1:
      raise NotImplementedError(
        'Only supports 1D action space; got: %s' % str(output_shape))
    self._action_space_size = output_shape[0]

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

  def SaveWeights(self, file_path: t.Text) -> None:
    """Saves the model's weights to a file."""
    self._model.save_weights(file_path)

  def LoadWeights(self, file_path: t.Text) -> None:
    """Loads a model's weights from a file saved by SaveWeights."""
    self._model.load_weights(file_path)

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


def CreateModel(
    state_shape: t.Tuple[int],
    action_space_size: int,
    hidden_layer_sizes: t.Iterable[int],
    activation: t.Text = _DEFAULT_ACTIVATION,
    optimizer: optimizers.Optimizer = None,
):
  """Creates a single head model.

  Following reference:
    https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

  Args:
    state_shape: the shape of the state ndarray.
    action_space_size: the size of the action space.
    hidden_layer_sizes: a list of number of nodes in the hidden layers,
      staring with the input layer.
    activation: the activation, for example "relu".
    optimizer: the optimizer to use. Default to the one created by
      _CreateDefaultOptimizer.
  """
  if optimizer is None:
    optimizer = _CreateDefaultOptimizer()

  hidden_layer_sizes = tuple(hidden_layer_sizes)
  model = models.Sequential()
  if len(state_shape) > 1:
    model.add(layers.Flatten(input_shape=state_shape))
    for num_nodes in hidden_layer_sizes:
      model.add(layers.Dense(units=num_nodes, activation=activation))
  else:
    model.add(
      layers.Dense(
        units=hidden_layer_sizes[0],
        activation=activation,
        input_dim=state_shape[0]))
    for num_nodes in hidden_layer_sizes[1:]:
      model.add(layers.Dense(units=num_nodes, activation=activation))
  model.add(layers.Dense(
    units=action_space_size, activation='linear'))

  model.compile(loss='mse', optimizer=optimizer)

  return model


def _CreateDefaultOptimizer() -> optimizers.Optimizer:
  """Creates a default optimizer."""
  # Ref:
  #   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
  return optimizers.RMSprop(lr=0.00025)
