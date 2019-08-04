"""QFunction implementations."""
import pickle

import keras
import numpy
from keras import layers
from keras import models
from keras import optimizers

from deep_learning.engine import q_base
from deep_learning.engine.q_base import States
from deep_learning.engine.q_base import Values
from qpylib import numpy_util
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

  # @Override
  def Save(self, filepath: t.Text) -> None:
    pass

  # @Override
  def Load(self, filepath: t.Text) -> None:
    pass

  # @Override
  def _protected_GetValues(self, states: States) -> Values:
    return numpy.random.randint(
      0, self._action_space_size - 1,
      size=(len(states), self._action_space_size))

  # @Override
  def _protected_SetValues(self, states: States, values: Values) -> None:
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

  # @Override
  def Save(self, filepath: t.Text) -> None:
    pickle.dump(self._storage, open(filepath, 'wb'))

  # @Override
  def Load(self, filepath: t.Text) -> None:
    self._storage = pickle.load(open(filepath, 'rb'))

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
  ) -> q_base.Values:
    qvalues = numpy.vstack([
      self._storage.get(self._Key(s), numpy.zeros((1, self._action_space_size)))
      for s in states])
    return qvalues

  # @Override
  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.Values,
  ) -> None:
    for s, v in zip(states, values):
      self._storage[self._Key(s)] = v


class DQN(q_base.QFunction):
  """DQN implemented using a Keras model."""

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
  def Save(self, filepath: t.Text) -> None:
    """Saves the model's weights to a file.

    Note that in order to load the saved weights, you need to create a model
    of exactly the same shape. Also the optimizer info is lost during this
    action.
    """
    self._model.save_weights(filepath)

  # @Override
  def Load(self, filepath: t.Text) -> None:
    """Loads a model's weights from a file saved by Save."""
    self._model.load_weights(filepath)

  # @Override
  def _protected_GetValues(
      self,
      states: q_base.States,
  ) -> q_base.Values:
    return self._model.predict(states)

  # @Override
  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.Values,
  ) -> None:
    self._model.fit(
      states, values, batch_size=self._training_batch_size, verbose=0)


class DQN_TargetNetwork(DQN):
  """DQN implemented using Keras model; target networks is used.

  Note that it's slower than DQN to process each step, and in terms of
  getting to convergence, but it's more stable than DQN.

  For target network see:
  https://jaromiru.com/2016/10/21/lets-make-a-dqn-full-dqn/
  """

  def __init__(
      self,
      model: keras.Model,
      update_target_network_every_num_of_steps: int = 1000,
      training_batch_size: object = _DEFAULT_TRAINING_BATCH_SIZE,
      discount_factor: object = None,
  ):
    """Constructor.

    Args:
      model: a compiled Keras model that powers the DQN interface. Its first
        layer's input_shape indicates the state shape, and its last layer's
        shape indicates the action space size. To be consistent with other
        classes in this package, the action space needs to be 1-dimensional.
      update_target_network_every_num_of_steps: the target network is updated
        every this number of steps.
      training_batch_size: the batch size used in training. When using
        experience replay runner, this size can be chosen to be the same
        as the experience sample size.
      discount_factor: gamma.
    """
    super().__init__(model, training_batch_size, discount_factor)
    self._update_target_network_every_num_of_steps = (
      update_target_network_every_num_of_steps)

    # Target network is only used for reading.
    self._target_network = models.clone_model(self._model)
    self._step_count = 0

  def _CopyWeightsToTargetNetwork(self):
    self._target_network.set_weights(self._model.get_weights())

  # @Override
  def Load(self, filepath: t.Text) -> None:
    """Loads a model's weights from a file saved by Save.

    The weights are immediately set to the target network as well since
    the weights in the target network is not saved by Save.
    """
    super().Load(filepath)
    self._CopyWeightsToTargetNetwork()

  # @Override
  def _protected_GetValues(
      self,
      states: q_base.States,
  ) -> q_base.Values:
    return self._target_network.predict(states)

  # @Override
  def _protected_SetValues(
      self,
      states: q_base.States,
      values: q_base.Values,
  ) -> None:
    super()._protected_SetValues(states, values)
    self._step_count += 1
    if self._step_count >= self._update_target_network_every_num_of_steps:
      self._CopyWeightsToTargetNetwork()
      self._step_count = 0


class DDQN(q_base.QFunction):
  """Double DQN.

  It's slower than DQN_TargetNetwork to process each step, and in terms
  of getting to convergence, but it's more stable than DQN_TargetNetwork.

  Reference:
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and
    -prioritized-experience-replay/
  """

  def __init__(
      self,
      model_pair: t.Tuple[keras.Model, keras.Model],
      training_batch_size: object = _DEFAULT_TRAINING_BATCH_SIZE,
      discount_factor: object = None,
  ):
    """Ctor.

    Args:
      model_pair: two compiled models that have identical shape.
      training_batch_size: the batch size used in training.
      discount_factor: gamma
    """
    # The Q-value iteration from q_base.QFunction will not be called at all.
    super().__init__(None, None)

    self._model, self._secondary_model = model_pair
    self._training_batch_size = training_batch_size
    self._gamma = (
      discount_factor if discount_factor is not None else
      q_base.DEFAULT_DISCOUNT_FACTOR)

    self._state_shape = self._model.layers[0].input_shape[1:]
    output_shape = self._model.layers[-1].output_shape[1:]  # type: t.Tuple[int]
    if len(output_shape) != 1:
      raise NotImplementedError(
        'Only supports 1D action space; got: %s' % str(output_shape))
    self._action_space_size = output_shape[0]

    # These will swap during the iterations. All value manipulations use these
    # indirect references.
    self._q1 = self._model
    self._q2 = self._secondary_model

  # @Override
  def Save(self, filepath: t.Text) -> None:
    self._q1.save_weights(filepath)

  # @Override
  def Load(self, filepath: t.Text) -> None:
    self._q1.load_weights(filepath)
    self._q2.load_weights(filepath)

  # @Override
  def _protected_GetValues(self, states: States) -> Values:
    return self._q1.predict(states)

  # @Override
  def _protected_SetValues(self, states: States, values: Values) -> None:
    self._q1.fit(
      states, values, batch_size=self._training_batch_size, verbose=0)

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[q_base.Transition],
  ) -> t.Tuple[q_base.States, q_base.Actions, q_base.ActionValues]:
    """Update Q-values using the given set of transitions.

    The iteration equation only calls interface methods to get values from
    self._q1. Reading from self._q2 is done directly.

    Args:
      transitions: an iterable of transitions to update values from.

    Returns:
      The target action values to update to.
    """
    states, actions, rewards, new_states, reward_mask = (
      self.CombineTransitions(transitions))

    q1_values = self.GetValues(new_states)
    q1_action_choices = numpy.argmax(q1_values, axis=1)
    q1_actions = numpy_util.EncodePositionsToOneHotArray(
      q1_action_choices, self._action_space_size)
    q2_values = self._q2.predict(new_states)
    new_action_values = self.GetActionValues(q2_values, q1_actions)
    learn_new_action_values = (
        rewards + self._gamma * new_action_values * reward_mask)
    target_action_values = self._SetActionValues(
      states, actions, learn_new_action_values)

    self._SwapModels()

    return target_action_values

  def _SwapModels(self):
    tmp = self._q1
    self._q1 = self._q2
    self._q2 = tmp


def CreateModel(
    state_shape: t.Sequence[int],
    action_space_size: int,
    hidden_layer_sizes: t.Iterable[int],
    activation: t.Text = _DEFAULT_ACTIVATION,
    optimizer: optimizers.Optimizer = None,
) -> keras.Model:
  """Creates a single head model.

  Following reference:
    https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

  Args:
    state_shape: the shape of the state ndarray.
    action_space_size: the size of the action space.
    hidden_layer_sizes: a list of number of nodes in the hidden layers,
      staring with the input layer.
    activation: the activation, for examples "relu".
    optimizer: the optimizer to use. Default to the one created by
      _CreateDefaultOptimizer.
  """
  if optimizer is None:
    optimizer = CreateDefaultOptimizer()

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


def CreateDefaultOptimizer() -> optimizers.Optimizer:
  """Creates a default optimizer."""
  # Ref:
  #   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
  return optimizers.RMSprop(lr=0.00025)
