"""Implements A3C.

Ref:
  https://github.com/jaromiru/AI-blog/blob/master/CartPole-A3C.py
  https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
"""
import keras
import tensorflow
from keras import backend
from keras import layers

from deep_learning.engine import base
from deep_learning.engine.base import Values
from qpylib import t

_DEFAULT_DISCOUNT_FACTOR = 0.99
_DEFAULT_LOSS_V = .5  # v loss coefficient
_DEFAULT_LOSS_ENTROPY = .01  # entropy coefficient


class A3C(base.Brain):
  """A A3C brain."""

  def __init__(
      self,
      model: keras.Model,
      optimizer: tensorflow.train.Optimizer = None,
      discount_factor: float = _DEFAULT_DISCOUNT_FACTOR,
      loss_v: float = _DEFAULT_LOSS_V,
      loss_entropy: float = _DEFAULT_LOSS_ENTROPY,
  ):
    """Ctor.

    Args:
      model: a model that
    """
    self._model = model
    self._optimizer = optimizer if optimizer else CreateDefaultOptimizer()
    self._gamma = discount_factor
    self._loss_v = loss_v
    self._loss_entropy = loss_entropy

    self._state_shape = self._model.layers[0].input_shape[1:]
    output_shape = self._model.layers[-2].output_shape[1:]  # type: t.Tuple[int]
    if len(output_shape) != 1:
      raise NotImplementedError(
        'Only supports 1D action space; got: %s' % str(output_shape))
    self._action_space_size = output_shape[0]

    self._graph = self._BuildGraph(self._model)

    self.session = tensorflow.Session()
    backend.set_session(self.session)
    backend.manual_variable_initialization(True)
    self.session.run(tensorflow.global_variables_initializer())
    self.default_graph = tensorflow.get_default_graph()
    self.default_graph.finalize()  # avoid modifications

  # @Override
  def GetValues(
      self,
      states: base.States,
  ) -> Values:
    pi_values, v = self._model.predict(states)
    return pi_values

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[base.Transition],
  ) -> None:
    states, actions, rewards, new_states, reward_mask = (
      self.CombineTransitions(transitions))

    pi_values, values = self._model.predict(states)
    rewards = rewards + self._gamma * values * reward_mask

    s_input, a_input, r_input, minimize = self._graph
    self.session.run(
      minimize, feed_dict={s_input: states, a_input: actions, r_input: rewards})

  # @Override
  def Save(self, filepath: t.Text) -> None:
    self._model.save_weights(filepath)

  # @Override
  def Load(self, filepath: t.Text) -> None:
    self._model.load_weights(filepath)

  def _BuildGraph(self, model):
    s = tensorflow.placeholder(tensorflow.float32, shape=self._state_shape)
    a = tensorflow.placeholder(
      tensorflow.float32, shape=(None, self._action_space_size))
    r = tensorflow.placeholder(tensorflow.float32, shape=(None, 1))

    pi_values, v = model(s)

    log_prob = tensorflow.log(
      tensorflow.reduce_sum(pi_values * a, axis=1, keep_dims=True) + 1e-10)
    advantage = r - v

    # maximize policy
    loss_policy = - log_prob * tensorflow.stop_gradient(advantage)
    # minimize value error
    loss_value = self._loss_v * tensorflow.square(advantage)
    # maximize entropy (regularization)
    entropy = self._loss_entropy * tensorflow.reduce_sum(
      pi_values * tensorflow.log(pi_values + 1e-10), axis=1,
      keep_dims=True)

    loss_total = tensorflow.reduce_mean(loss_policy + loss_value + entropy)
    minimize = self._optimizer.minimize(loss_total)
    return s, a, r, minimize


def CreateModel(
    state_shape: t.Sequence[int],
    action_space_size: int,
    hidden_layer_sizes: t.Iterable[int],
    activation: t.Text = 'relu',
):
  """Builds a model for A3C.

  Args:
    state_shape: the shape of the state ndarray.
    action_space_size: the size of the action space.
    hidden_layer_sizes: a list of number of nodes in the hidden layers,
      staring with the input layer.
    activation: the activation, for example "relu".
  """
  hidden_layer_sizes = tuple(hidden_layer_sizes)

  if len(state_shape) > 1:
    l_input = layers.Flatten(input_shape=state_shape)
    l = l_input
    for num_nodes in hidden_layer_sizes:
      l = layers.Dense(units=num_nodes, activation=activation)(l)
  else:
    l_input = layers.Dense(
      units=hidden_layer_sizes[0],
      activation=activation,
      input_dim=state_shape[0])
    l = l_input
    for num_nodes in hidden_layer_sizes[1:]:
      l = layers.Dense(units=num_nodes, activation=activation)(l)

  out_pi = layers.Dense(action_space_size, activation='softmax')(l)
  out_v = layers.Dense(1, activation='linear')(l)

  model = keras.Model(inputs=[l_input], outputs=[out_pi, out_v])

  return model


def CreateDefaultOptimizer() -> tensorflow.train.Optimizer:
  """Creates a default optimizer."""
  return tensorflow.train.RMSPropOptimizer(5e-3, decay=.99)
