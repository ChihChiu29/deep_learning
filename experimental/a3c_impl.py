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
from deep_learning.engine.base import Brain
from deep_learning.engine.base import Transition
from deep_learning.engine.base import Values
from qpylib import logging
from qpylib import t

_DEFAULT_DISCOUNT_FACTOR = 0.99
_DEFAULT_LOSS_V = .5  # v loss coefficient
_DEFAULT_LOSS_ENTROPY = .01  # entropy coefficient

# Stores active instances.
_ACTIVE_INSTANCES = []  # type: t.List['A3C']


# Not ready for mass usage:
#   * Interval World: always stable.
#   * Circular World: use SimpleRunner, and a Greedy policy with epsilon=0.1
#     works with a (20, 20, 20) model.
#   * CartPole-v0: doesn't work.
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
    if _ACTIVE_INSTANCES:
      instance = _ACTIVE_INSTANCES[0]
      logging.printf(
        'WARNING: only one A3C instance can be active; the previous instance '
        '%s is now deactivated.',
        instance)
      instance.Deactivate()
      _ACTIVE_INSTANCES.pop()

    self._model = model
    self._optimizer = optimizer if optimizer else CreateDefaultOptimizer()
    self._gamma = discount_factor
    self._loss_v = loss_v
    self._loss_entropy = loss_entropy

    self._state_batch_shape = self._model.layers[0].input_shape
    # Layer -1 is the output for V, -2 is for the values of Pi.
    output_shape = self._model.layers[-2].output_shape[1:]  # type: t.Tuple[int]
    if len(output_shape) != 1:
      raise NotImplementedError(
        'Only supports 1D action space; got: %s' % str(output_shape))
    self._action_space_size = output_shape[0]

    self._graph = self._BuildGraph(self._model)

    self.session = tensorflow.Session()
    backend.set_session(self.session)
    self.session.run(tensorflow.global_variables_initializer())

    # Only one A3C instance can be active at a time.
    self._active = True
    _ACTIVE_INSTANCES.append(self)

  def Deactivate(self):
    """Deactivates this instance."""
    self._active = False

  def _CheckActive(self):
    """Checks if this instance is active."""
    if self._active:
      return
    else:
      raise RuntimeError('Instance %s is no longer active.' % self)

  # @Override
  def GetValues(
      self,
      states: base.States,
  ) -> Values:
    """Use Pi values to make decision."""
    self._CheckActive()
    pi_values, v = self._model.predict(states)
    logging.vlog(20, 'GET pi for state %s: %s', states, pi_values)
    return pi_values

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[base.Transition],
  ) -> None:
    self._CheckActive()
    states, actions, rewards, new_states, reward_mask = (
      self.CombineTransitions(transitions))

    v_values = self._GetV(states)
    rewards = rewards + self._gamma * v_values * reward_mask

    s_input, a_input, r_input, minimize = self._graph
    self.session.run(
      minimize, feed_dict={s_input: states, a_input: actions, r_input: rewards})

  # @Override
  def Save(self, filepath: t.Text) -> None:
    self._CheckActive()
    self._model.save_weights(filepath)

  # @Override
  def Load(self, filepath: t.Text) -> None:
    self._CheckActive()
    self._model.load_weights(filepath)

  def _BuildGraph(self, model):
    s = tensorflow.placeholder(
      tensorflow.float32, shape=self._state_batch_shape)
    a = tensorflow.placeholder(
      tensorflow.float32, shape=(None, self._action_space_size))
    r = tensorflow.placeholder(tensorflow.float32, shape=(None,))

    pi_values, v = model(s)

    log_prob = tensorflow.log(
      tensorflow.reduce_sum(pi_values * a, axis=1, keepdims=True) + 1e-10)
    advantage = r - v

    # maximize policy
    loss_policy = - log_prob * tensorflow.stop_gradient(advantage)
    # minimize value error
    loss_value = self._loss_v * tensorflow.square(advantage)
    # maximize entropy (regularization)
    entropy = self._loss_entropy * tensorflow.reduce_sum(
      pi_values * tensorflow.log(pi_values + 1e-10), axis=1,
      keepdims=True)

    loss_total = tensorflow.reduce_mean(loss_policy + loss_value + entropy)
    minimize = self._optimizer.minimize(loss_total)
    return s, a, r, minimize

  def _GetV(self, states: base.States) -> base.OneDArray:
    _, values = self._model.predict(states)
    return values.flatten()


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
  l_input = layers.Input(batch_shape=(None, state_shape[0]))
  l_dense = layers.Dense(16, activation='relu')(l_input)

  out_actions = layers.Dense(action_space_size, activation='softmax')(l_dense)
  out_value = layers.Dense(1, activation='linear')(l_dense)

  model = keras.Model(inputs=[l_input], outputs=[out_actions, out_value])
  model._make_predict_function()  # have to initialize before threading

  return model

  hidden_layer_sizes = tuple(hidden_layer_sizes)
  input_layer = layers.Input(shape=state_shape)

  if len(state_shape) > 1:
    l = layers.Flatten()(input_layer)
    for num_nodes in hidden_layer_sizes:
      l = layers.Dense(units=num_nodes, activation=activation)(l)
  else:
    l = layers.Dense(
      units=hidden_layer_sizes[0], activation=activation)(input_layer)
    for num_nodes in hidden_layer_sizes[1:]:
      l = layers.Dense(units=num_nodes, activation=activation)(l)

  out_pi = layers.Dense(action_space_size, activation='softmax')(l)
  out_v = layers.Dense(1, activation='linear')(l)

  model = keras.Model(inputs=[input_layer], outputs=[out_pi, out_v])
  return model


def CreateDefaultOptimizer(
    learning_rate: float = 5e-3,
) -> tensorflow.train.Optimizer:
  """Creates a default optimizer."""
  return tensorflow.train.RMSPropOptimizer(learning_rate, decay=.99)


class NStepExperienceRunner(base.Runner):
  """A runner the uses n-step experience."""

  def __init__(
      self,
      discount_factor: float = _DEFAULT_DISCOUNT_FACTOR,
      n_step_return: int = 8,
  ):
    """Ctor.

    Args:
      discount_factor: the discount factor gamma.
      n_step_return: use this n-step-return.
    """
    super().__init__()
    self._gamma = discount_factor
    self._n_step_return = n_step_return

    self._inv_gamma_n = 1.0 / self._gamma

    # Stores the last n_step_return transitions.
    self._memory = []  # type: t.List[base.Transition]

  def _protected_ProcessTransition(
      self,
      brain: Brain,
      transition: Transition,
      step_idx: int,
  ) -> None:
    train_transitions = []
    self._memory.append(transition)
    train_transitions.append(self._GetNStepTransition())

    if len(self._memory) >= self._n_step_return:
      self._memory.pop(0)

    if transition.sp is None:
      while self._memory:
        train_transitions.append(self._GetNStepTransition())
        self._memory.pop(0)

    brain.UpdateFromTransitions(train_transitions)

  def _GetNStepTransition(self) -> base.Transition:
    # TODO: can be optimized.
    R = 0.0
    next_discount_factor = 1.0
    for tran in self._memory:
      R += tran.r * next_discount_factor
      next_discount_factor *= self._gamma
    return base.Transition(
      s=self._memory[0].s,
      a=self._memory[0].a,
      r=R,
      sp=self._memory[-1].sp,
    )
