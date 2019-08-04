"""Implements A3C.

Ref:
  https://github.com/jaromiru/AI-blog/blob/master/CartPole-A3C.py
  https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
"""
import keras
import tensorflow
from keras import backend
from keras import layers

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Values
from qpylib import t


class A3C(q_base.Brain):
  """A A3C brain."""

  def __init__(
      self,
      model: keras.Model,

  ):
    """Ctor.

    Args:
      model: a model that
    """
    self._model = model

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
      states: q_base.States,
  ) -> Values:
    pass

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[q_base.Transition],
  ) -> t.Tuple[q_base.States, q_base.Actions, q_base.ActionValues]:
    pass

  # @Override
  def Save(self, filepath: t.Text) -> None:
    pass

  # @Override
  def Load(self, filepath: t.Text) -> None:
    pass

  def _BuildGraph(self, model):
    s_t = tensorflow.placeholder(tensorflow.float32, shape=self._state_shape)
    a_t = tensorflow.placeholder(
      tensorflow.float32, shape=(None, self._action_space_size))
    r_t = tensorflow.placeholder(tensorflow.float32, shape=(None, 1))

    p, v = model(s_t)

    log_prob = tensorflow.log(
      tensorflow.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    advantage = r_t - v

    loss_policy = - log_prob * tensorflow.stop_gradient(
      advantage)  # maximize policy
    loss_value = LOSS_V * tensorflow.square(advantage)  # minimize value error
    entropy = LOSS_ENTROPY * tensorflow.reduce_sum(
      p * tensorflow.log(p + 1e-10), axis=1,
      keep_dims=True)  # maximize entropy (regularization)

    loss_total = tensorflow.reduce_mean(loss_policy + loss_value + entropy)

    optimizer = tensorflow.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
    minimize = optimizer.minimize(loss_total)

    return s_t, a_t, r_t, minimize

  def optimize(self):
    if len(self.train_queue[0]) < MIN_BATCH:
      time.sleep(0)  # yield
      return

    with self.lock_queue:
      if len(self.train_queue[
               0]) < MIN_BATCH:  # more thread could have passed without lock
        return  # we can't yield inside lock

      s, a, r, s_, s_mask = self.train_queue
      self.train_queue = [[], [], [], [], []]

    s = np.vstack(s)
    a = np.vstack(a)
    r = np.vstack(r)
    s_ = np.vstack(s_)
    s_mask = np.vstack(s_mask)

    if len(s) > 5 * MIN_BATCH: print(
      "Optimizer alert! Minimizing batch of %d" % len(s))

    v = self.predict_v(s_)
    r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

    s_t, a_t, r_t, minimize = self.graph
    self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})


def BuildModel(
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
