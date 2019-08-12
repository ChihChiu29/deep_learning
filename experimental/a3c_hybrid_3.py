# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

# WORKING

import gym

from deep_learning.engine import base
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import runner_extension_impl
from deep_learning.engine.base import States
from deep_learning.engine.base import Transition
from deep_learning.engine.base import Values
from deep_learning.experimental import a3c_impl
from qpylib import logging
from qpylib import running_environment
from qpylib import t

running_environment.ForceCpuForTheRun()

import threading

import tensorflow as tf
from absl import app
from keras.layers import *
from keras.models import *

# -- constants
ENV = 'CartPole-v0'

RUN_TIME = 60
THREADS = 1
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

count = 0
rewards = []

env = environment_impl.GymEnvironment(gym_env=gym.make('CartPole-v0'))
NUM_STATE = env.GetStateShape()[0]
NUM_ACTIONS = env.GetActionSpaceSize()
NONE_STATE = np.zeros(NUM_STATE)


class A3C_EXP(base.Brain):
  """A A3C brain."""

  def __init__(self):
    self.session = tf.Session()
    K.set_session(self.session)
    K.manual_variable_initialization(True)

    self.model = self._build_model()
    self.graph = self._build_graph(self.model)

    self.session.run(tf.global_variables_initializer())
    self.default_graph = tf.get_default_graph()

    self.default_graph.finalize()  # avoid modifications

  def _build_model(self):
    l_input = Input(batch_shape=(None, NUM_STATE))
    l_dense = Dense(16, activation='relu')(l_input)

    out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
    out_value = Dense(1, activation='linear')(l_dense)

    model = Model(inputs=[l_input], outputs=[out_actions, out_value])
    model._make_predict_function()  # have to initialize before threading

    return model

  def _build_graph(self, model):
    s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
    a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
    r_t = tf.placeholder(tf.float32, shape=(
      None, 1))  # not immediate, but discounted n step reward

    p, v = model(s_t)

    log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    advantage = r_t - v

    loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
    loss_value = LOSS_V * tf.square(advantage)  # minimize value error
    entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                           keep_dims=True)  # maximize entropy (regularization)

    loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
    minimize = optimizer.minimize(loss_total)

    return s_t, a_t, r_t, minimize

  # @Override
  def GetValues(
      self,
      states: base.States,
  ) -> Values:
    """Use Pi values to make decision."""
    pi_values, v = self.predict(states)
    logging.vlog(20, 'GET pi for state %s: %s', states, pi_values)
    return pi_values

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[base.Transition],
  ) -> None:
    states, actions, rewards, new_states, reward_mask = (
      self.CombineTransitions(transitions))

    v_values = self._GetV(states)
    rewards = rewards + self._gamma * v_values * reward_mask

    s_input, a_input, r_input, minimize = self._graph
    self.session.run(
      minimize, feed_dict={s_input: states, a_input: actions, r_input: rewards})

    v = self.predict_v(s_)
    r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

    s_t, a_t, r_t, minimize = self.graph
    self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

  # @Override
  def Save(self, filepath: t.Text) -> None:
    pass

  # @Override
  def Load(self, filepath: t.Text) -> None:
    pass

  def _GetV(self, states: base.States) -> base.OneDArray:
    return self.predict_v(states)

  def predict(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return p, v

  def predict_p(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return p

  def predict_v(self, s):
    with self.default_graph.as_default():
      p, v = self.model.predict(s)
      return v


# ---------
class JBrain:
  train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
  lock_queue = threading.Lock()

  def __init__(self):
    self.session = tf.Session()
    K.set_session(self.session)
    K.manual_variable_initialization(True)

    self.model = self._build_model()
    self.graph = self._build_graph(self.model)

    self.session.run(tf.global_variables_initializer())
    self.default_graph = tf.get_default_graph()

    self.default_graph.finalize()  # avoid modifications

  def optimize(self):
    if len(self.train_queue[0]) < MIN_BATCH:
      # time.sleep(0)  # yield
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

  def train_push(self, s, a, r, s_):
    with self.lock_queue:
      self.train_queue[0].append(s)
      self.train_queue[1].append(a)
      self.train_queue[2].append(r)

      if s_ is None:
        self.train_queue[3].append(NONE_STATE)
        self.train_queue[4].append(0.)
      else:
        self.train_queue[3].append(s_)
        self.train_queue[4].append(1.)


class QBrain(base.Brain):

  def __init__(self):
    self._jbrain = JBrain()

  def GetValues(self, states: States) -> Values:
    return self._jbrain.predict_p(states)

  def UpdateFromTransitions(self, transitions: t.Iterable[Transition]) -> None:
    for tran in transitions:
      if tran.sp is not None:
        self._jbrain.train_push(tran.s[0], tran.a[0], tran.r, tran.sp[0])
      else:
        self._jbrain.train_push(tran.s[0], tran.a[0], tran.r, NONE_STATE)
    self._jbrain.optimize()

  def Save(self, filepath: t.Text) -> None:
    pass

  def Load(self, filepath: t.Text) -> None:
    pass


def main(_):
  brain = QBrain()
  policy = policy_impl.PolicyWithDecreasingRandomness(
    base_policy=a3c_impl.WeightedPiPolicy(),
    initial_epsilon=0.4,
    final_epsilon=0.05,
    decay_by_half_after_num_of_episodes=500,
  )
  runner = a3c_impl.NStepExperienceRunner()
  runner.AddCallback(
    runner_extension_impl.ProgressTracer(report_every_num_of_episodes=10))
  runner.Run(env=env, brain=brain, policy=policy, num_of_episodes=1200)


if __name__ == '__main__':
  app.run(main)
