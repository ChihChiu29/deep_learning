# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

# VERY UNSTABLE; ALMOST NOT WORKING

import numpy
from matplotlib import pyplot

from deep_learning.engine import base
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.experimental import a3c_impl
from qpylib import running_environment

running_environment.ForceCpuForTheRun()

import random
import threading
import time

import gym
from absl import app
from keras.layers import *

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


# ---------
class Brain:

  def __init__(self):
    self._env = environment_impl.GymEnvironment(gym_env=gym.make('CartPole-v0'))
    self._brain = a3c_impl.A3C(
      model=a3c_impl.CreateModel(
        state_shape=self._env.GetStateShape(),
        action_space_size=self._env.GetActionSpaceSize(),
        hidden_layer_sizes=(12,),
      )
    )

  def predict_p(self, s):
    return self._brain.GetValues(s)

  def train_push(self, s, a, r, s_):
    return self._brain.UpdateFromTransitions(
      [base.Transition(
        s=numpy.array([s]), a=numpy.array([a]), r=r, sp=numpy.array([s_]))])

  def optimize(self):
    pass


# ---------
frames = 0


class Agent:
  def __init__(self, eps_start, eps_end, eps_steps):
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_steps = eps_steps

    self.memory = []  # used for n_step return
    self.R = 0.

    self._policy = policy_impl.GreedyPolicyWithDecreasingRandomness(
      initial_epsilon=0.4, final_epsilon=0.05,
      decay_by_half_after_num_of_episodes=500)
    self._a3c_runner = a3c_impl.NStepExperienceRunner()

  def getEpsilon(self):
    if (frames >= self.eps_steps):
      return self.eps_end
    else:
      return self.eps_start + frames * (
          self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

  def act(self, s):
    # return brain._env.GetChoiceFromAction(
    #   self._policy.Decide(brain._env, brain._brain, numpy.array([s]), 0, 0))
    eps = self.getEpsilon()
    global frames;
    frames = frames + 1

    if random.random() < eps:
      return random.randint(0, NUM_ACTIONS - 1)

    else:
      s = np.array([s])
      p = brain.predict_p(s)[0]

      # a = np.argmax(p)
      a = np.random.choice(NUM_ACTIONS, p=p)

      return a

  def train(self, s, a, r, s_):

    def get_sample(memory, n):
      s, a, _, _ = memory[0]
      _, _, _, s_ = memory[n - 1]

      return s, a, self.R, s_

    a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
    a_cats[a] = 1

    # trans = self._a3c_runner._protected_ProcessTransition(
    #   mock.MagicMock(), base.Transition(s=s, a=a_cats, r=r, sp=s_), 0)
    # print('R cmp, mine: %s' % ','.join(str(t) for t in trans))

    self.memory.append((s, a_cats, r, s_))

    self.R = (self.R + r * GAMMA_N) / GAMMA

    if s_ is None:
      while len(self.memory) > 0:
        n = len(self.memory)
        s, a, r, s_ = get_sample(self.memory, n)
        # print('R cmp, original: %s' % ([s, a, r, s_]))
        brain.train_push(s, a, r, s_)

        self.R = (self.R - self.memory[0][2]) / GAMMA
        self.memory.pop(0)

      self.R = 0

    if len(self.memory) >= N_STEP_RETURN:
      s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
      # print('R cmp, original: %s' % ([s, a, r, s_]))
      brain.train_push(s, a, r, s_)

      self.R = self.R - self.memory[0][2]
      self.memory.pop(0)

      # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
  stop_signal = False

  def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP,
               eps_steps=EPS_STEPS):
    threading.Thread.__init__(self)

    self.render = render
    self.env = gym.make(ENV)
    self.agent = Agent(eps_start, eps_end, eps_steps)

  def runEpisode(self):
    s = self.env.reset()

    R = 0
    while True:
      time.sleep(THREAD_DELAY)  # yield

      if self.render: self.env.render()

      a = self.agent.act(s)
      s_, r, done, info = self.env.step(a)

      if done:  # terminal state
        s_ = None

      self.agent.train(s, a, r, s_)

      s = s_
      R += r

      if done or self.stop_signal:
        global count
        count += 1
        print("%d: Total R: %s" % (count, R))
        rewards.append(R)
        break

    # print("Total R:", R)

  def run(self):
    while not self.stop_signal:
      self.runEpisode()

  def stop(self):
    self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
  stop_signal = False

  def __init__(self):
    threading.Thread.__init__(self)

  def run(self):
    while not self.stop_signal:
      brain.optimize()

  def stop(self):
    self.stop_signal = True


env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]


def main(_):
  for o in opts:
    o.start()

  for e in envs:
    e.start()

  time.sleep(RUN_TIME)

  for e in envs:
    e.stop()
  for e in envs:
    e.join()

  for o in opts:
    o.stop()
  for o in opts:
    o.join()

  print("Training finished")
  pyplot.plot(rewards)
  pyplot.show()
  # env_test.run()


if __name__ == '__main__':
  app.run(main)
