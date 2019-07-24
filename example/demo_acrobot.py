"""Demo Acrobot-v1.

It reads a pre-saved model, which has not completely solved the problem, but
it's clear from the animation that the agent is learning.
"""

import gym
from absl import app

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl
from qpylib import logging


# For parameters see:
#   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
def main(_):
  batch_size = 64  # used in qfunc and runner.
  env = environment_impl.GymEnvironment(gym.make('Acrobot-v1'))
  qfunc = qfunc_impl.DQN(
    model=qfunc_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(20, 20, 20)),
    training_batch_size=batch_size,
    discount_factor=0.99,
  )
  qfunc.LoadModel(
    'saved_models/acrobot_v1_shape_20-20-20_rmsprop_gamma_0.99.model')
  policy = policy_impl.GreedyPolicy()
  runner = runner_impl.ExperienceReplayRunner(
    experience_capacity=100000, experience_sample_batch_size=batch_size)

  env.TurnOnRendering(should_render=True, fps=10)
  logging.ENV.debug_verbosity = 9
  runner.Run(env=env, qfunc=qfunc, policy=policy, num_of_episodes=10)


if __name__ == '__main__':
  app.run(main)
