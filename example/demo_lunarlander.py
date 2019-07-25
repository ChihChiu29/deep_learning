"""Demo LunarLander.

The exported model is not the final version, but the improvement by the
training is already obvious from the animations.
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
  env = environment_impl.GymEnvironment(gym.make('LunarLander-v2'))
  rand_qfunc = qfunc_impl.RandomValueQFunction(env.GetActionSpaceSize())
  qfunc = qfunc_impl.DQN(
    model=qfunc_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(20, 20, 20)),
    training_batch_size=batch_size,
    discount_factor=0.99,
  )
  qfunc._model.load_weights(
    'saved_models/lunarlander_shape_20-20-20_rmsprop_gamma_099.weights')
  policy = policy_impl.GreedyPolicy()
  runner = runner_impl.ExperienceReplayRunner(
    experience_capacity=100000, experience_sample_batch_size=batch_size)

  env.TurnOnRendering(should_render=True, fps=24)
  logging.ENV.debug_verbosity = 9

  # First 2 runs with random actions:
  runner.Run(env=env, qfunc=rand_qfunc, policy=policy, num_of_episodes=2)
  # Then 10 runs with trained qfunc:
  runner.Run(env=env, qfunc=qfunc, policy=policy, num_of_episodes=10)


if __name__ == '__main__':
  app.run(main)
