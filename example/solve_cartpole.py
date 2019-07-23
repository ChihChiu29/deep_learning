"""Solves Cartpole-v0."""
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
  env = environment_impl.GymEnvironment(gym.make('CartPole-v0'))
  qfunc = qfunc_impl.DQN(
    model=qfunc_impl.CreateSingleModelWithRMSProp(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(20, 20, 20)),
    training_batch_size=batch_size,
    discount_factor=0.99,
  )
  runner = runner_impl.ExperienceReplayRunner(
    experience_capacity=100000, experience_sample_batch_size=batch_size)

  # Train 500 episodes.
  logging.ENV.debug_verbosity = 3
  policy = policy_impl.GreedyPolicyWithRandomness(epsilon=0.1)
  runner.Run(env=env, qfunc=qfunc, policy=policy, num_of_episodes=500)

  # Test for 100 episodes.
  logging.ENV.debug_verbosity = 4
  policy = policy_impl.GreedyPolicy()
  runner.Run(env=env, qfunc=qfunc, policy=policy, num_of_episodes=100)


if __name__ == '__main__':
  app.run(main)
