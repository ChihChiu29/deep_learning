"""Solves Cartpole-v0."""
import gym
from absl import app

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl
from qpylib import logging


# Profiler instruction:
# 1) Generate profiler file:
#   $ python -m cProfile -o result.prof deep_learning/examples/solve_cartpole.py
# 2) Visualize it:
#   $ snakeviz result.prof
#   It prints a link that shows the viz.


# For parameters see:
#   https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
def main(_):
  batch_size = 64  # used in qfunc and runner.
  env = environment_impl.GymEnvironment(gym.make('CartPole-v0'))
  qfunc = qfunc_impl.DQN(
    model=qfunc_impl.CreateModel(
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

  # Demo with video.
  env.TurnOnRendering(should_render=True, fps=24)
  # env.StartRecording(video_filename='demo.mp4')  # uncomment to record video.
  # First 5 runs with random actions:
  runner.Run(
    env=env,
    qfunc=qfunc_impl.RandomValueQFunction(env.GetActionSpaceSize()),
    policy=policy,
    num_of_episodes=5)
  # Then 10 runs with trained qfunc:
  runner.Run(env=env, qfunc=qfunc, policy=policy, num_of_episodes=10)
  # env.StopRecording()  # uncomment if record video is called.


if __name__ == '__main__':
  app.run(main)
