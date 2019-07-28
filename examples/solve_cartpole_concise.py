"""Solves Cartpole-v0.

This this the concise version that uses `shortcut` module to speed up the
running setup. It has less flexibility but is very quick to demonstrate the
fun of CL. For a "full" version see solve_cartpole.py.
"""
from absl import app

from deep_learning.examples import shortcut


def main(_):
  pipeline = shortcut.FullRunPipeline(
    gym_env_name='CartPole-v0', model_shape=(20, 20, 20))
  # First train it for 500 episodes.
  pipeline.Train(num_of_episodes=500)
  # Then demo it with video!
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
