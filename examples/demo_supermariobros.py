"""Demo SuperMarioBros."""

import gym_super_mario_bros
from absl import app

from deep_learning.examples import shortcut
from qpylib import running_environment


def main(_):
  running_environment.ForceCpuForTheRun()
  print(gym_super_mario_bros.__doc__)
  pipeline = shortcut.ScreenLearningPipeline(gym_env_name='SuperMarioBros-v0')
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
