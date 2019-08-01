"""Demo PacMan."""

from absl import app

from deep_learning.examples import shortcut
from qpylib import running_environment


def main(_):
  running_environment.ForceCpuForTheRun()
  pipeline = shortcut.ScreenLearningPipeline(gym_env_name='MsPacman-v0')
  pipeline.LoadWeights()
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
