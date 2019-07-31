"""Demo LunarLander."""

from absl import app

from deep_learning.examples import shortcut


def main(_):
  pipeline = shortcut.ScreenLearningPipeline(gym_env_name='Seaquest-v0')
  pipeline.LoadWeights()
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
