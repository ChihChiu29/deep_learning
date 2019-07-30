"""Demo LunarLander."""

from absl import app

from deep_learning.examples import shortcut


def main(_):
  pipeline = shortcut.StateLearningPipeline(
    gym_env_name='Seaquest-ram-v0', model_shape=(20, 20, 20))
  pipeline.LoadWeights()
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
