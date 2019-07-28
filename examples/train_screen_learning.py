"""An example of training using screen.

Mainly used for performance analysis.
"""
from absl import app

from deep_learning.examples import shortcut


# Profiler instruction:
# 1) Generate profiler file:
#   $ python -m cProfile -o result.prof \
#     deep_learning/examples/train_screen_learning.py
# 2) Visualize it:
#   $ snakeviz result.prof
#   It prints a link that shows the viz.


def main(_):
  pipeline = shortcut.ScreenLearningPipeline(gym_env_name='Seaquest-v0')
  pipeline.Train(num_of_episodes=20)


if __name__ == '__main__':
  app.run(main)
