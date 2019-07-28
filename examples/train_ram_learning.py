"""An example of training using ram states.

Mainly used for performance analysis.
"""
from absl import app

from deep_learning.examples import shortcut


# Profiler instruction:
# 1) Generate profiler file:
#   $ python -m cProfile -o result.prof \
#     deep_learning/examples/train_ram_learning.py
# 2) Visualize it:
#   $ snakeviz result.prof
#   It prints a link that shows the viz.

def main(_):
  pipline = shortcut.StateLearningPipeline(
    'Seaquest-ram-v0',
    model_shape=(64, 32, 20),
    report_every_num_of_episodes=100,
  )
  pipline.Train(num_of_episodes=500)


if __name__ == '__main__':
  app.run(main)
