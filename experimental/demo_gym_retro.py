"""Demo Gym retro usage."""
import retro
from absl import app

from deep_learning.examples import shortcut
from qpylib import running_environment


# It does NOT work because the input space is MultiBinary(9), but the
# interface does not support taking simultaneous actions.
def main(_):
  running_environment.ForceCpuForTheRun()

  retro.data.merge('../../roms/contraforce.nes', quiet=False)
  env = retro.make(game='ContraForce-Nes')
  pipeline = shortcut.ScreenLearningPipeline(
    gym_env_name='ContraForce', gym_env=env)
  pipeline.Demo()


if __name__ == '__main__':
  app.run(main)
