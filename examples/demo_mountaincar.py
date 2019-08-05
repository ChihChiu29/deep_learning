"""Demo MountainCar.

The exported model is not the final version, but the improvement by the
training is already obvious from the animations.
"""

import gym
from absl import app

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl
from qpylib import logging


def main(_):
  batch_size = 64  # used in qfunc and runner.
  env = environment_impl.GymEnvironment(gym.make('MountainCar-v0'))
  env.SetGymEnvMaxEpisodeSteps(400)
  qfunc = qfunc_impl.DQN(
    model=qfunc_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(64,)),
    training_batch_size=batch_size,
    discount_factor=0.99,
  )
  qfunc.Load('saved_models/mountaincar_shape_64_rmsprop_gamma_099.weights')
  policy = policy_impl.GreedyPolicy()
  runner = runner_impl.NoOpRunner()

  env.TurnOnRendering(should_render=True, fps=24)
  logging.ENV.debug_verbosity = 9

  env.StartRecording(video_filename='mountaincar_demo.mp4')
  # First 5 runs with random actions:
  rand_qfunc = qfunc_impl.RandomQFunction(env.GetActionSpaceSize())
  runner.Run(env=env, brain=rand_qfunc, policy=policy, num_of_episodes=5)
  # Then 10 runs with trained qfunc:
  runner.Run(env=env, brain=qfunc, policy=policy, num_of_episodes=10)
  env.StopRecording()


if __name__ == '__main__':
  app.run(main)
