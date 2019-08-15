"""Demos A3C with Asynchronous runner."""
import gym
import keras
from absl import app
from keras import layers

from deep_learning.engine import a3c_impl
from deep_learning.engine import async_runner_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import runner_extension_impl
from deep_learning.engine import runner_impl
from deep_learning.engine import screen_learning
from qpylib import logging
from qpylib import t

# running_environment.ForceCpuForTheRun()

logging.ENV.debug_verbosity = 6

IMAGE_STACK = 2
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84


def CreateModel(
    state_shape: t.Sequence[int],
    action_space_size: int,
    activation: t.Text = 'relu',
):
  """Builds a model for A3C.

  Args:
    state_shape: the shape of the state ndarray.
    action_space_size: the size of the action space.
    activation: the activation, for example "relu".
  """
  input_layer = layers.Input(shape=state_shape)

  l = layers.Conv2D(
    16, (8, 8),
    strides=(4, 4),
    activation=activation,
    input_shape=(IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT),
    data_format='channels_first')(input_layer)
  l = layers.Conv2D(
    16,
    (4, 4),
    strides=(2, 2),
    activation=activation)(l)
  l = layers.Flatten()(l)
  l = layers.Dense(units=32, activation=activation)(l)

  out_pi = layers.Dense(action_space_size, activation='softmax')(l)
  out_v = layers.Dense(1, activation='linear')(l)

  model = keras.Model(inputs=[input_layer], outputs=[out_pi, out_v])
  model._make_predict_function()  # have to initialize before threading
  return model


def Train(_):
  env = screen_learning.ScreenGymEnvironment(gym.make('SpaceInvaders-v0'))
  async_env_runners = []  # type: t.List[async_runner_impl.AsyncEnvRunner]
  for _ in range(10):
    async_env_runners.append(async_runner_impl.AsyncEnvRunner(
      env=screen_learning.ScreenGymEnvironment(gym.make('SpaceInvaders-v0')),
      runner=runner_impl.NStepExperienceRunner(n_step_return=10),
    ))
  brain = async_runner_impl.AsyncBrain(a3c_impl.A3C(
    model=CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
    )
  ))
  brain.Load('saved_models/a3c_invader.weights')  # warm start

  policy = policy_impl.PolicyWithDecreasingRandomness(
    base_policy=policy_impl.PiWeightedPolicy(),
    initial_epsilon=0.2,
    final_epsilon=0.05,
    decay_by_half_after_num_of_episodes=500,
  )
  runner = async_runner_impl.ParallelRunner(async_env_runners)
  runner.AddCallback(
    async_runner_impl.AsyncRunnerExtension(
      runner_extension_impl.ProgressTracer(report_every_num_of_episodes=10)))
  runner.AddCallback(
    async_runner_impl.AsyncRunnerExtension(
      runner_extension_impl.ModelSaver(
        save_filepath='saved_models/a3c_invader.weights',
        use_averaged_value_over_num_of_episodes=30)))

  runner.Run(brain=brain, policy=policy, num_of_episodes=200)


def Demo(_):
  env = screen_learning.ScreenGymEnvironment(gym.make('SpaceInvaders-v0'))
  brain = a3c_impl.A3C(
    model=a3c_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(12,),
    )
  )
  brain.Load('saved_models/a3c_invader.weights')
  policy = policy_impl.GreedyPolicy()

  env.StartRecording('a3c_invader.mp4')
  runner = runner_impl.SimpleRunner()
  runner.Run(env=env, brain=brain, policy=policy, num_of_episodes=10)
  env.StopRecording()


if __name__ == '__main__':
  app.run(Train)
  # app.run(Demo)
