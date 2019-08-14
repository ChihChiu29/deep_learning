"""Demos A3C with Asynchronous runner."""
import gym
from absl import app

from deep_learning.engine import a3c_impl
from deep_learning.engine import async_runner_impl
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import runner_extension_impl
from deep_learning.engine import runner_impl
from qpylib import logging
from qpylib import running_environment

running_environment.ForceCpuForTheRun()

logging.ENV.debug_verbosity = 6


def SynchronousMultiEnvs(_):
  envs = [
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
  ]
  brain = a3c_impl.A3C(
    model=a3c_impl.CreateModel(
      state_shape=envs[0].GetStateShape(),
      action_space_size=envs[0].GetActionSpaceSize(),
      hidden_layer_sizes=(12,),
    )
  )

  policy = policy_impl.PolicyWithDecreasingRandomness(
    base_policy=policy_impl.PiWeightedPolicy(),
    initial_epsilon=0.2,
    final_epsilon=0.05,
    decay_by_half_after_num_of_episodes=500,
  )
  runner = async_runner_impl.MultiEnvsParallelBatchedRunner(batch_size=32)
  runner.AddCallback(
    runner_extension_impl.ProgressTracer(report_every_num_of_episodes=100))
  runner.AddCallback(
    runner_extension_impl.ModelSaver(
      save_filepath='saved_models/a3c_cartpole_12.weights',
      use_averaged_value_over_num_of_episodes=10))

  runner.Run(envs=envs, brain=brain, policy=policy, num_of_episodes=2000)


# Asynchronous
def AsyncMultiEnvs(_):
  envs = [
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
  ]
  brain = a3c_impl.A3C(
    model=a3c_impl.CreateModel(
      state_shape=envs[0].GetStateShape(),
      action_space_size=envs[0].GetActionSpaceSize(),
      hidden_layer_sizes=(12,),
    )
  )

  policy = policy_impl.PolicyWithDecreasingRandomness(
    base_policy=policy_impl.PiWeightedPolicy(),
    initial_epsilon=0.2,
    final_epsilon=0.05,
    decay_by_half_after_num_of_episodes=500,
  )
  runner = runner_impl.MultiEnvsSequentialBatchedRunner(batch_size=32)
  runner.AddCallback(
    runner_extension_impl.ProgressTracer(report_every_num_of_episodes=100))

  runner.Run(envs=envs, brain=brain, policy=policy, num_of_episodes=1200)


def NStepReward(_):
  batch_size = 64  # used in qfunc and runner.
  env = environment_impl.GymEnvironment(gym.make('CartPole-v0'))
  brain = a3c_impl.A3C(
    model=a3c_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(12,),
    )
  )

  policy = policy_impl.PolicyWithDecreasingRandomness(
    base_policy=policy_impl.PiWeightedPolicy(),
    initial_epsilon=0.4,
    final_epsilon=0.05,
    decay_by_half_after_num_of_episodes=500,
  )
  runner = runner_impl.NStepExperienceRunner()
  runner.AddCallback(
    runner_extension_impl.ProgressTracer(report_every_num_of_episodes=100))
  runner.AddCallback(
    runner_extension_impl.ModelSaver(
      save_filepath='saved_models/a3c_cartpole_12.weights',
      use_averaged_value_over_num_of_episodes=20))

  runner.Run(env=env, brain=brain, policy=policy, num_of_episodes=1600)


def Demo(_):
  env = environment_impl.GymEnvironment(gym.make('CartPole-v0'))
  brain = a3c_impl.A3C(
    model=a3c_impl.CreateModel(
      state_shape=env.GetStateShape(),
      action_space_size=env.GetActionSpaceSize(),
      hidden_layer_sizes=(12,),
    )
  )
  brain.Load('saved_models/a3c_cartpole_12.weights')
  policy = policy_impl.GreedyPolicy()

  env.StartRecording('a3c_cartpole.mp4')
  runner = runner_impl.SimpleRunner()
  runner.Run(env=env, brain=brain, policy=policy, num_of_episodes=10)
  env.StopRecording()


if __name__ == '__main__':
  # app.run(NStepReward)
  app.run(Demo)
