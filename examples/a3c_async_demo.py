"""Demos A3C with Asynchronous runner."""
import gym
from absl import app

from deep_learning.engine import a3c_impl
from deep_learning.engine import async_runner_impl
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import runner_extension_impl
from qpylib import logging
from qpylib import running_environment

running_environment.ForceCpuForTheRun()

logging.ENV.debug_verbosity = 20


def main(_):
  envs = [
    environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    # environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    # environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    # environment_impl.GymEnvironment(gym.make('CartPole-v0')),
    # environment_impl.GymEnvironment(gym.make('CartPole-v0')),
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

  runner.Run(envs=envs, brain=brain, policy=policy, num_of_episodes=1200)


if __name__ == '__main__':
  app.run(main)
