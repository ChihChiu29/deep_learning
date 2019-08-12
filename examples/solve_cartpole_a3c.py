"""Demo using A3C to solve CartPole-v0."""
import gym

from deep_learning.engine import a3c_impl
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import runner_extension_impl
from qpylib import running_environment

running_environment.ForceCpuForTheRun()

from absl import app


# Profiler instruction:
# 1) Generate profiler file:
#   $ python -m cProfile -o result.prof deep_learning/examples/solve_cartpole_a3c.py
# 2) Visualize it:
#   $ snakeviz result.prof
#   It prints a link that shows the viz.


def main(_):
  running_environment.ForceCpuForTheRun()

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
  runner = a3c_impl.NStepExperienceRunner()
  # runner = runner_impl.SimpleRunner()
  runner.AddCallback(
    runner_extension_impl.ProgressTracer(report_every_num_of_episodes=100))

  runner.Run(env=env, brain=brain, policy=policy, num_of_episodes=1200)


if __name__ == '__main__':
  app.run(main)
