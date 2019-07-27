"""Quick demo of Gym environment with random actions."""
import gym
from absl import app

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_impl


def main(_):
  env = environment_impl.GymEnvironment(gym.make('Seaquest-v0'))
  env.TurnOnRendering(should_render=True, fps=24)
  qfunc = qfunc_impl.RandomValueQFunction(
    action_space_size=env.GetActionSpaceSize())
  policy = policy_impl.GreedyPolicyWithRandomness(epsilon=1.0)
  runner = runner_impl.NoOpRunner()

  runner.Run(env, qfunc, policy, num_of_episodes=10)


if __name__ == '__main__':
  app.run(main)
