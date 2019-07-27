"""Provides quick construction of commonly used objects.

Default values are heavily used to keep the interface simple.
"""
import os

import gym

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import reporter_impl
from deep_learning.engine import runner_impl
from qpylib import t

DEFAULT_BATCH_SIZE = 64  # type: int
SAVE_SUB_DIRECTORY = 'saved_models'


class RunEnvironment:
  """Helps to quickly generate objects required for a full run.

  When created, default implementations are used to create environment,
  Q-function, policy, and runner, and only the most commonly changed parameters
  are exposed for quick modification.

  If you need full flexibility, either modify the attributes on the created
  instance (this is less efficient) or create those objects directly using
  <type>_impl modules.
  """

  def __init__(
      self,
      gym_env_name: t.Text,
      model_shape: t.Tuple[int] = (20, 20, 20),
  ):
    """Ctor.

    Default implementations are provided for all objects. They can be changed

    Args:
      gym_env_name: name of the gym environment, like "LunarLander-v2".
      model_shape: a list of number of nodes per hidden layer.
    """
    self._gym_env_name = gym_env_name
    self._model_shape = model_shape

    self.env = environment_impl.GymEnvironment(gym.make(gym_env_name))
    self.qfunc = qfunc_impl.DQN(
      model=qfunc_impl.CreateModel(
        state_shape=self.env.GetStateShape(),
        action_space_size=self.env.GetActionSpaceSize(),
        hidden_layer_sizes=model_shape),
      training_batch_size=DEFAULT_BATCH_SIZE,
      discount_factor=0.99,
    )
    self.policy = policy_impl.GreedyPolicyWithDecreasingRandomness(
      initial_epsilon=1.0,
      final_epsilon=0.1,
      decay_by_half_after_num_of_episodes=500)

    self.runner = runner_impl.ExperienceReplayRunner(
      experience_capacity=100000,
      experience_sample_batch_size=DEFAULT_BATCH_SIZE)

  def Train(self, num_of_episodes: int = 5000):
    """Starts a training run.

    Args:
      num_of_episodes: runs a training run for this number of episodes.
    """
    self.runner.ClearReporters()
    self.runner.AddReporter(reporter_impl.ProgressTracer())

    self.runner.Run(
      env=self.env,
      qfunc=self.qfunc,
      policy=self.policy,
      num_of_episodes=num_of_episodes)

  def Demo(self, num_of_episodes: int = 10, save_video_to: t.Text = 'demo.mp4'):
    """Starts a demo run.

    Args:
      num_of_episodes: number of runs to demo.
      save_video_to: saves the demo video for the run to a file of this
        name. It must ends with mp4.
    """
    self.env.StartRecording(video_filename=save_video_to)
    runner_impl.NoOpRunner().Run(
      env=self.env,
      qfunc=self.qfunc,
      policy=policy_impl.GreedyPolicy(),
      num_of_episodes=num_of_episodes)
    self.env.StopRecording()

  def SaveWeights(self):
    """Saves weights to a "saved_models" sub-directory."""
    self.qfunc.SaveWeights(self._GetModelWeightsFilepath())

  def LoadWeights(self):
    self.qfunc.LoadWeights(self._GetModelWeightsFilepath())

  def _GetModelWeightsFilepath(self):
    return os.path.join(SAVE_SUB_DIRECTORY, '%s_%s.weights' % (
      self._gym_env_name, '-'.join(str(n) for n in self._model_shape)))
