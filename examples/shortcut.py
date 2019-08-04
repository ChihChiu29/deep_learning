"""Provides quick construction of commonly used objects.

Default values are heavily used to keep the interface simple.
"""
import os

import gym

from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_extension_impl
from deep_learning.engine import runner_impl
from deep_learning.engine import screen_learning
from qpylib import logging
from qpylib import string
from qpylib import t

DEFAULT_BATCH_SIZE = 64  # type: int
SAVE_SUB_DIRECTORY = 'saved_models'


class StateLearningPipeline:
  """Quickly generate objects required for a state based learning.

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
      model_shape: t.Iterable[int] = (20, 20, 20),
      report_every_num_of_episodes: int = 100,
  ):
    """Ctor.

    Default implementations are provided for all objects. They can be changed
    by directly setting the public properties after the creation.

    Args:
      gym_env_name: name of the gym environment, like "LunarLander-v2".
      model_shape: a list of number of nodes per hidden layer.
      report_every_num_of_episodes: do progress report every this number of
        episodes.
    """
    self._gym_env_name = gym_env_name
    self._model_shape = tuple(model_shape)

    self.env = environment_impl.GymEnvironment(gym.make(gym_env_name))
    self.qfunc = qfunc_impl.DDQN(
      model_pair=(
        qfunc_impl.CreateModel(
          state_shape=self.env.GetStateShape(),
          action_space_size=self.env.GetActionSpaceSize(),
          hidden_layer_sizes=model_shape),
        qfunc_impl.CreateModel(
          state_shape=self.env.GetStateShape(),
          action_space_size=self.env.GetActionSpaceSize(),
          hidden_layer_sizes=model_shape)),
      training_batch_size=DEFAULT_BATCH_SIZE,
      discount_factor=0.99,
    )
    logging.printf(
      'Using qfunc implementation: %s', string.GetClassName(self.qfunc))
    self.policy = policy_impl.GreedyPolicyWithDecreasingRandomness(
      initial_epsilon=1.0,
      final_epsilon=0.1,
      decay_by_half_after_num_of_episodes=500)
    logging.printf(
      'Using policy implementation: %s', string.GetClassName(self.policy))

    self.runner = runner_impl.ExperienceReplayRunner(
      experience_capacity=100000,
      experience_sample_batch_size=DEFAULT_BATCH_SIZE)
    logging.printf(
      'Using runner implementation: %s', string.GetClassName(self.runner))

    self._progress_tracer = runner_extension_impl.ProgressTracer(
      report_every_num_of_episodes=report_every_num_of_episodes)
    self._model_saver = runner_extension_impl.ModelSaver(
      self._GetModelWeightsFilepath())

  def Train(self, num_of_episodes: int = 5000):
    """Starts a training run.

    Args:
      num_of_episodes: runs a training run for this number of episodes.
    """
    self.runner.ClearCallbacks()
    self.runner.AddCallback(self._progress_tracer)
    self.runner.AddCallback(self._model_saver)

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
    self.env.TurnOnRendering(should_render=True)
    self.env.StartRecording(video_filename=save_video_to)
    runner = runner_impl.NoOpRunner()
    runner.AddCallback(runner_extension_impl.ProgressTracer(
      report_every_num_of_episodes=1))
    runner.Run(
      env=self.env,
      qfunc=self.qfunc,
      policy=policy_impl.GreedyPolicy(),
      num_of_episodes=num_of_episodes)
    self.env.StopRecording()
    self.env.TurnOnRendering(should_render=False)

  def SaveWeights(self):
    """Saves weights to a "saved_models" sub-directory."""
    self.qfunc.Save(self._GetModelWeightsFilepath())

  def LoadWeights(self):
    """Loads weights from a "saved_models" sub-directory."""
    self.qfunc.Load(self._GetModelWeightsFilepath())

  def _GetModelWeightsFilepath(self):
    return os.path.join(SAVE_SUB_DIRECTORY, '%s_%s_%s.weights' % (
      self._gym_env_name,
      string.GetClassName(self.qfunc),
      '-'.join(str(n) for n in self._model_shape)))


class ScreenLearningPipeline:
  """Quickly generates objects required for a full screen based learning."""

  def __init__(
      self,
      gym_env_name: t.Text,
      gym_env=None,
      report_every_num_of_episodes: int = 1,
      use_ddqn: bool = True,
      use_large_model: bool = True,
  ):
    """Ctor.

    Args:
      gym_env_name: name of the gym environment that will be created.
      gym_env: Gym environment. If set, use the provided Gym environment and
        gym_env_name is only used as a tag.
      report_every_num_of_episodes: do progress report every this number of
        episodes.
      use_ddqn: whether to use DDQN or DQN_TargetNetwork.
      use_large_model: whether to use the larger model. Without GPU it's very
        slow to use it.
    """
    self._gym_env_name = gym_env_name
    if gym_env:
      env = gym_env
    else:
      env = gym.make(gym_env_name)
    self.env = screen_learning.ScreenGymEnvironment(env)
    if use_large_model:
      model_pair = (
        screen_learning.CreateOriginalConvolutionModel(
          action_space_size=self.env.GetActionSpaceSize()),
        screen_learning.CreateOriginalConvolutionModel(
          action_space_size=self.env.GetActionSpaceSize()))
    else:
      model_pair = (
        screen_learning.CreateConvolutionModel(
          action_space_size=self.env.GetActionSpaceSize()),
        screen_learning.CreateConvolutionModel(
          action_space_size=self.env.GetActionSpaceSize()))
    if use_ddqn:
      self.qfunc = qfunc_impl.DDQN(
        model_pair=model_pair,
        training_batch_size=DEFAULT_BATCH_SIZE,
        discount_factor=0.99,
      )
    else:
      self.qfunc = qfunc_impl.DQN_TargetNetwork(
        model=model_pair[0],
        training_batch_size=DEFAULT_BATCH_SIZE,
        discount_factor=0.99)
    logging.printf(
      'Using qfunc implementation: %s', string.GetClassName(self.qfunc))
    self.policy = policy_impl.GreedyPolicyWithDecreasingRandomness(
      initial_epsilon=1.0,
      final_epsilon=0.1,
      decay_by_half_after_num_of_episodes=50)
    logging.printf(
      'Using policy implementation: %s', string.GetClassName(self.policy))

    self.runner = runner_impl.ExperienceReplayRunner(
      experience_capacity=100000,
      experience_sample_batch_size=DEFAULT_BATCH_SIZE)
    logging.printf(
      'Using runner implementation: %s', string.GetClassName(self.runner))

    self._progress_tracer = runner_extension_impl.ProgressTracer(
      report_every_num_of_episodes=report_every_num_of_episodes)
    self._model_saver = runner_extension_impl.ModelSaver(
      self._GetModelWeightsFilepath(),
      use_averaged_value_over_num_of_episodes=report_every_num_of_episodes)

  def Train(self, num_of_episodes: int = 5000):
    """Starts a training run.

    Args:
      num_of_episodes: runs a training run for this number of episodes.
    """
    self.runner.ClearCallbacks()
    self.runner.AddCallback(self._progress_tracer)
    self.runner.AddCallback(self._model_saver)

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
    self.env.TurnOnRendering(should_render=True, fps=24)
    self.env.StartRecording(video_filename=save_video_to)
    runner_impl.NoOpRunner().Run(
      env=self.env,
      qfunc=self.qfunc,
      policy=policy_impl.GreedyPolicy(),
      num_of_episodes=num_of_episodes)
    self.env.StopRecording()
    self.env.TurnOnRendering(should_render=False)

  def SaveWeights(self):
    """Saves weights to a "saved_models" sub-directory."""
    self.qfunc.Save(self._GetModelWeightsFilepath())

  def LoadWeights(self):
    """Loads weights from a "saved_models" sub-directory."""
    self.qfunc.Load(self._GetModelWeightsFilepath())

  def _GetModelWeightsFilepath(self):
    return os.path.join(SAVE_SUB_DIRECTORY, '%s_%s.weights' % (
      self._gym_env_name, string.GetClassName(self.qfunc)))
