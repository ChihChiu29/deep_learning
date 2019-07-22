"""Provides some environment implementations."""
import numpy
from gym.wrappers.monitoring import video_recorder

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Action, Transition
from qpylib import t


class GymEnvironmentError(Exception):
  pass


class SingleStateEnvironment(q_base.Environment):
  """An environment of a single state."""

  def __init__(
      self,
      action_space_size: int,
      step_limit: int,
  ):
    """Constructor.

    Args:
      action_space_size: the size of the action space. All actions are no-ops.
      step_limit: the number of actions allowed to take before environment
        is "done".
    """
    super().__init__(state_space_dim=1, action_space_size=action_space_size)
    self._step_limit = step_limit

    self._action_count = None
    self._state = numpy.array([[0]])
    self.Reset()

  # @Override
  def Reset(self) -> q_base.State:
    self._action_count = 0
    return self._state

  # @Override
  def TakeAction(self, action: Action) -> Transition:
    if self._action_count >= self._step_limit:
      sp = None
    else:
      sp = self._state

    self._action_count += 1
    return q_base.Transition(s=self._state, a=action, r=0.0, sp=sp)


class GymEnvironment(q_base.Environment):
  """Wraps a OpenAI Gym environment."""

  def __init__(
      self,
      gym_env,
  ):
    """Constructor.

    Args:
      gym_env: an environment made from `gym.make`.
    """
    if len(gym_env.observation_space.shape) != 1:
      raise GymEnvironmentError('observation_space is not 1-d.')

    super().__init__(
      state_space_dim=gym_env.observation_space.shape[0],
      action_space_size=gym_env.action_space.n)

    self._gym_env = gym_env

    self._current_state = None  # Initialized by `Reset`.

    self._render_frames = False
    self._recorder = None  # Initialized in `StartRecording`.

  def TurnOnRendering(self, render: bool):
    """Whether to render each frame.

    Note that the "native" rendering for each frame is fast, but it's very
    slow when each frame is showed in notebook -- use recording instead.
    """
    self._render_frames = render

  def _ConvertState(self, state):
    """Converts the Gym state to the interface standard."""
    return state[numpy.newaxis, :]

  # @Override
  def Reset(self) -> q_base.State:
    self._current_state = self._ConvertState(self._gym_env.reset())
    return self._current_state

  # @Override
  def TakeAction(self, action: Action) -> Transition:

    if self._render_frames:
      self._gym_env.render(mode='rgb_array')

    if self._recorder and self._recorder.enabled:
      self._recorder.capture_frame()

    observation, reward, done, info = self._gym_env.step(
      self.GetChoiceFromAction(action))

    if done:
      sp = None
    else:
      sp = self._ConvertState(observation)

    transition = q_base.Transition(
      s=self._current_state, a=action, r=reward, sp=sp)
    self._current_state = sp
    return transition

  def StartRecording(self, video_filename: t.Text):
    """Starts to record a new animation; requires plot=True."""
    self._recorder = video_recorder.VideoRecorder(
      self._gym_env, video_filename, enabled=True)

  def StopRecording(self):
    """Stops recording."""
    self._recorder.close()
    self._recorder = None

  def GetGymEnvMaxEpisodeSteps(self) -> int:
    """Gets the max episode steps for the gym environment."""
    return self._gym_env._max_episode_steps

  def SetGymEnvMaxEpisodeSteps(self, steps: int) -> None:
    """Sets the max episode steps for the gym environment."""
    self._gym_env._max_episode_steps = steps
