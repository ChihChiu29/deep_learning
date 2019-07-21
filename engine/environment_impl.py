"""Provides some environment implementations."""
import IPython
import numpy

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Action, Transition


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

    # For recording
    self._frames = None
    self._in_recording = False

  # @Override
  def Reset(self) -> q_base.State:
    self._current_state = self._gym_env.reset()
    IPython.get_ipython().magic('matplotlib inline')
    return self._current_state

  # @Override
  def TakeAction(self, action: Action) -> Transition:

    if self._in_recording:
      self._frames.append(self._gym_env.render(mode='rgb_array'))

    observation, reward, done, info = self._gym_env.step(
      self.GetChoiceFromAction(action))

    if done:
      sp = None
    else:
      sp = observation

    transition = q_base.Transition(
      s=self._current_state, a=action, r=reward, sp=sp)
    self._current_state = sp
    return transition

  def StartRecording(self):
    """Starts to record a new animation; requires plot=True."""
    self._frames = []
    self._in_recording = True

  def StopRecording(self):
    """Stops recording."""
    self._in_recording = False

  def PlayRecording(self):
    """Plays the last recording."""
    _DisplayFramesAsGif(self._frames)
