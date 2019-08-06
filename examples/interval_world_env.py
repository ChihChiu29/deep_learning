"""A circular world environment."""
import numpy

from deep_learning.engine import base

STEP_LIMIT = 500


class IntervalWorld(base.Environment):
  """A world consists of integers between -n and n.

  Possible actions are go left (0), right (2), or stay (1). Going left from -n
  or going right from n ends the environment. Any action getting closer
  to 0 is given +1 reward, with getting away given -1 reward and not moving
  with 0 reward.
  """

  def __init__(self, size: int = 5):
    """Constructor.

    Args:
      size: the integers between +/- size are used.
    """
    super().__init__(state_shape=(1,), action_space_size=3)
    self._size = size

    self._current_state = None
    self._num_actions_taken = None
    self.Reset()

  def Reset(self) -> base.State:
    self._current_state = numpy.random.randint(-self._size, self._size + 1)
    self._num_actions_taken = 0
    return numpy.array([[self._current_state]])

  def TakeAction(self, action: base.Action) -> base.Transition:
    current_state = self._current_state
    move = self.GetChoiceFromAction(action) - 1  # -1, 0, 1
    new_state = current_state + move

    s = numpy.array([[current_state]])
    a = action

    r = None
    if move == 0:
      r = 0
    else:
      if move == 1 and current_state < 0:
        r = 1
      elif move == -1 and current_state > 0:
        r = 1
      else:
        r = -1

    if new_state > self._size:
      sp = None
    elif new_state < -self._size:
      sp = None
    else:
      sp = numpy.array([[new_state]])

    if self._num_actions_taken >= STEP_LIMIT:
      sp = None

    self._current_state = new_state
    self._num_actions_taken += 1

    return base.Transition(s, a, r, sp)

  def GetAllStates(self):
    return numpy.vstack(
      [numpy.array([s]) for s in range(-self._size, self._size + 1)])
