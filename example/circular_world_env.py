"""A circular world environment."""
import numpy

from deep_learning.engine import q_base

STEP_LIMIT = 500


class CircularWorld(q_base.Environment):
  """A circular world consists of integers between -n and n.

  Possible actions are go left (0), right (2), or stay (1). Going left from -n
  ends with n, and going right from n ends with -n. Any action getting closer
  to 0 is given +1 reward, with getting away given -1 reward and not moving
  with 0 reward.
  """

  def __init__(self, size: int = 5):
    """Constructor.

    Args:
      size: the integers between +/- size are used.
    """
    super().__init__(state_space_size=1, action_space_size=3)
    self._size = size

    self._current_state = 0
    self._num_actions_taken = None
    self.Reset()

  def Reset(self):
    self._current_state = 0
    self._num_actions_taken = 0

  def TakeAction(self, action: q_base.Action) -> q_base.Transition:
    current_state = self._current_state
    move = self.GetChoice(action) - 1  # -1, 0, 1
    new_state = current_state + move

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
      new_state = -self._size
    elif new_state < -self._size:
      new_state = self._size

    s = numpy.array([[current_state]])
    a = action
    if self._num_actions_taken >= STEP_LIMIT:
      sp = None
    else:
      sp = numpy.array([[new_state]])

    self._current_state = new_state
    self._num_actions_taken += 1

    return q_base.Transition(s, a, r, sp)
