"""Q-learning base classes and types.

See: https://en.wikipedia.org/wiki/Q-learning

This version has a QFunction setup as described in:
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
which is more CPU friendly.
"""
import abc

import numpy

from qpylib import t, numpy_util

DEFAULT_DISCOUNT_FACTOR = 0.9  # aka "gamma"

# A state is a 1 x n numpy array, where n is the dimension of the state vector.
State = numpy.ndarray

# An m x n numpy array holding m states.
States = numpy.ndarray

# An action is a 1 x k one-hot vector, where k is the number of possible
# actions.
Action = numpy.ndarray

# An m x k numpy array holding m states.
Actions = numpy.ndarray

# A Q-value is a 1 x k vector, and the i-th component is the Q value for
# the i-th action (all values are for the same state).
QValue = numpy.ndarray

# An m x k numpy array holding m Q-values.
QValues = numpy.ndarray

# Q-value for a state and an action.
QActionValue = float

# An (m,)-shape numpy array holding m Q-action-values.
QActionValues = numpy.ndarray

# Reward from environment for a single step.
Reward = float

# An (m,)-shape numpy array holding m rewards.
Rewards = numpy.ndarray


class EnvironmentDoneSignal(Exception):
  """Signal for environment terminates."""
  pass


class Transition:
  """A transition of an agent in an environment."""

  def __init__(
      self,
      s: State,
      a: Action,
      r: Reward,
      sp: t.Optional[State],
  ):
    """Constructor.

    Args:
      s: the state before the transition.
      a: the action that caused the transition.
      r: the reward for the transition.
      sp: the new state after the transition. If it's None, the environment
        needs to be reset.
    """
    self.s = s
    self.a = a
    self.r = r
    self.sp = sp


class Environment(abc.ABC):
  """A generic environment class."""

  def __init__(
      self,
      state_array_size: int,
      action_space_size: int,
  ):
    self._state_array_size = state_array_size
    self._action_space_size = action_space_size

  def GetStateArraySize(self) -> int:
    """Gets the size of all state arrays (they are all 1-d)."""
    return self._state_array_size

  def GetActionSpaceSize(self) -> int:
    """Gets the action space, which is uniform per environment."""
    return self._action_space_size

  def GetAction(self, choice: int) -> Action:
    """Gets a one-hot vector for the action of the choice.

    Args:
      choice: an integer from 0 to action_space_size-1 indicating an action.
    """
    action = numpy.zeros((1, self._action_space_size))
    action[0, choice] = 1
    return action

  @abc.abstractmethod
  def Reset(self):
    pass

  @abc.abstractmethod
  def TakeAction(self, action: Action) -> Transition:
    pass


class QFunction(abc.ABC):
  """A generic Q-function."""

  @abc.abstractmethod
  def GetValues(
      self,
      states: States,
  ) -> QValues:
    """Gets the Q values for states, for all actions."""
    pass

  def GetActionValues(
      self,
      states: States,
      actions: Actions,
  ) -> QActionValues:
    """Gets Q values for (state, action) pairs.

    The numbers of states and actions must equal.
    """
    return numpy_util.SelectReduce(self.GetValues(states), actions)

  @abc.abstractmethod
  def _protected_SetValues(
      self,
      states: States,
      values: QValues,
  ) -> None:
    """Sets/trains Q values for states.

    This function is the one subclass uses to update the value storage. The
    runners use SetActionValues to indirectly set values.

    The number of states and values must equal. Values for all actions are
    set at the same time.
    """
    pass

  def SetActionValues(
      self,
      states: States,
      actions: Actions,
      action_values: QActionValues,
  ) -> None:
    """Sets/trains the Q values for (s, a) pairs.

    For each state, only one action is taken (the one with the same index in
    the actions array), and Q-values for other actions are fetched using
    GetValues with the current internal states (before set happens).

    The numbers of states, actions, and values must all be equal.
    """
    self._protected_SetValues(
      states,
      numpy_util.Replace(self.GetValues(states), actions, action_values))

  def UpdateValuesFromTransitions(
      self,
      transitions: t.Iterable[Transition],
      discount_factor: float = None
  ) -> None:
    """Update Q-values using the given set of transitions."""
    if not discount_factor:
      discount_factor = DEFAULT_DISCOUNT_FACTOR

    s_list = []  # type: t.List[State]
    a_list = []  # type: t.List[Action]
    r_list = []  # type: t.List[Reward]
    sp_list = []  # type: t.List[State]
    for transition in transitions:
      s_list.append(transition.s)
      a_list.append(transition.a)
      r_list.append(transition.r)
      sp_list.append(transition.sp)
    states, actions, rewards, new_states = (
      numpy.concatenate(s_list),
      numpy.concatenate(a_list),
      numpy.concatenate(r_list),
      numpy.concatenate(sp_list),
    )
    new_action_values = rewards + discount_factor * numpy.amax(
      self.GetValues(new_states), axis=1)
    self.SetActionValues(states, actions, new_action_values)


class Policy(abc.ABC):
  """The Policy that uses a Q-function to make decisions."""

  @abc.abstractmethod
  def Decide(
      self,
      env: Environment,
      q_function: QFunction,
      state: State,
  ) -> Action:
    """Makes an decision using a QFunction."""
    pass
