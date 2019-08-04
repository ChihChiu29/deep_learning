"""Q-learning base classes and types.

See: https://en.wikipedia.org/wiki/Q-learning

This version has a QFunction setup as described in:
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
which is more CPU friendly.
"""
import abc

import numpy

from qpylib import logging
from qpylib import numpy_util
from qpylib import t

# vlog convention (frequency is estimated over 1000 episodes):
# 1: logs once per 1000 episodes.
# 2: logs once per 500 episodes.
# 3: logs once per 200 episodes.
# 4: logs once per 200 episodes.
# 5: logs once per 100 episodes.
# 6: logs once per 50 episodes.
# 7: logs once per 10 episodes.
# 10: logs every episode.
# 20: logs 1 line per step.
# 25: logs <5 lines per step.

DEFAULT_DISCOUNT_FACTOR = 0.99  # "gamma"
DEFAULT_LEARNING_RATE = 0.9  # "alpha"

# A state is a 1 x (state.shape) numpy array.
State = numpy.ndarray

# An m x (state.shape) numpy array holding m states.
States = numpy.ndarray

# An action is a 1 x k one-hot vector, where k is the number of possible
# actions. The index of the position of the 1 in the vector is called a
# "choice".
Action = numpy.ndarray

# An m x k numpy array holding m states.
Actions = numpy.ndarray

# A value is a 1 x k vector, and the i-th component is the value for
# the i-th action (all values are for the same state). An example of value is
# the value of a Q-function for a given state.
Value = numpy.ndarray

# An m x k numpy array holding m values.
Values = numpy.ndarray

# A value for a state and an action.
ActionValue = float

# An (m,)-shape numpy array holding m action-values.
ActionValues = numpy.ndarray

# Reward from environment for a single step.
Reward = float

# An (m,)-shape numpy array holding m rewards.
Rewards = numpy.ndarray


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

  def __str__(self):
    return '(state %s, action %s) -> state %s: reward %3.2f' % (
      self.s, self.a, self.sp, self.r)


class Environment(abc.ABC):
  """A generic environment class."""

  def __init__(
      self,
      state_shape: t.Sequence[int],
      action_space_size: int,
  ):
    """Constructor.

    Args:
      state_shape: tuple of int, the shape of the numpy.ndarray state.
      action_space_size: the size of the discrete action space.
    """
    self._state_shape = state_shape
    self._action_space_size = action_space_size

  def GetStateShape(self) -> t.Sequence[int]:
    return self._state_shape

  def GetActionSpaceSize(self) -> int:
    return self._action_space_size

  def GetActionFromChoice(self, choice: int) -> Action:
    """Gets a one-hot vector for the action of the choice.

    Args:
      choice: an integer from 0 to action_space_size-1 indicating an action.
    """
    action = numpy.zeros((1, self._action_space_size))
    action[0, choice] = 1
    return action

  def GetRandomChoice(self) -> int:
    """Gets a random choice."""
    return numpy.random.randint(0, self.GetActionSpaceSize())

  def GetChoiceFromAction(self, action: Action) -> int:
    """Gets the int choice corresponding to the action."""
    return int(numpy.argmax(action))

  @abc.abstractmethod
  def Reset(self) -> State:
    """Resets an environment.

    This function is expected to be called first, which also returns the
    initial state in the environment.
    """
    pass

  @abc.abstractmethod
  def TakeAction(self, action: Action) -> Transition:
    """Returns the transition from an action.

    If transition.sp is None, it means the environment is done.
    """
    pass

  def TakeRandomAction(self) -> Transition:
    return self.TakeAction(self.GetActionFromChoice(self.GetRandomChoice()))


class Brain(abc.ABC):
  """A generic brain that learns from environment."""

  @abc.abstractmethod
  def GetValues(
      self,
      states: States,
  ) -> Values:
    """Gets the values for states, for all actions.

    For a given value, the index that has the largest value is the choice of
    the best action.
    """
    pass

  @staticmethod
  def GetActionValues(
      values: Values,
      actions: Actions,
  ) -> ActionValues:
    """Gets values for (state, action) pairs from values.

    The numbers of states and actions must equal. You should call GetValues
    to get the Q values. It is not done here automatically so that GetValues
    calls can be grouped for efficiency.
    """
    return numpy_util.SelectReduce(values, actions)

  @abc.abstractmethod
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[Transition],
  ) -> t.Tuple[States, Actions, ActionValues]:
    """Updates brain from a set of transitions.

    Notes when there are multiple transitions from the same state
    (different actions), there is a conflict since the values for other actions
    for each transition is read from the current QFunction before update.

    Args:
      transitions: an iterable of transitions to update values from.

    Returns:
      A tuple of (states, actions, target_action_values).
    """
    pass

  @staticmethod
  def CombineTransitions(
      transitions: t.Iterable[Transition],
  ) -> t.Tuple[States, Actions, Rewards, States, numpy.ndarray]:
    """Groups properties in transitions into arrays.

    State, action, reward, and new_state are all grouped into numpy arrays.
    A new numpy array that has the same shape of rewards is also generated

    Args:
      transitions: an iterable of transitions.

    Returns:
      A tuple of (states, actions, rewards, new_states, reward_mask).
      states: stacked states from all transitions.
      actions: stacked actions from all transitions.
      rewards: stacked rewards from all transitions.
      new_states: stacked new_states from all transitions. If a new_state is
        None, state from the same transition is used.
      reward_mask: an array having the same shape of rewards. Its i-th
        component is 1 if the corresponding new_state is not None, otherwise 0.
    """
    s_list = []  # type: t.List[State]
    a_list = []  # type: t.List[Action]
    r_list = []  # type: t.List[Reward]
    sp_list = []  # type: t.List[State]
    r_mask_list = []  # type: t.List[int]
    for idx, transition in enumerate(transitions):
      s_list.append(transition.s)
      a_list.append(transition.a)
      r_list.append(transition.r)
      if transition.sp is not None:
        sp_list.append(transition.sp)
        r_mask_list.append(1)
      else:
        sp_list.append(transition.s)
        r_mask_list.append(0)
    return (
      numpy.concatenate(s_list),
      numpy.concatenate(a_list),
      numpy.array(r_list),
      numpy.concatenate(sp_list),
      numpy.array(r_mask_list),
    )

  @abc.abstractmethod
  def Save(self, filepath: t.Text) -> None:
    """Saves the brain state (maybe partially) to a file."""
    pass

  @abc.abstractmethod
  def Load(self, filepath: t.Text) -> None:
    """Loads the brain state from the file saved by Save.

    The convention is that the parameters used to construct the brain is not
    guaranteed to be saved. The user is responsible to create an instance that
    has the save configuration as the saved one in order for load to work.
    """
    pass


class QFunction(Brain, abc.ABC):
  """A brain implemented using Q-value function."""

  def __init__(
      self,
      discount_factor: float = None,
      learning_rate: float = None,
  ):
    """Constructor.

    Args:
      discount_factor: gamma, discount factor, must be strictly less than 1.
        Affects iterations in UpdateValues.
      learning_rate: alpha, learning rate = 1 means completely ignores previous
        Q values during iteration in UpdateValues.
    """
    self._gamma = (
      discount_factor if discount_factor is not None
      else DEFAULT_DISCOUNT_FACTOR)
    self._alpha = (
      learning_rate if learning_rate is not None
      else DEFAULT_LEARNING_RATE)

  # @ Override
  def GetValues(
      self,
      states: States,
  ) -> Values:
    """Gets the Q values for states, for all actions."""
    values = self._protected_GetValues(states)
    logging.vlog(26, 'GET: (%s) -> %s', states, values)
    return values

  @abc.abstractmethod
  def _protected_GetValues(
      self,
      states: States,
  ) -> Values:
    """Gets the Q values for states, for all actions."""
    pass

  def _SetValues(
      self,
      states: States,
      values: Values,
  ) -> None:
    """Sets/trains Q values for states.

    This function is the one subclass uses to update the value storage. The
    runners use UpdateValuesFromTransitions to indirectly set values.

    The number of states and values must equal. Values for all actions are
    set at the same time.
    """
    logging.vlog(26, 'SET: (%s) <- %s', states, values)
    self._protected_SetValues(states, values)

  @abc.abstractmethod
  def _protected_SetValues(
      self,
      states: States,
      values: Values,
  ) -> None:
    """Sets/trains Q values for states.

    This function is the one subclass uses to update the value storage. The
    runners use UpdateValuesFromTransitions to indirectly set values.

    The number of states and values must equal. Values for all actions are
    set at the same time.
    """
    pass

  def _SetActionValues(
      self,
      states: States,
      actions: Actions,
      action_values: ActionValues,
      values: Values = None,
  ) -> t.Tuple[States, Actions, ActionValues]:
    """Sets/trains the Q values for (s, a) pairs.

    For each state, only one action is taken (the one with the same index in
    the actions array), and Q-values for other actions are fetched using
    GetValues with the current internal states (before set happens).

    The numbers of states, actions, and values must all be equal.

    Args:
      states: the states to set (s, a) values for.
      actions: the actions to set (s, a) values for.
      action_values: the new (s, a) values to set.
      values: if set, use this as the Q values instead of reading it using
        GetValues(states). This parameter is introduced in case Q values is
        already known, in which case passing it in is more efficient.

    Returns:
      The tuple of (states, actions, target_action_values).
    """
    if values is None:
      values = self.GetValues(states)

    self._SetValues(states, numpy_util.Replace(values, actions, action_values))
    return states, actions, action_values

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[Transition],
  ) -> t.Tuple[States, Actions, ActionValues]:
    """Update Q-values using the given set of transitions.

    Notes when there are multiple transitions from the same state
    (different actions), there is a conflict since the values for other actions
    for each transition is read from the current QFunction before update.

    Args:
      transitions: an iterable of transitions to update values from.

    Returns:
      The tuple of (states, actions, target_action_values).
    """
    s_list = []  # type: t.List[State]
    a_list = []  # type: t.List[Action]
    r_list = []  # type: t.List[Reward]
    sp_list = []  # type: t.List[State]
    done_sp_indices = []
    for idx, transition in enumerate(transitions):
      s_list.append(transition.s)
      a_list.append(transition.a)
      r_list.append(transition.r)
      if transition.sp is not None:
        sp_list.append(transition.sp)
      else:
        # If environment is done, max(Q*(sp,a)) is replaced by 0.
        sp_list.append(transition.s)
        done_sp_indices.append(idx)
    states, actions, rewards, new_states = (
      numpy.concatenate(s_list),
      numpy.concatenate(a_list),
      numpy.array(r_list),
      numpy.concatenate(sp_list),
    )
    # See: https://en.wikipedia.org/wiki/Q-learning
    # axis=1 because action is always assumed to be 1-dimensional.
    new_action_values = numpy.amax(self.GetValues(new_states), axis=1)
    for idx in done_sp_indices:
      new_action_values[idx] = 0.0
    learn_new_action_values = rewards + self._gamma * new_action_values

    if self._alpha < 0.9999999:
      values = self.GetValues(states)
      old_action_values = self.GetActionValues(values, actions)
      return self._SetActionValues(
        states, actions,
        ((1.0 - self._alpha) * old_action_values
         + self._alpha * learn_new_action_values),
        values=values)
    else:
      return self._SetActionValues(states, actions, learn_new_action_values)


class Policy(abc.ABC):
  """The Policy that uses a Q-function to make decisions."""

  @abc.abstractmethod
  def Decide(
      self,
      env: Environment,
      brain: Brain,
      state: State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> Action:
    """Makes an decision using a QFunction.

    The episode info is provided to support policy that changes parameters
    over training.
    """
    pass


class RunnerExtension(abc.ABC):
  """Used to extend a Runner."""

  @abc.abstractmethod
  def OnEpisodeFinishedCallback(
      self,
      env: Environment,
      brain: Brain,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int,
  ):
    """Called at the end of each episode."""
    pass

  @abc.abstractmethod
  def OnCompletionCallback(
      self,
      env: Environment,
      brain: Brain,
      num_of_episodes: int,
  ):
    """Called at the end of the runner.Run method."""
    pass


class Runner(abc.ABC):

  def __init__(self):
    self._callbacks = []

  @abc.abstractmethod
  def _protected_ProcessTransition(
      self,
      brain: Brain,
      transition: Transition,
      step_idx: int,
  ) -> None:
    """Processes a new transition; e.g. to train the QFunction."""
    pass

  def AddCallback(self, ext: RunnerExtension):
    """Adds a callback which extends Runner's ability."""
    self._callbacks.append(ext)

  def ClearCallbacks(self):
    """Removes all registered callbacks."""
    self._callbacks = []

  # @Final
  def Run(
      self,
      env: Environment,
      qfunc: Brain,
      policy: Policy,
      num_of_episodes: int,
  ):
    """Runs an agent for some episodes.

    For each episode, the environment is reset first, then run until it's
    done. Between episodes, Report function is called to give user feedback.
    """
    for episode_idx in range(num_of_episodes):
      logging.vlog(10, 'Running episode: %d', episode_idx)

      s = env.Reset()
      step_idx = 0
      episode_reward = 0.0
      while True:
        logging.vlog(20, 'Running episode: %d, step: %d', episode_idx, step_idx)
        tran = env.TakeAction(
          policy.Decide(
            env=env,
            brain=qfunc,
            state=s,
            episode_idx=episode_idx,
            num_of_episodes=num_of_episodes,
          ))
        logging.vlog(26, '%s', tran)
        self._protected_ProcessTransition(
          brain=qfunc,
          transition=tran,
          step_idx=step_idx)
        episode_reward += tran.r
        s = tran.sp
        if tran.sp is None:
          break
        step_idx += 1

      # Handle callback functions.
      for reporter in self._callbacks:
        reporter.OnEpisodeFinishedCallback(
          env=env,
          brain=qfunc,
          episode_idx=episode_idx,
          num_of_episodes=num_of_episodes,
          episode_reward=episode_reward,
          steps=step_idx,
        )

    # All runs finished.
    for reporter in self._callbacks:
      reporter.OnCompletionCallback(
        env=env,
        brain=qfunc,
        num_of_episodes=num_of_episodes,
      )
