"""Key interfaces and functions used in Q-Learning.

See: https://en.wikipedia.org/wiki/Q-learning

This version has a QFuntion setup as described in:
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
which is more CPU friendly.
"""

from abc import ABC, abstractmethod

import numpy

from qpylib import t, numpy_util

DEFAULT_DISCOUNT_FACTOR = 0.9

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


class Environment(ABC):
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

  def GetActionArray(self, choice: int) -> Action:
    """Gets a one-hot vector for the action of the choice.

    Args:
      choice: an integer from 0 to action_space_size-1 indicating an action.
    """
    action = numpy.zeros((1, self._action_space_size))
    action[0, choice] = 1
    return action

  @abstractmethod
  def Reset(self):
    pass

  @abstractmethod
  def TakeAction(self, action: Action) -> Transition:
    pass


class QFunction(ABC):
  """A generic Q-function."""

  def __init__(
      self,
      discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
  ):
    self._gamma = discount_factor

  @abstractmethod
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

  @abstractmethod
  def SetValues(
      self,
      states: States,
      values: QValues,
  ) -> None:
    """Sets/trains Q values for states.

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
    self.SetValues(
      numpy_util.Replace(self.GetValues(states), actions, action_values))

  def GetNewValueFromTransition(
      self,
      state_t: State,
      action_t: Action,
      reward_t: Reward,
      state_t_plus_1: State,
      action_space: ActionSpace,
  ) -> float:
    """Gets a new values caused by a transition.

    Args:
        state_t: the state at t.
        action_t: the action to perform at t.
        reward_t: the direct reward as the result of (s_t, a_t).
        state_t_plus_1: the state to land at after action_t.
        action_space: the possible actions to take at state_t_plus_1.
    """
    estimated_best_future_value = max(
      self.GetValue(state_t_plus_1, action_t_plut_1)
      for action_t_plut_1 in action_space)

    return ((1.0 - self._alpha) * self.GetValue(state_t, action_t) +
            self._alpha * (
                reward_t + self._gamma * estimated_best_future_value))


class QFunctionPolicy(ABC):
  """The Policy that uses a Q-function to make decisions."""

  def __init__(self):
    self.debug_verbosity = 0

  @abstractmethod
  def Decide(
      self,
      q_function: QFunction,
      current_state: State,
      action_space: ActionSpace,
  ) -> Action:
    """Makes an decision using a QFunction."""
    pass


class CallbackFunctionInterface(ABC):

  @abstractmethod
  def Call(
      self,
      env: Environment,
      episode_idx: int,
      total_reward_last_episode: float,
      num_steps_last_episode: int,
  ) -> None:
    pass


def SimpleRun(
    env_factory: Callable[[], Environment],
    qfunc: QFunction,
    policy: QFunctionPolicy,
    num_of_episode: int,
    callback_func: CallbackFunctionInterface = None,
    debug_verbosity: int = 0,
):
  """Runs a simple simulation.

  The simulation runs multiple episodes. For episode, a new environment is
  created, and it is used until it is "done". Feedback is given to the model
  after each step.

  Args:
      env_factory: a factory function returns an environment. It is called
          for each episode.
      qfunc: a Q-Function.
      policy: a policy.
      num_of_episode: how many episodes to run.
      callback_func: a callback function invoked after every episode.
      debug_verbosity: what verbosity to use.
  """
  for episode_idx in range(num_of_episode):
    env = env_factory()
    env.debug_verbosity = debug_verbosity
    qfunc.debug_verbosity = debug_verbosity
    policy.debug_verbosity = debug_verbosity

    step_idx = 0
    total_reward = 0.0
    while True:
      try:
        s = env.GetState()
        a = policy.Decide(qfunc, s, env.GetActionSpace())
        r = env.TakeAction(a)
        s_new = env.GetState()
        total_reward += r

        qfunc.SetValue(s, a, qfunc.GetNewValueFromTransition(
          s, a, r, s_new, env.GetActionSpace()))
        step_idx += 1
      except EnvironmentDoneSignal:
        break

    callback_func.Call(env, episode_idx, total_reward, step_idx)


def DQNRun(
    env_factory: Callable[[], Environment],
    qfunc: QFunction,
    policy: QFunctionPolicy,
    num_of_episode: int,
    experience_history_capacity: int,
    num_training_samples: int,
    training_every_steps: int = 1,
    callback_func: CallbackFunctionInterface = None,
    debug_verbosity: int = 0,
):
  """Runs a simple simulation.

  The simulation runs multiple episodes. For episode, a new environment is
  created, and it is used until it is "done". Feedback is given to the model
  after each step.

  Args:
      env_factory: a factory function returns an environment. It is called
          for each episode.
      qfunc: a Q-Function.
      policy: a policy.
      num_of_episode: how many episodes to run.
      experience_history_capacity: how large is the experience history.
      num_training_samples: how many events to poll from history to train
          with.
      training_every_steps: how often to give feedback to the Q-functions.
      callback_func: a callback function invoked after every episode.
      debug_verbosity: what verbosity to use.
  """
  qfunc_snapshot = qfunc.MakeCopy()  # Used to update qfunc.
  experience_history = _ExperienceHistory(experience_history_capacity)
  for episode_idx in range(num_of_episode):
    env = env_factory()
    env.debug_verbosity = debug_verbosity
    qfunc.debug_verbosity = debug_verbosity
    policy.debug_verbosity = debug_verbosity

    step_idx = 0
    total_reward = 0.0
    while True:
      try:
        s = env.GetState()
        a = policy.Decide(qfunc, s, env.GetActionSpace())
        r = env.TakeAction(a)
        s_new = env.GetState()
        total_reward += r
        experience_history.AddEvent(s, a, s_new, r)

        if step_idx % training_every_steps == 0:
          # Update Q-Function.
          qfunc.UpdateCopy(qfunc_snapshot)
          for _ in range(num_training_samples):
            qfunc.SetValue(
              s, a, qfunc_snapshot.GetNewValueFromTransition(
                s, a, r, s_new, env.GetActionSpace()))

        step_idx += 1
      except EnvironmentDoneSignal:
        break

    callback_func.Call(env, episode_idx, total_reward, step_idx)


class _ExperienceHistory:
  """A fixed size history of experiences."""

  def __init__(self, capacity: int):
    """Constructor.

    Args:
        capacity: how many past events to save. If capacity is full,
            old events are discarded as new events are recorded.
    """
    self._capacity = capacity

    # Events inserted later are placed at tail.
    self._history = []

  def AddEvent(
      self,
      state: State,
      action: Action,
      new_state: State,
      reward: Reward,
  ) -> None:
    """Adds an event to history."""
    self._history.append((state, action, new_state, reward))
    if len(self._history) > self._capacity:
      self._history.pop(0)

  def Sample(
      self,
  ) -> Tuple[
    State,
    Action,
    State,
    Reward,
  ]:
    """Samples an event from the history.

    Returns:
        A tuple of (state_t, action_t, state_t_plus_1, reward).
    """
    return numpy.random.choice(self._history)
