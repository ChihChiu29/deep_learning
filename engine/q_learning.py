"""For Q-Learning, version 4.

See: https://en.wikipedia.org/wiki/Q-learning

This version has a QFuntion model setup as described in:
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
which is more CPU friendly.
"""

from abc import ABC, abstractmethod

import numpy

from qpylib import t

DEFAULT_DISCOUNT_FACTOR = 0.9

# A state is a n x 1 numpy array.
State = numpy.ndarray

# An action is represented with a k x 1 one-hot vector in the space that has
# dimension equals to the number of possible actions.
Action = numpy.ndarray

# A k x 1 vector, the i-th component is the Q value for the i-th action.
QValue = numpy.ndarray

# All rewards are floats.
Reward = float

# A value vector has dimension 1 x n.
Values = numpy.ndarray


class EnvironmentSignal(Exception):
  pass


class EnvironmentDoneSignal(EnvironmentSignal):
  """Signal for environment terminates."""
  pass


class Transition:
  """A transition of an agent in an enviroment."""

  def __init__(
      self,
      from_state: State,
      action: Action,
      reward: Reward,
      to_state: State,
  ):
    self.s = from_state
    self.a = action
    self.r = reward
    self.sp = to_state


class Environment(ABC):
  """A generic environment class."""

  def __init__(
      self,
      state_array_size: int,
      action_space_size: int,
  ):
    self._state_array_size = state_array_size
    self._action_space_size = action_space_size

    self._zero_state = numpy.zeros((self._state_array_size, 1))
    self._done = False

  def GetStateArraySize(self) -> int:
    """Gets the size of all state arrays (they are all 1-d)."""
    return self._state_array_size

  def GetActionSpaceSize(self) -> int:
    """Gets the action space, which is uniform per environment."""
    return self._action_space_size

  def GetState(self) -> State:
    """Gets the current state."""
    if self._done:
      raise EnvironmentDoneSignal()
    return self._state

  @abstractmethod
  def TakeAction(self, action: Action) -> Transition:
    """Takes an action, updates state."""
    pass

  def _protected_SetState(self, state: State) -> None:
    """Used by subclasses to set state."""
    self._state = state

  def _protected_SetDone(self, done: bool) -> None:
    """Used by subclasses to set done status."""
    self._done = done


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
      states: t.Iterable[State],
      actions: t.Iterable[Action],
  ) -> Values:
    """Gets the Q values for (s, a) pairs.

    The numbers of states and actions must equal. Returns a 1 x m
    vector for the Q function values, where m is the number of (s, a) pairs.
    """
    pass

  @abstractmethod
  def SetValues(
      self,
      states: t.Iterable[State],
      actions: t.Iterable[Action],
      values: Values,
  ) -> None:
    """Sets/trains the Q values for (s, a) pairs.

    The numbers of states, actions, and values must all be equal.
    """
    pass

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
