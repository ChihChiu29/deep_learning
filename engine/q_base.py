"""Q-learning base classes and types.

See: https://en.wikipedia.org/wiki/Q-learning

This version has a QFunction setup as described in:
https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
which is more CPU friendly.
"""
import abc

import numpy
from IPython import display
from matplotlib import pyplot

from qpylib import logging
from qpylib import numpy_util
from qpylib import t

DEFAULT_DISCOUNT_FACTOR = 0.9  # "gamma"
DEFAULT_LEARNING_RATE = 0.9  # "alpha"

# A state is a 1 x (state.shape) numpy array.
State = numpy.ndarray

# An m x (state.shape) numpy array holding m states.
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
      state_shape: t.Tuple[int],
      action_space_size: int,
  ):
    """Constructor.

    Args:
      state_shape: the shape of the numpy.ndarray state.
      action_space_size: the size of the discrete action space.
    """
    self._state_shape = state_shape
    self._action_space_size = action_space_size

  def GetStateShape(self) -> t.Tuple[int]:
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


class QFunction(abc.ABC):
  """A generic Q-function."""

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

  def GetValues(
      self,
      states: States,
  ) -> QValues:
    """Gets the Q values for states, for all actions."""
    values = self._protected_GetValues(states)
    logging.vlog(10, 'GET: (%s) -> %s', states, values)
    return values

  @abc.abstractmethod
  def _protected_GetValues(
      self,
      states: States,
  ) -> QValues:
    """Gets the Q values for states, for all actions."""
    pass

  def GetActionValues(
      self,
      values: QValues,
      actions: Actions,
  ) -> QActionValues:
    """Gets Q values for (state, action) pairs from Q values.

    The numbers of states and actions must equal. You should call GetValues
    to get the Q values. It is not done here automatically so that GetValues
    calls can be grouped for efficiency.
    """
    return numpy_util.SelectReduce(values, actions)

  def _SetValues(
      self,
      states: States,
      values: QValues,
  ) -> None:
    """Sets/trains Q values for states.

    This function is the one subclass uses to update the value storage. The
    runners use UpdateValuesFromTransitions to indirectly set values.

    The number of states and values must equal. Values for all actions are
    set at the same time.
    """
    logging.vlog(10, 'SET: (%s) <- %s', states, values)
    return self._protected_SetValues(states, values)

  @abc.abstractmethod
  def _protected_SetValues(
      self,
      states: States,
      values: QValues,
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
      action_values: QActionValues,
      values: QValues = None,
  ) -> None:
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
    """
    if values is None:
      values = self.GetValues(states)

    self._SetValues(
      states,
      numpy_util.Replace(values, actions, action_values))

  def UpdateValues(
      self,
      transitions: t.Iterable[Transition],
  ) -> None:
    """Update Q-values using the given set of transitions.

    Notes when there are multiple transitions from the same state
    (different actions), there is a conflict since the values for other actions
    for each transition is read from the current QFunction before update.
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
      self._SetActionValues(
        states, actions,
        ((1.0 - self._alpha) * old_action_values
         + self._alpha * learn_new_action_values),
        values=values)
    else:
      self._SetActionValues(states, actions, learn_new_action_values)


class Policy(abc.ABC):
  """The Policy that uses a Q-function to make decisions."""

  @abc.abstractmethod
  def Decide(
      self,
      env: Environment,
      qfunc: QFunction,
      state: State,
      episode_idx: int,
      num_of_episodes: int,
  ) -> Action:
    """Makes an decision using a QFunction.

    The episode info is provided to support policy that changes parameters
    over training.
    """
    pass


class Runner(abc.ABC):

  def __init__(self):
    self._episode_rewards = []

  @abc.abstractmethod
  def _protected_ProcessTransition(
      self,
      qfunc: QFunction,
      transition: Transition,
      step_idx: int,
  ) -> None:
    """Processes a new transition; e.g. to train the QFunction."""
    pass

  # @Final
  def Run(
      self,
      env: Environment,
      qfunc: QFunction,
      policy: Policy,
      num_of_episodes: int,
  ):
    """Runs an agent for some episodes.

    For each episode, the environment is reset first, then run until it's
    done. Between episodes, Report function is called to give user feedback.
    """
    for episode_idx in range(num_of_episodes):
      logging.vlog(5, 'Running episode: %d', episode_idx)

      s = env.Reset()
      step_idx = 0
      episode_reward = 0.0
      while True:
        logging.vlog(20, 'Running episode: %d, step: %d', episode_idx, step_idx)
        tran = env.TakeAction(
          policy.Decide(
            env=env,
            qfunc=qfunc,
            state=s,
            episode_idx=episode_idx,
            num_of_episodes=num_of_episodes,
          ))
        logging.vlog(15, str(tran))
        self._protected_ProcessTransition(
          qfunc=qfunc,
          transition=tran,
          step_idx=step_idx)
        episode_reward += tran.r
        s = tran.sp
        if tran.sp is None:
          break
        step_idx += 1

      self._protected_Report(
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
        episode_reward=episode_reward,
        steps=step_idx)

  def _protected_Report(
      self,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int,
  ):
    """Creates a report after an episode.

    Subclass can override this function to provide custom reports.

    Args:
      episode_idx: the index of the episode.
      num_of_episodes: the total number of episodes to run.
      episode_reward: reward for this episode.
      steps: number of steps in this episode.
    """
    self._episode_rewards.append(episode_reward)

    progress_msg = (
        'Episode %d/%d: episode_total_reward = %3.2f, episode_steps=%d' % (
      episode_idx + 1, num_of_episodes, episode_reward, steps))
    if episode_idx % 1000 == 0:
      logging.vlog(2, progress_msg)
    elif episode_idx % 100 == 0:
      logging.vlog(3, progress_msg)
    elif episode_idx % 10 == 0:
      logging.vlog(4, progress_msg)
    else:
      logging.vlog(5, progress_msg)

    if logging.ENV.debug_verbosity >= 4:
      display.clear_output(wait=True)
      pyplot.plot(self._episode_rewards)
      pyplot.show(block=False)
