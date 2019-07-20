"""DL runner.

Ref:
  https://en.wikipedia.org/wiki/Q-learning
  https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
"""
import abc

import numpy

from deep_learning.engine import q_base
from qpylib import t


class _Experience:
  """A fixed size history of experiences."""

  def __init__(self, capacity: int):
    """Constructor.

    Args:
        capacity: how many past events to save. If capacity is full,
            old events are discarded as new events are recorded.
    """
    self._capacity = capacity

    # Events inserted later are placed at tail.
    self._history = []  # type: t.List[q_base.Transition]

  def AddEvent(
      self,
      transition: q_base.Transition,
  ) -> None:
    """Adds an event to history."""
    self._history.append(transition)
    if len(self._history) > self._capacity:
      self._history.pop(0)

  def Sample(self) -> t.Iterable[q_base.Transition]:
    """Samples an event from the history."""
    return numpy.random.choice(self._history)


class CallbackFunctionInterface(abc.ABC):

  @abc.abstractmethod
  def Call(
      self,
      env: q_base.Environment,
      episode_idx: int,
      total_reward_last_episode: float,
      num_steps_last_episode: int,
  ) -> None:
    pass


def SimpleRun(
    env_factory: q_base.Environment,
    qfunc: q_base.QFunction,
    policy: q_base.Policy,
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


def DQN(
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
