"""DL runner implementations.

Ref:
  https://en.wikipedia.org/wiki/Q-learning
  https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
"""

import numpy

from deep_learning.engine import base
from qpylib import logging
from qpylib import t


class NoOpRunner(base.Runner):
  """A runner that doesn't do anything to qfunc."""

  # @Override
  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    pass


class SimpleRunner(base.Runner):
  """A simple runner that updates the QFunction after each step."""

  # @Override
  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    brain.UpdateFromTransitions([transition])


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
    self._history = []  # type: t.List[base.Transition]

  def AddTransition(
      self,
      transition: base.Transition,
  ) -> None:
    """Adds an event to history."""
    self._history.append(transition)
    if len(self._history) > self._capacity:
      self._history.pop(0)

  def Sample(self, size: int) -> t.Iterable[base.Transition]:
    """Samples an event from the history."""
    # numpy.random.choice converts a list to numpy array first, which is very
    # inefficient, see:
    # https://stackoverflow.com/questions/18622781/why-is-random-choice-so-slow
    for idx in numpy.random.randint(0, len(self._history), size=size):
      yield self._history[idx]


class ExperienceReplayRunner(base.Runner):
  """A runner that implements experience replay."""

  def __init__(
      self,
      experience_capacity: int,
      experience_sample_batch_size: int,
      train_every_n_steps: int = 1,
  ):
    super().__init__()
    self._experience_capacity = experience_capacity
    self._experience_sample_batch_size = experience_sample_batch_size
    self._train_every_n_steps = train_every_n_steps

    self._experience = _Experience(capacity=self._experience_capacity)

  # @Override
  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    self._experience.AddTransition(transition)
    if step_idx % self._train_every_n_steps == 0:
      brain.UpdateFromTransitions(
        self._experience.Sample(self._experience_sample_batch_size))

  def SampleFromHistory(self, size: int) -> t.Iterable[base.Transition]:
    """Samples a set of transitions from experience history."""
    return self._experience.Sample(size)


class NStepExperienceRunner(base.Runner):
  """A runner the uses n-step experience."""

  def __init__(
      self,
      discount_factor: float = 0.99,
      n_step_return: int = 8,
  ):
    """Ctor.

    Args:
      discount_factor: the discount factor gamma.
      n_step_return: use this n-step-return.
    """
    super().__init__()
    self._gamma = discount_factor
    self._n_step_return = n_step_return

    gamma_power = 1.0
    gamma_powers = []
    for _ in range(self._n_step_return):
      gamma_powers.append(gamma_power)
      gamma_power *= self._gamma
    self._gamma_powers = numpy.array(gamma_powers)

    # Stores the last n_step_return transitions.
    self._memory = []  # type: t.List[base.Transition]

  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    train_transitions = []
    self._memory.append(transition)
    train_transitions.append(self._GetNStepTransition())

    if len(self._memory) >= self._n_step_return:
      self._memory.pop(0)

    if transition.sp is None:
      while self._memory:
        train_transitions.append(self._GetNStepTransition())
        self._memory.pop(0)
    brain.UpdateFromTransitions(train_transitions)

  def _GetNStepTransition(self) -> base.Transition:
    # This implementation takes 3.542e-06 sec per call.
    R = 0.0
    next_discount_factor = 1.0
    for tran in self._memory:
      R += tran.r * next_discount_factor
      next_discount_factor *= self._gamma

    # The commented implementation takes 7.322e-06 sec per call.
    # rewards = numpy.zeros(self._n_step_return)
    # for idx, tran in enumerate(self._memory):
    #   rewards[idx] = tran.r
    # R = numpy.sum(self._gamma_powers * rewards)

    return base.Transition(
      s=self._memory[0].s,
      a=self._memory[0].a,
      r=R,
      sp=self._memory[-1].sp,
    )


class SimpleMultiEnvironmentRunner:
  """A simple runner that uses multiple environments.

  It runs one environment for one step, then changes to another environment.
  Each step the brain is updated. An episode is finished when all environments
  are done.
  """

  def __init__(self):
    self._callbacks = []

  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    """Processes a new transition; e.g. to train the QFunction."""
    brain.UpdateFromTransitions([transition])

  def AddCallback(self, ext: base.RunnerExtension):
    """Adds a callback which extends Runner's ability."""
    self._callbacks.append(ext)

  def ClearCallbacks(self):
    """Removes all registered callbacks."""
    self._callbacks = []

  def Run(
      self,
      envs: t.Iterable[base.Environment],
      brain: base.Brain,
      policy: base.Policy,
      num_of_episodes: int,
  ):
    """Runs an agent for some episodes.

    For each episode, the environment is reset first, then run until it's
    done. Between episodes, Report function is called to give user feedback.
    """
    envs_list = list(envs)
    for episode_idx in range(num_of_episodes):
      logging.vlog(10, 'Running episode: %d', episode_idx)

      queue = [(env, env.Reset()) for env in envs_list]
      step_idx = 0
      episode_reward = 0.0
      while queue:
        env, s = queue.pop(0)
        logging.vlog(
          20, 'Running environment %s: episode: %d, step: %d',
          env, episode_idx, step_idx)
        tran = env.TakeAction(
          policy.Decide(
            env=env,
            brain=brain,
            state=s,
            episode_idx=episode_idx,
            num_of_episodes=num_of_episodes,
          ))
        logging.vlog(26, '%s', tran)
        self._protected_ProcessTransition(
          brain=brain,
          transition=tran,
          step_idx=step_idx)
        episode_reward += tran.r
        if tran.sp is not None:
          queue.append((env, tran.sp))
        step_idx += 1

      # Handle callback functions.
      for reporter in self._callbacks:
        reporter.OnEpisodeFinishedCallback(
          env=None,
          brain=brain,
          episode_idx=episode_idx,
          num_of_episodes=num_of_episodes,
          episode_reward=episode_reward / len(envs_list),
          steps=float(step_idx) / len(envs_list),
        )

    # All runs finished.
    for reporter in self._callbacks:
      reporter.OnCompletionCallback(
        env=None,
        brain=brain,
        num_of_episodes=num_of_episodes,
      )
