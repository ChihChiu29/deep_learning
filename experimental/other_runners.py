import numpy

from deep_learning.engine import base
from qpylib import logging
from qpylib import t
from qpylib.data_structure import sum_tree


class MultiEnvironmentRunner:
  """A runner that uses multiple environments."""

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
          episode_reward=episode_reward,
          steps=step_idx,
        )

    # All runs finished.
    for reporter in self._callbacks:
      reporter.OnCompletionCallback(
        env=None,
        brain=brain,
        num_of_episodes=num_of_episodes,
      )


# Doesn't seem to work (tried with DDQN). Keep in experimental for now.
class PrioritizedExperienceReplayRunner(base.Runner):
  """A runner that implements prioritized experience replay."""

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

    self._experience = _PrioritizedExperience(
      capacity=self._experience_capacity)

  # @Override
  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    # states, actions, action_values = brain.UpdateValues([transition])
    # err = numpy.sum(numpy.abs(
    #   brain.GetActionValues(brain.GetValues(states), actions) - action_values))
    # self._experience.AddTransition(transition, float(err))
    #
    # if step_idx % self._train_every_n_steps == 0:
    #   brain.UpdateValues(
    #     self._experience.Sample(self._experience_sample_batch_size))
    pass

  def SampleFromHistory(self, size: int) -> t.Iterable[base.Transition]:
    """Samples a set of transitions from experience history."""
    return self._experience.Sample(size)


class _PrioritizedExperience:
  """Fixed size experience with priorities.

  Ref:
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and
    -prioritized-experience-replay/
  """

  def __init__(
      self,
      capacity: int,
      epsilon: float = 0.01,
      alpha: float = 0.6,
  ):
    """Constructor.

    Args:
        capacity: how many past events to save. If capacity is full,
            old events are discarded as new events are recorded.
        epsilon: p = (err + epsilon)^alpha
        alpha: p = (err + epsilon)^alpha
    """
    self._capacity = capacity
    self._epsilon = epsilon
    self._alpha = alpha

    # Events inserted later are placed at tail.
    self._history = sum_tree.SumTree(capacity=self._capacity)

  def _CalculatePriority(self, error: float):
    # p = (err + epsilon)^alpha
    return numpy.power(numpy.abs(error) + self._epsilon, self._alpha)

  def AddTransition(
      self,
      transition: base.Transition,
      error: float,
  ) -> None:
    """Adds an event to history."""
    p = self._CalculatePriority(error)
    self._history.add(p, transition)

  def Sample(self, size: int) -> t.Iterable[base.Transition]:
    """Samples an event from the history."""
    # Gets a random number between 0 and total sum, then get the transition
    # corresponding to it.
    for s in numpy.random.uniform(0, self._history.total(), size=size):
      _, tran = self._history.get(s)
      yield tran
