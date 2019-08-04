"""DL runner implementations.

Ref:
  https://en.wikipedia.org/wiki/Q-learning
  https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
"""

import numpy

from deep_learning.engine import q_base
from qpylib import t


class NoOpRunner(q_base.Runner):
  """A runner that doesn't do anything to qfunc."""

  # @Override
  def _protected_ProcessTransition(
      self,
      qfunc: q_base.Brain,
      transition: q_base.Transition,
      step_idx: int,
  ) -> None:
    pass


class SimpleRunner(q_base.Runner):
  """A simple runner that updates the QFunction after each step."""

  # @Override
  def _protected_ProcessTransition(
      self,
      qfunc: q_base.Brain,
      transition: q_base.Transition,
      step_idx: int,
  ) -> None:
    qfunc.UpdateValues([transition])


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

  def AddTransition(
      self,
      transition: q_base.Transition,
  ) -> None:
    """Adds an event to history."""
    self._history.append(transition)
    if len(self._history) > self._capacity:
      self._history.pop(0)

  def Sample(self, size: int) -> t.Iterable[q_base.Transition]:
    """Samples an event from the history."""
    # numpy.random.choice converts a list to numpy array first, which is very
    # inefficient, see:
    # https://stackoverflow.com/questions/18622781/why-is-random-choice-so-slow
    for idx in numpy.random.randint(0, len(self._history), size=size):
      yield self._history[idx]


class ExperienceReplayRunner(q_base.Runner):
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
      qfunc: q_base.Brain,
      transition: q_base.Transition,
      step_idx: int,
  ) -> None:
    self._experience.AddTransition(transition)
    if step_idx % self._train_every_n_steps == 0:
      qfunc.UpdateValues(
        self._experience.Sample(self._experience_sample_batch_size))

  def SampleFromHistory(self, size: int) -> t.Iterable[q_base.Transition]:
    """Samples a set of transitions from experience history."""
    return self._experience.Sample(size)
