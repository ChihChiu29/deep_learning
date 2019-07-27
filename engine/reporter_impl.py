"""Additional reporters that can be used by a runner."""
import numpy

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Environment
from deep_learning.engine.q_base import QFunction
from qpylib import t


class ValueTracer(q_base.Reporter):
  """Traces values of certain (s, a)s during a run."""

  def __init__(
      self,
      trace_states: t.Iterable[q_base.State],
      trace_actions: t.Iterable[int],
  ):
    """Constructor.

    Args:
      trace_states: trace values of these states.
      trace_actions: the action choices to trace.
    """
    self._states = numpy.concatenate(list(trace_states))  # type: q_base.States

    self._values = []

  # @Override
  def EndOfEpisodeReport(
      self,
      env: Environment,
      qfunc: QFunction,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int):
    pass

  # @Override
  def FinalReport(
      self,
      env: Environment,
      qfunc: QFunction,
      num_of_episodes: int,
  ):
    pass
