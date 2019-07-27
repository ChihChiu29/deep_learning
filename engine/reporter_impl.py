"""Additional reporters that can be used by a runner."""
import numpy
from matplotlib import pyplot

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
      plot_every_num_of_episodes: int = 50,
  ):
    """Constructor.

    Args:
      trace_states: trace values of these states.
      trace_actions: the action choices to trace.
      plot_every_num_of_episodes: plot every this number of episodes.
    """
    self._states = numpy.concatenate(list(trace_states))  # type: q_base.States
    self._actions = list(trace_actions)
    self._plot_every_num_of_episodes = plot_every_num_of_episodes

    self._num_of_states = len(self._states)

    self._value_traces = {}  # {action: {state_idx: [values]}}
    for a in self._actions:
      a_values = {}
      for state_idx in range(self._num_of_states):
        a_values[state_idx] = []
      self._value_traces[a] = a_values

  # @Override
  def EndOfEpisodeReport(
      self,
      env: Environment,
      qfunc: QFunction,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int):
    values = qfunc.GetValues(self._states)
    for idx, v in enumerate(values):
      for a in self._actions:
        self._value_traces[a][idx].append(v[a])

    if (episode_idx + 1) % self._plot_every_num_of_episodes == 0:
      self._PlotValueTraces()

  # @Override
  def FinalReport(
      self,
      env: Environment,
      qfunc: QFunction,
      num_of_episodes: int,
  ):
    self._PlotValueTraces()

  def _PlotValueTraces(self):
    for a in self._actions:
      pyplot.title('Action: %d' % a)
      for s_values in self._value_traces[a].values():
        pyplot.plot(s_values)
      pyplot.show(block=False)
