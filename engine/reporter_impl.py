"""Additional reporters that can be used by a runner."""
import numpy
from matplotlib import pyplot

from deep_learning.engine import q_base
from deep_learning.engine.q_base import Environment
from deep_learning.engine.q_base import QFunction
from qpylib import logging
from qpylib import t


class ProgressTracer(q_base.Reporter):
  """Prints out info on running progress and rewards.

  The average rewards and steps are print out after every certain number of
  episodes specified by the user. The overall reward and step charts are
  generated at the end of a run, which can include previous runs as well.
  """

  def __init__(
      self,
      report_every_num_of_episodes: int = 100,
  ):
    """Ctor.

    Args:
      report_every_num_of_episodes: prints a report every this number of
        episodes.
    """
    self._report_every_num_of_episodes = report_every_num_of_episodes

    self._episode_rewards = []
    self._episode_steps = []

  def ReportEpisode(
      self,
      env: Environment,
      qfunc: QFunction,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int):
    """Reports episode progress and rewards."""
    self._episode_rewards.append(episode_reward)
    self._episode_steps.append(steps)

    episode_idx += 1  # make it 1-based.
    if episode_idx % self._report_every_num_of_episodes == 0:
      logging.printf(
        'Episode %d/%d: avg_reward = %3.2f, '
        'avg_steps=%3.2f (over %d episodes)',
        episode_idx,
        num_of_episodes,
        float(numpy.mean(
          self._episode_rewards[-self._report_every_num_of_episodes:])),
        float(numpy.mean(
          self._episode_steps[-self._report_every_num_of_episodes:])),
        self._report_every_num_of_episodes,
      )

  def FinalReport(
      self,
      env: Environment,
      qfunc: QFunction,
      num_of_episodes: int):
    logging.printf(
      'Total: run %d episodes, avg_reward = %3.2f, avg_steps=%3.2f',
      num_of_episodes,
      float(numpy.mean(
        self._episode_rewards[-num_of_episodes:])),
      float(numpy.mean(
        self._episode_steps[-num_of_episodes:])),
    )

    pyplot.title('Episode Rewards')
    pyplot.plot(self._episode_rewards)
    pyplot.show(block=False)

    pyplot.title('Episode Steps')
    pyplot.plot(self._episode_steps)
    pyplot.show(block=False)


class ValueTracer(q_base.Reporter):
  """Traces values of certain (s, a)s during a run.

  Report is only given after the whole run.
  """

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
  def ReportEpisode(
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

  # @Override
  def FinalReport(
      self,
      env: Environment,
      qfunc: QFunction,
      num_of_episodes: int,
  ):
    for a in self._actions:
      pyplot.title('Action: %d' % a)
      for s_values in self._value_traces[a].values():
        pyplot.plot(s_values)
      pyplot.show(block=False)
