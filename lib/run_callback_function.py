"""Callback functions used with run methods in q_learning_v3.py."""

from IPython import display
from matplotlib import pyplot

from lib import q_learning_v3


class MonitoringCallback(q_learning_v3.CallbackFunctionInterface):

    def __init__(self):
        self._reward_history = []

    def GetRewardHistory(self):
        return self._reward_history

    # @Override
    def Call(
            self,
            env: q_learning_v3.Environment,
            episode_idx: int,
            total_reward_last_episode: float,
            num_steps_last_episode: int,
    ) -> None:
        self._reward_history.append((episode_idx, float(
            total_reward_last_episode) / num_steps_last_episode))
        self.PlotHistory()

    def PlotHistory(self):
        display.clear_output(wait=True)
        y = [v[1] for v in self._reward_history]
        x = [v[0] for v in self._reward_history]
        pyplot.clf()
        pyplot.plot(x, y)
        pyplot.title('Average Reward per Episode versus Episode Index')
        display.display(pyplot.gcf())
