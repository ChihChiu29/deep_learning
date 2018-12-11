"""Wraps around objects from OpenAI."""

from IPython import display
from matplotlib import pyplot

from lib import q_learning_v2


class OpenAiWrapperError():
    pass


class GymEnvironment(q_learning_v2.Environment):
    """Wrapper for Gym environment."""
    
    def __init__(self, gym_env):
        """Constructor.
        
        Args:
            gym_env: an environment made from `gym.make`.
        """
        if len(gym_env.observation_space.shape) != 1:
            raise OpenAiWrapperError('observation_space is not 1-d')
        
        super().__init__(
            state_array_size=gym_env.observation_space.shape[0],
            action_space_size=list(range(gym_env.action_space.n)))
        
        self._gym_env = gym_env
        
    #@ Override
    def TakeAction(self, action: q_learning_v2.Action) -> q_learning_v2.Reward:
        current_state = self.GetState()
        observation, reward, done, info = self._gym_env.step(action)
        new_state = observation
        
        if self.debug_verbosity >= 1:
            print('Action %s: (%s) -> (%s), reward: %s' % (
                action, current_state, new_state, reward))
        self._protected_SetState(new_state)
        return reward
        
    def PlotState(self):
        pyplot.imshow(env.render(mode='rgb_array'))
        display.clear_output(wait=True)
        display.display(plt.gcf())
