"""Wraps around objects from OpenAI."""

from IPython import display
from matplotlib import pyplot

from lib import q_learning_v2


class OpenAiWrapperError(Exception):
    pass


class OpenAiEnvironmentDone(OpenAiWrapperError):
    pass


class GymEnvironment(q_learning_v2.Environment):
    """Wrapper for Gym environment."""
    
    def __init__(self, gym_env, reset=True):
        """Constructor.
        
        Args:
            gym_env: an environment made from `gym.make`.
            reset: if to reset/initialize the environment.
        """
        # Debug verbosity encoding:
        # 0: nothing
        # 1-9: debug output
        # 10: animation
        # 15: plots and logs
        
        if len(gym_env.observation_space.shape) != 1:
            raise OpenAiWrapperError('observation_space is not 1-d.')

        super().__init__(
            state_array_size=gym_env.observation_space.shape[0],
            action_space_size=gym_env.action_space.n)
        
        self._gym_env = gym_env
            
        if reset:
            self.Reset()

        self.ChangeSettings()
        
    def ChangeSettings(
        self,
        continue_from_done: bool = True,
        reward_when_done: float = 0.0,
        plot: bool = False,
    ) -> None:
        """Change settings."""
        self._continue_from_done = continue_from_done
        self._reward_when_done = reward_when_done
        self._plot = plot
        
    def Reset(self):
        self._gym_env.reset()
        
    #@ Override
    def TakeAction(self, action: q_learning_v2.Action) -> q_learning_v2.Reward:
        if self._plot:
            self.PlotState()
        current_state = self.GetState()
        observation, reward, done, info = self._gym_env.step(action)
        
        new_state = observation
        if self.debug_verbosity >= 1:
            print('Action %s: (%s) -> (%s), reward: %s' % (
                action, current_state, new_state, reward))
        
        if done:
            if self._continue_from_done:
                self._gym_env.reset()
                if self.debug_verbosity >= 1:
                    print('Environment reset, reward: %s' %
                          self._reward_when_done)
                return self._reward_when_done
            else:
                raise OpenAiEnvironmentDone('gym environment returned done.')

        self._protected_SetState(new_state)
        return reward
        
    def PlotState(self):
        """Plots the current state."""
        pyplot.imshow(self._gym_env.render(mode='rgb_array'))
        if self.debug_verbosity < 15:
            display.clear_output(wait=True)
        display.display(pyplot.gcf())
