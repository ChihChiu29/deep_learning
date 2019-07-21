"""Wraps around objects from OpenAI."""

from IPython import display
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from matplotlib import pyplot

from lib import q_learning_v3


class OpenAiWrapperError(Exception):
    pass


class OpenAiEnvironmentDone(OpenAiWrapperError):
    pass


class GymEnvironment(q_learning_v3.Environment):
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
        
        # For recording
        self._frames = None
        self._in_recording = False
        
    def ChangeSettings(
            self,
            continue_from_done: bool = False,
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
    def TakeAction(self, action: q_learning_v3.Action) -> q_learning_v3.Reward:
        if self._plot:
            self.PlotState()
        if self._in_recording:
            self._frames.append(self._gym_env.render(mode='rgb_array'))
            
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
                self._protected_SetDone(True)

        self._protected_SetState(new_state)
        return reward
        
    def PlotState(self):
        """Plots the current state."""
        pyplot.imshow(self._gym_env.render(mode='rgb_array'))
        if self.debug_verbosity < 15:
            display.clear_output(wait=True)
        display.display(pyplot.gcf())

    def StartRecording(self):
        """Starts to record a new animation; requires plot=True."""
        self._frames = []
        self._in_recording = True
        
    def StopRecording(self):
        """Stops recording."""
        self._in_recording = False
        
    def PlayRecording(self):
        """Plays the last recording."""
        _DisplayFramesAsGif(self._frames)
        
        
def _DisplayFramesAsGif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    pyplot.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = pyplot.imshow(frames[0])
    pyplot.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(pyplot.gcf(), animate, frames = len(frames), interval=50)
    display.display(display_animation(anim, default_mode='loop'))
