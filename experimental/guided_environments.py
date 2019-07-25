"""A collection of modified environments that give the agent more guidance."""
import gym

from deep_learning.engine import environment_impl
from deep_learning.engine import q_base
from deep_learning.engine.q_base import Action
from deep_learning.engine.q_base import State
from deep_learning.engine.q_base import Transition


class GuidedMountainCar(q_base.Environment):
  """A modified environment that have more ways to give rewards.

  The Gym environment does not give positive reward when the car is at the
  bottom. This means that unless the agent is really lucky to get a score,
  no actual training would happen since there is no indication on what is
  the "right thing to do". This version rewards the agent slightly when it
  manages to get some momentum.
  """

  def __init__(self):
    self._original_env = environment_impl.GymEnvironment(
      gym.make('MountainCar-v0'))
    super().__init__(
      state_shape=self._original_env.GetStateShape(),
      action_space_size=self._original_env.GetActionSpaceSize())

  def Reset(self) -> State:
    return self._original_env.Reset()

  def TakeAction(self, action: Action) -> Transition:
    tran = self._original_env.TakeAction(action)
    if tran.sp is not None:
      # sp[1] is velocity, see:
      # https://github.com/openai/gym/wiki/MountainCarContinuous-v0#observation
      tran.r += abs(tran.sp[0][1])
    return tran
