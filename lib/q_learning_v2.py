"""For Q-Learning, version 2.

See: https://en.wikipedia.org/wiki/Q-learning

This version has tighter interface definition, which primany targets
reinforcement learning with openai gym.
"""
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

import numpy

DEFAULT_LEARNING_RATE = 0.9
DEFAULT_DISCOUNT_FACTOR = 0.9


# A state is a 1-d numpy array.
State = numpy.ndarray
# An action is an integer.
Action = int
# An action space of size "n" contains actions from "0" to "n-1".
ActionSpace = range
# All rewards are floats.
Reward = float


class Environment(ABC):
    """A generic environment class."""
    
    def __init__(self, state_array_size: int, action_space_size: int):
        self.debug_verbosity = 0
        
        self._state_array_size = state_array_size
        self._action_space = range(action_space_size)
        
        self._state = numpy.zeros(state_array_size)
    
    def GetStateArraySize(self) -> int:
        """Gets the size of all state arrays (they are all 1-d)."""
        return self._state_array_size
    
    def GetActionSpace(self) -> ActionSpace:
        """Gets the action space, which is uniform per environment."""
        return self._action_space
    
    def GetState(self) -> State:
        """Gets the current state."""
        return self._state

    @abstractmethod
    def TakeAction(self, action: Action) -> Reward:
        """Takes an action, updates state."""
        pass
    
    def _protected_SetState(self, state: State) -> None:
        """Used by subclasses to set state."""
        self._state = state


class QFunction(ABC):
    """A generic Q-function."""

    def __init__(
        self,
        learning_rate: float = None,
        discount_factor: float = None,
    ):
        self.debug_verbosity = 0
        
        if learning_rate:
            self._alpha = learning_rate
        else:
            self._alpha = DEFAULT_LEARNING_RATE
            
        if discount_factor:
            self._gamma = discount_factor
        else:
            self._gamma = DEFAULT_DISCOUNT_FACTOR
        
        assert 0 <= self._alpha <= 1
        assert 0 <= self._gamma < 1
        
    @abstractmethod
    def GetValue(
        self,
        state: State,
        action: Action,
    ) -> float:
        """Gets the value for a (s, a) pair."""
        pass
    
    @abstractmethod
    def _SetValue(
        self,
        state: State,
        action: Action,
        new_value: float,
    ) -> None:
        """Sets the value for a (s, a) pair."""
        pass
    
    def UpdateWithTransition(
        self,
        state_t: State,
        action_t: Action,
        reward_t: Reward,
        state_t_plus_1: State,
        action_space: ActionSpace,
    ) -> None:
        """Updates values by a transition.
        
        Args:
            state_t: the state at t.
            action_t: the action to perform at t.
            reward_t: the direct reward as the result of (s_t, a_t).
            state_t_plus_1: the state to land at after action_t.
            action_space: the possible actions to take.
        """
        estimated_best_future_value = max(
            self.GetValue(state_t_plus_1, action_t_plut_1)
            for action_t_plut_1 in action_space)
        
        self._SetValue(
            state_t,
            action_t,
            (1.0 - self._alpha) * self.GetValue(state_t, action_t) + 
            self._alpha * (
                reward_t + self._gamma * estimated_best_future_value))


class Policy(ABC):
    """The Policy interface."""
    
    def __init__(self):
        self.debug_verbosity = 0
    
    @abstractmethod
    def Decide(
        self,
        q_function: QFunction,
        current_state: State,
        action_space: ActionSpace,
    ) -> Action:
        """Makes an decision using a QFunction."""
        pass


def Run(
    env: Environment,
    qfunc: QFunction,
    policy: Policy,
    num_of_runs: int = 10,
    debug_verbosity: int = 0):
    """Runs a training or testing loop.
    
    Args:
        env: an environment.
        qfunc: a Q-Function.
        policy: a policy.
        num_of_runs: how many runs this runs for.
        debug_verbosity: what verbosity to use
        
    """
    env.debug_verbosity = debug_verbosity
    qfunc.debug_verbosity = debug_verbosity
    policy.debug_verbosity = debug_verbosity
    for _ in range(num_of_runs):
        s = env.GetState()
        a = policy.Decide(qfunc, s, env.GetActionSpace())
        r = env.TakeAction(a)
        s_new = env.GetState()
        qfunc.UpdateWithTransition(s, a, r, s_new)
