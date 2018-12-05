"""For Q-Learning.

See: https://en.wikipedia.org/wiki/Q-learning
"""

from abc import ABC, abstractmethod
from typing import Iterable, List


class State:
    """A generic state."""
    pass


class Action:
    """A generic action."""
    pass


class Environment(ABC):
    """A generic environment class."""
    
    def __init__(self):
        self._current_state = None  # type: State
        self._last_action = None  # type: Action
        self._last_reward = 0.0  # type: float
        
    def GetCurrentState(self) -> State:
        return self._current_state
        
    def GetLastAction(self) -> Action:
        return self._last_action
        
    def GetLastReward(self) -> float:
        return self._last_reward
        
    @abstractmethod
    def GetActionSpace(self) -> List[Action]:
        """Returns a list of possible actions for the current state."""
        pass

    @abstractmethod
    def TakeAction(self, action: Action) -> None:
        """Takes an action, updates internal states."""
        pass


class QFunction(ABC):
    """A generic Q-function."""

    def __init__(self):
        self._alpha = 0.5
        self._gamma = 0.5
        
    def SetLearningRate(self, learning_rate: float) -> None:
        self._alpha = learning_rate
        assert 0 <= self._alpha <= 1
        
    def SetDiscountFactor(self, discount_factor: float) -> None:
        self._gamma = discount_factor
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
            reward_t: float,
            state_t_plus_1: State,
            action_space_t_plus_1: Iterable[State],
        ) -> None:
        """Updates values by a transition.
        
        Args:
            state_t: the state at t.
            action_t: the action to perform at t.
            reward_t: the direct reward as the result of (s_t, a_t).
            state_t_plus_1: the state to land at after action_t.
            action_space_t_plus_1: the possible actions to take for
                state_t_plus_1.
        """
        estimated_best_future_value = max(
            self.GetValue(state_t_plus_1, action_t_plut_1)
            for action_t_plut_1 in action_space_t_plus_1)
        
        self._SetValue(
            state_t,
            action_t,
            (1.0 - self._alpha) * self.GetValue(state_t, action_t) + 
            self._alpha * (
                reward_t + self._gamma * estimated_best_future_value))


class Policy(ABC):
    """The Policy interface."""
    
    @abstractmethod
    def Decide(
        self,
        q_function: QFunction,
        current_state: State,
        action_space: Iterable[Action],
    ) -> Action:
        """Makes an decision using a QFunction."""
        pass
