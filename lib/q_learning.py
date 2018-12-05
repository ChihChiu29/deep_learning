"""For Q-Learning.

See: https://en.wikipedia.org/wiki/Q-learning
"""

from abc import ABC, abstractmethod
from typing import Iterable


class State:
    pass

class Action:
    pass


class QFunction(ABC):

    def __init__(
            self, 
            learning_rate: float,
            discount_factor: float,
        ):
        self._alpha = learning_rate
        self._gamma = discount_factor
        
        assert 0 <= self._alpha <= 1
        assert 0 <= self._gamma < 1
        
        self._1_minus_alpha = 1.0 - self._alpha
        
    
    @abstractmethod
    def GetValue(
            self,
            state: State,
            action: Action,
        ) -> float:
        """Gets the value for a (s, a) pair."""
        pass
    
    @abstractmethod
    def SetValue(
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
        
        self.SetValue(
            state_t_plus_1,
            self._1_minus_alpha * GetValue(state_t, action_t) + 
            self._alpha * (
                reward_t + self._gamma * estimated_best_future_value))


class Policy(ABC):
    
    @abstractmethod
    def Decide(
        self,
        q_function: QFunction,
        current_state: State,
        action_space: Iterable[Action],
    ) -> Action:
        """Makes an decision using a QFunction."""
        pass
