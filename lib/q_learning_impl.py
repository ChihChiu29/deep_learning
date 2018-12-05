"""Implementations for some Q-Learning."""

from typing import Dict, Iterable, Tuple

import numpy

from lib import q_learning


class HashableState(q_learning.State):
    """Represents a hashable state (not checked)."""
    pass


class HashableAction(q_learning.Action):
    """Represents a hashable action (not checked)."""
    pass


class FiniteStateQFunction(q_learning.QFunction):
    """A Q-Function implementation based on memoization.
    
    Assumes there the {(s, a)} space is a discrete and finite.
    """
    
    def __init__(self):
        super().__init__()
        
        # type: Dict[Tuple[HashableState, HashableAction], float]
        self._values = {}
        
    @staticmethod
    def Hash(state: HashableState, action: HashableAction) -> int:
        return hash((state, action))
        
    # @Override
    def GetValue(
            self,
            state: HashableState,
            action: HashableAction,
        ) -> float:
        return self._values.setdefault(self.Hash(state, action), 0.0)
        
    # @Override
    def _SetValue(
            self,
            state: HashableState,
            action: HashableAction,
            new_value: float,
        ) -> None:
        self._values[self.Hash(state, action)] = new_value
    
    
class MaxValuePolicy(q_learning.Policy):
    """A policy that returns the action that yields the max value."""
    
    # @Override
    def Decide(
        self,
        q_function: q_learning.QFunction,
        current_state: q_learning.State,
        action_space: Iterable[q_learning.Action],
    ) -> q_learning.Action:
        max_value = -numpy.inf
        max_value_action = None
        for action in action_space:
            try_value = q_function.GetValue(current_state, action)
            if try_value > max_value:
                max_value = try_value
                max_value_action = action
        return max_value_action
