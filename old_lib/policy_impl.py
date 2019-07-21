"""Implementations for policy in q_learning_v2.py."""

from typing import Dict, Iterable, List, Tuple

import numpy
from keras import layers
from keras import models

from lib import q_learning_v2

    
class MaxValuePolicy(q_learning_v2.Policy):
    """A policy that returns the action that yields the max value."""
    
    # @Override
    def Decide(
        self,
        q_function: q_learning_v2.QFunction,
        current_state: q_learning_v2.State,
        action_space: q_learning_v2.ActionSpace,
    ) -> q_learning_v2.Action:
        max_value = -numpy.inf
        max_value_action = None
        for action in action_space:
            try_value = q_function.GetValue(current_state, action)
            if try_value > max_value:
                max_value = try_value
                max_value_action = action
        return max_value_action


class RandomActionPolicy(q_learning_v2.Policy):
    """A policy that returns a random action."""
    
    # @Override
    def Decide(
        self,
        q_function: q_learning_v2.QFunction,
        current_state: q_learning_v2.State,
        action_space: q_learning_v2.ActionSpace,
    ) -> q_learning_v2.Action:
        return numpy.random.choice(action_space)


class MaxValueWithRandomnessPolicy(q_learning_v2.Policy):
    """A policy that returns the action that yields the max value."""
    
    def __init__(self, certainty: float = 0.9):
        super().__init__()
        
        self._certainty = certainty
    
    # @Override
    def Decide(
        self,
        q_function: q_learning_v2.QFunction,
        current_state: q_learning_v2.State,
        action_space: q_learning_v2.ActionSpace,
    ) -> q_learning_v2.Action:
        max_value = -numpy.inf
        max_value_action = None
        for action in action_space:
            try_value = q_function.GetValue(current_state, action)
            if try_value > max_value:
                max_value = try_value
                max_value_action = action

        if numpy.random.random() < self._certainty:
            return max_value_action
        else:
            if self.debug_verbosity >= 2:
                print('<use random choice>')
            return numpy.random.choice(action_space)