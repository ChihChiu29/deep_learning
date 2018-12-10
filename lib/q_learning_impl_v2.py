"""Implementations for classes in q_learning_v2.py."""

from typing import Dict, Iterable, Tuple

import numpy
from keras import models

from lib import q_learning_v2


class KerasModelQFunction(q_learning_v2.QFunction):
    """A Q-Function implementation using a model built in Keras."""

    def __init__(
        self,
        state_array_size: int,
        action_space: q_learning_v2.ActionSpace,
        num_nodes_in_layers: Iterable[int],
        learning_rate: float = None,
        discount_factor: float = None,
    ):
        """Constructor.
        
        Args:
            state_array_size: the size of the state arrays.
            action_space: the action space.
            num_nodes_in_layers: a list of how many nodes are used in each
                layer, starting from the input layter.
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)
            
        self.state_array_size = state_array_size
        self._action_space = action_space
        self._model = _BuildClassifierModel(
            state_array_size, len(action_space), num_nodes_in_layers)
        
        self._debug_verbosity = 0
        self._input_array = numpy.zeros(state_array_size + 1)
        
    def SetDebugVerbosity(self, debug_verbosity: int) -> None:
        """Sets the debug verbosity, which controls the amount of output."""
        self._debug_verbosity = debug_verbosity

    # @Override
    def GetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
    ) -> float:
        value = self._model.predict(self._GetStateActionArray(state, action))
        if self._debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value
        
    # @Override
    def _SetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
        new_value: float,
    ) -> None:
        if self._debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        return self._model.fit(
            self._GetStateActionArray(state, action), new_value, verbose=0)
            
    def _GetStateActionArray(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
    ) -> numpy.ndarray:
        """Creates a (state, action) array."""
        self._input_array[:self._state_size] = state
        self._input_array[-1] = action
        return self._input_array
    

def _BuildClassifierModel(
    state_array_size: int,
    action_space_size: int,
    num_nodes_in_layers:Iterable[int],
) -> models.Model:
    """Builds a model with the given info."""
    input_size = state_array_size + 1
    model = Sequential()
    model.add(Dense(input_size, activation='relu', input_dim=input_size))
    for num_nodes in num_nodes_in_layers:
        model.add(Dense(num_nodes, activation='relu'))
    model.add(Dense(action_space_size, activation='softmax'))
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model    
    
    
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
