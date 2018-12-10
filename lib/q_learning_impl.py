"""Implementations for some Q-Learning."""

from typing import Dict, Iterable, Tuple

import numpy
from keras import models

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
    

NpArrayState = numpy.ndarray
NpArrayAction = numpy.ndarray
_NpArrayStateActionVector = numpy.ndarray
    
class KerasModelQFunction(q_learning.QFunction):
    """A Q-Function implementation using a model built in Keras.
    
    This function takes in a Keras model whose "fit" action is used to write
    to the model, and "predict" is used to train the model.
    
    In this setup both the state and the action are numpy arrays of any
    dimention. However when fed into the model they are flattened and
    concatenated into a 1-d array, so the model's input layer should be
    declared to have input_dim equals to the combined degree of freedom of
    the state and action space. The output layer should output a single float
    for the value.
    """
    
    def __init__(self, model: models.Model):
        super().__init__()
        self.debug = False  # Sets to True to print out stuff.
        
        self._model = model
        
    @staticmethod
    def _GetStateActionVector(
        state: NpArrayState,
        action: NpArrayAction,
    ) -> _NpArrayStateActionVector:
        """Prepares a [[state], [action]] vector suitable as an input."""
        return numpy.concatenate(
            (state.reshape(state.size),
             action.reshape(action.size)), axis=0).reshape(
                 1, state.size + action.size)
        
    # @Override
    def GetValue(
        self,
        state: NpArrayState,
        action: NpArrayAction,
    ) -> float:
        value = self._model.predict(self._GetStateActionVector(state, action))
        if self.debug:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value
        
    # @Override
    def _SetValue(
        self,
        state: NpArrayState,
        action: NpArrayAction,
        new_value: float,
    ) -> None:
        return self._model.fit(
            self._GetStateActionVector(state, action), new_value, verbose=0)
    
    
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


class MaxValueWithRandomnessPolicy(q_learning.Policy):
    """A policy that returns the action that yields the max value."""
    
    def __init__(self, certainty: float = 0.9):
        self._certainty = certainty
    
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

        if numpy.random.random() < self._certainty:
            return max_value_action
        else:
            return action_space[numpy.random.choice(range(len(action_space)))]
