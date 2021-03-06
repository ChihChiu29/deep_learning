"""Implementations for classes in q_learning_v3.py."""

from typing import Iterable

import numpy
from keras import layers
from keras import models

from lib.q_learning_v3 import *


class KerasModelQFunctionOneHotAction(QFunction):
    """A Q-Function implementation using a model built in Keras.

    This version uses one-hot vector for actions.
    """

    def __init__(
            self,
            env: Environment,
            num_nodes_in_layers: Iterable[int],
            learning_rate: float = None,
            discount_factor: float = None,
    ):
        """Constructor.

        Args:
            env: environment.
            num_nodes_in_layers: a list of how many nodes are used in each
                layer, starting from the input layter.
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)

        self._env = env
        self._state_array_size = env.GetStateArraySize()
        self._action_space_size = len(env.GetActionSpace())
        self._input_size = self._state_array_size + self._action_space_size

        self._model = _BuildModelOneHotAction(
            self._env.GetStateArraySize(),
            len(self._env.GetActionSpace()),
            num_nodes_in_layers)

        # For creating copy.
        self._num_nodes_in_layers = num_nodes_in_layers
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    # @Override
    def MakeCopy(self) -> 'KerasModelQFunctionOneHotAction':
        new_q_func = KerasModelQFunctionOneHotAction(
            self._env, self._num_nodes_in_layers,
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor)
        self.UpdateCopy(new_q_func)
        return new_q_func

    # @Override
    def UpdateCopy(self, target: 'KerasModelQFunctionOneHotAction') -> None:
        target._model.set_weights(self._model.get_weights())
        target.debug_verbosity = self.debug_verbosity

    def GetValue(
            self,
            state: State,
            action: Action,
    ) -> float:
        value = self._model.predict(self._GetStateActionArray(state, action))
        if self.debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value

    # @Override
    def SetValue(
            self,
            state: State,
            action: Action,
            new_value: float,
    ) -> None:
        if self.debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        return self._model.fit(
            self._GetStateActionArray(state, action), new_value, verbose=0)

    def _GetStateActionArray(
            self,
            state: State,
            action: Action,
    ) -> numpy.ndarray:
        """Creates a (state, action) array."""
        input_array = numpy.zeros((1, self._input_size))
        input_array[0, :self._state_array_size] = state
        # Use hot-spot for action.
        input_array[0, self._state_array_size + action] = 1.0
        return input_array


def _BuildModelOneHotAction(
        state_array_size: int,
        action_space_size: int,
        num_nodes_in_layers: Iterable[int],
) -> models.Model:
    """Builds a model with the given info; use one-hot vector for action."""
    input_size = state_array_size + action_space_size
    model = models.Sequential()
    model.add(
        layers.Dense(input_size, activation='relu', input_dim=input_size))
    for num_nodes in num_nodes_in_layers:
        model.add(layers.Dense(num_nodes, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='sgd', loss='mse')

    return model


class KerasModelQFunctionIntAction(QFunction):
    """A Q-Function implementation using a model built in Keras.

    This version uses an int for actions.
    """

    def __init__(
            self,
            env: Environment,
            num_nodes_in_layers: Iterable[int],
            learning_rate: float = None,
            discount_factor: float = None,
    ):
        """Constructor.

        Args:
            env: environment.
            num_nodes_in_layers: a list of how many nodes are used in each
                layer, starting from the input layter.
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)

        self._env = env
        self._state_array_size = env.GetStateArraySize()
        self._action_space_size = len(env.GetActionSpace())
        self._input_size = self._state_array_size + 1

        self._model = _BuildModelIntAction(
            self._env.GetStateArraySize(),
            num_nodes_in_layers)

        # For faster input vector creation.
        self._input_vector = numpy.zeros((1, self._input_size))

        # For creating copy.
        self._num_nodes_in_layers = num_nodes_in_layers
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    # @Override
    def MakeCopy(self) -> 'KerasModelQFunctionIntAction':
        new_q_func = KerasModelQFunctionIntAction(
            self._env, self._num_nodes_in_layers,
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor)
        self.UpdateCopy(new_q_func)
        return new_q_func

    # @Override
    def UpdateCopy(self, target: 'KerasModelQFunctionIntAction') -> None:
        target._model.set_weights(self._model.get_weights())
        target.debug_verbosity = self.debug_verbosity

    def GetValue(
            self,
            state: State,
            action: Action,
    ) -> float:
        value = self._model.predict(self._GetStateActionArray(state, action))
        if self.debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value

    # @Override
    def SetValue(
            self,
            state: State,
            action: Action,
            new_value: float,
    ) -> None:
        if self.debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        return self._model.fit(
            self._GetStateActionArray(state, action), new_value, verbose=0)

    def _GetStateActionArray(
            self,
            state: State,
            action: Action,
    ) -> numpy.ndarray:
        """Creates a (state, action) array."""
        self._input_vector[0, :self._state_array_size] = state
        self._input_vector[-1] = action
        return self._input_vector


def _BuildModelIntAction(
        state_array_size: int,
        num_nodes_in_layers: Iterable[int],
) -> models.Model:
    """Builds a model with the given info; using an int for action."""
    input_size = state_array_size + 1
    model = models.Sequential()
    model.add(
        layers.Dense(input_size, activation='relu', input_dim=input_size))
    for num_nodes in num_nodes_in_layers:
        model.add(layers.Dense(num_nodes, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='sgd', loss='mse')

    return model


class MultiModelQFunction(QFunction):
    """A Q-Function implementation using one model per action."""

    def __init__(
            self,
            env: Environment,
            num_nodes_in_shared_layers: Iterable[int],
            num_nodes_in_multi_head_layers: Iterable[int],
            activation: str = 'relu',
            learning_rate: float = None,
            discount_factor: float = None,
    ):
        """Constructor.

        Args:
            env: the environment.
            num_nodes_in_shared_layers: a list of how many nodes are used in
                each shared layer, starting from the input layter.
            num_nodes_in_multi_head_layers: a list of how many nodes are used
                in the rest of the model for each action, starting from the
                next layer after the last shared layer.
            activation: the activation function.
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)

        self._env = env
        self._models = _BuildMultiHeadModels(
            self._env.GetStateArraySize(),
            self._env.GetActionSpace(),
            num_nodes_in_shared_layers,
            num_nodes_in_multi_head_layers,
            activation)

        # For creating copy.
        self._num_nodes_in_shared_layers = num_nodes_in_shared_layers
        self._num_nodes_in_multi_head_layers = num_nodes_in_multi_head_layers
        self._activation = activation
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor

    # @Override
    def MakeCopy(self) -> 'MultiModelQFunction':
        new_q_func = MultiModelQFunction(
            self._env, self._num_nodes_in_shared_layers,
            self._num_nodes_in_multi_head_layers, activation=self._activation,
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor)
        self.UpdateCopy(new_q_func)
        return new_q_func

    # @Override
    def UpdateCopy(self, target: 'MultiModelQFunction') -> None:
        target.debug_verbosity = self.debug_verbosity
        for idx, _ in enumerate(self._models):
            target._models[idx].set_weights(self._models[idx].get_weights())

    # @Override
    def GetValue(
            self,
            state: State,
            action: Action,
    ) -> float:
        value = self._models[action].predict(state.reshape(1, state.size))
        if self.debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value

    # @Override
    def SetValue(
            self,
            state: State,
            action: Action,
            new_value: float,
    ) -> None:
        if self.debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        self._models[action].fit(
            state.reshape(1, state.size), new_value, verbose=0)


def _BuildMultiHeadModels(
        state_array_size: int,
        action_space: ActionSpace,
        num_nodes_in_shared_layers: Iterable[int],
        num_nodes_in_multi_head_layers: Iterable[int],
        activation: str,
) -> Tuple[models.Model]:
    """Builds a model with the given info."""
    shared_layers = []
    for num_nodes in num_nodes_in_shared_layers:
        shared_layers.append(layers.Dense(num_nodes, activation=activation))

    input_size = state_array_size
    action_models = []
    for action in action_space:
        model = models.Sequential()
        model.add(layers.Dense(
            input_size, activation=activation, input_dim=input_size))
        # First shared layers.
        for layer in shared_layers:
            model.add(layer)
        # Then separated layers.
        for num_nodes in num_nodes_in_multi_head_layers:
            model.add(layers.Dense(num_nodes, activation=activation))
        # Output layer.
        model.add(layers.Dense(1))
        model.compile(optimizer='sgd', loss='mse')
        action_models.append(model)

    return tuple(action_models)
