"""Implementations for memoization based Q-functions in q_learning_v2.py."""

from typing import Dict, Iterable, List, Tuple

import numpy
from keras import layers
from keras import models

from lib import q_learning_v2


class MemoizationQFunction(q_learning_v2.QFunction):
    """A Q-Function memoizes values for (state, action)."""

    def __init__(
        self,
        env: q_learning_v2.Environment,
        learning_rate: float = None,
        discount_factor: float = None,
    ):
        """Constructor.
        
        Args:
            env: the environment.
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)
        
        self._env = env
            
        # The "actual" q-funtion. {(s, a): value}
        self._values = {}
        # When get values, use this value if the key does not exist.
        self._default_value = 0.0
        
    def ChangeSettings(
        self,
        default_value = 0.0,
    ) -> None:
        self._default_value = default_value
        
    # @Override
    def GetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
    ) -> float:
        value = self._values.get(
            self._GetStateActionHashKey(state, action), 0.0)
        if self.debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value
        
    # @Override
    def _SetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
        new_value: float,
    ) -> None:
        if self.debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        self._values[self._GetStateActionHashKey(state, action)] = new_value
            
    def _GetStateActionHashKey(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
    ) -> numpy.ndarray:
        """Gets a hashable (state, action) state."""
        return (str(state), action)
        
    # @Shadow
    def UpdateWithTransition(
        self,
        state_t: q_learning_v2.State,
        action_t: q_learning_v2.Action,
        reward_t: q_learning_v2.Reward,
        state_t_plus_1: q_learning_v2.State,
    ) -> None:
        """Updates values by a transition.
        
        Args:
            state_t: the state at t.
            action_t: the action to perform at t.
            reward_t: the direct reward as the result of (s_t, a_t).
            state_t_plus_1: the state to land at after action_t.
        """
        super().UpdateWithTransition(
            state_t, action_t, reward_t, state_t_plus_1,
            self._env.GetActionSpace())


class ContinuousMemoizationQFunction(q_learning_v2.QFunction):
    """A Q-Function dynnamically memoizes values for (state, action)."""

    def __init__(
        self,
        env: q_learning_v2.Environment,
        buffer_size: int = 500,
        learning_rate: float = None,
        discount_factor: float = None,
    ):
        """Constructor.
        
        Args:
            env: the environment.
            buffer_size: the size of the memoization buffers (per action).
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor)
        
        self._env = env
        # One value container per action.
        self._value_containers = [
            _ContinuousMemoizationContainer(buffer_size)
            for _ in env.GetActionSpace()]
            
        self.ChangeSettings()

    def ChangeSettings(
        self,
        debug_verbosity = 0,
        distance_threshold: float = 0.1,
        num_averaged_states: int = 5,      
        default_value = 0.0,
    ) -> None:
        self._default_value = default_value
        for container in self._value_containers:
            container.ChangeSettings(
                debug_verbosity = debug_verbosity,
                distance_threshold = distance_threshold,
                num_averaged_states = num_averaged_states,
                default_value = default_value)
        
    # @Override
    def GetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
    ) -> float:
        value = self._value_containers[action].Read(state)
        if self.debug_verbosity >= 5:
            print('GET: (%s, %s) -> %s' % (state, action, value))
        return value
        
    # @Override
    def _SetValue(
        self,
        state: q_learning_v2.State,
        action: q_learning_v2.Action,
        new_value: float,
    ) -> None:
        if self.debug_verbosity >= 5:
            print('SET: (%s, %s) <- %s' % (state, action, new_value))
        self._value_containers[action].Write(state, new_value)

    # @Shadow
    def UpdateWithTransition(
        self,
        state_t: q_learning_v2.State,
        action_t: q_learning_v2.Action,
        reward_t: q_learning_v2.Reward,
        state_t_plus_1: q_learning_v2.State,
    ) -> None:
        """Updates values by a transition.
        
        Args:
            state_t: the state at t.
            action_t: the action to perform at t.
            reward_t: the direct reward as the result of (s_t, a_t).
            state_t_plus_1: the state to land at after action_t.
        """
        super().UpdateWithTransition(
            state_t, action_t, reward_t, state_t_plus_1,
            self._env.GetActionSpace())


class _ContinuousMemoizedState:
    """A memoized state for _ContinuousMemoizationContainer."""
    
    def __init__(self, state: numpy.ndarray) -> None:
        self.state = state
        self.value = None  # float
        self.distance = None  # float, used for sorting


class _ContinuousMemoizationContainer:
    """Provides memoization for continuous numpy.ndarray states.
    
    The basic idea is this: two states are either "close" or "far".
    
    For two states A and B that are "close", assume we memoized value of A but
    not B, then when reading value for B we can return the value of A; when
    writing value for B we memoize value of B, then remove the memoization for
    A. When there are multiple memoized states close to B, we take average.
    
    For two states A and B that are "far", usually they have no interaction.
    But if we don't have any "close" states we can use to guess the value when
    reading, we'll use closest "far" states to provide a better guess.
    
    When states are finite, this memoization reduces to the usual memoization.
    """
    
    def __init__(
        self,
        buffer_size: int,
    ) -> None:
        """Constructor.
        
        Args:
            buffer_size: how large is the internal buffer.
        """
        self._buffer_size = buffer_size
        self._states = []  # Collection of _ContinuousMemoizedState.
        
        self.ChangeSettings()
        
    def ChangeSettings(
        self,
        debug_verbosity = 0,
        distance_threshold: float = 1.0,
        num_averaged_states: int = 5,
        default_value: float = 0.0,
    ) -> None:
        """Change settings.
        
        Args:
            debug_verbosity: debug verbosity.
            distance_threshold: for two states, when their distance is
                less/greater than this value, they are considered as
                "close"/"far".
            num_averaged_states: when averaging value over states, pick this
                number of state closest to the target.
        """
        self._debug_verbosity = debug_verbosity
        self._distance_threshold = distance_threshold
        self._num_averaged_states = num_averaged_states
        self._default_value = default_value
        
    def Read(self, state: numpy.ndarray) -> float:
        """Reads the value for a state."""
        close_states, far_states = self._OrderByDistance(state)
        if close_states:
            return numpy.average([s.value for s in close_states])
        else:
            return self._GuessValueFromFarStates(
                far_states[:self._num_averaged_states])
                
    def Write(self, state: numpy.ndarray, value: float) -> None:
        """Writes a new value for a state."""
        new_memoized_state = _ContinuousMemoizedState(state)
        new_memoized_state.value = value
        
        close_states, far_states = self._OrderByDistance(state)
        if close_states:
            # Replace all close states with the new one.
            self._states = far_states
            self._states.append(new_memoized_state)
        else:
            if len(self._states) < self._buffer_size:
                # There is room in the buffer, just add the new state.
                self._states.append(new_memoized_state)
            else:
                # Kick out the state with farrest distance.
                far_states[-1] = new_memoized_state
                self._states = close_states
                self._states.extend(far_states)
    
    def _OrderByDistance(
        self,
        target_state: numpy.ndarray,
    ) -> Tuple[List[_ContinuousMemoizedState], List[_ContinuousMemoizedState]]:
        """Order all states by distance to the target state.
        
        Args:
            target_state: the target state.
        
        Returns:
            Two lists of states. Each list is ordered by distance to the
            target_state from small to large values. The first list contains
            those states whose distance are less than the threshold; the second
            list contains states whose distance are greater than the threshold.
        """
        close_states, far_states = [], []
        for state in self._states:
            state.distance = numpy.linalg.norm(state.state - target_state)
            if state.distance < self._distance_threshold:
                close_states.append(state)
            else:
                far_states.append(state)
        close_states.sort(key=lambda s: s.distance)
        far_states.sort(key=lambda s: s.distance)
        return close_states, far_states
    
    def _GuessValueFromFarStates(
        self,
        states: Iterable[_ContinuousMemoizedState],
    ) -> float:
        """Gets an educated guess of a value from given far states."""
        if not states:
            return self._default_value
        
        dist_scale = -1.0 / self._distance_threshold
        overall_factor = numpy.exp(states[0].distance * dist_scale)
        
        weight_sum = 0.0
        partial_sum = 0.0
        for state in states:
            weight = numpy.exp(state.distance * dist_scale)
            weight_sum += weight
            partial_sum += state.value * weight
        return partial_sum / weight_sum


def TestContinuousMemoizationContainer():
    container = _ContinuousMemoizationContainer(5)
    container.ChangeSettings(
        debug_verbosity=10,
        distance_threshold=1.0,
        num_averaged_states=2,
    )
    
    # Read default value:
    print('expect 0.0: %s' % container.Read(numpy.array([1])))
    
    # Writes a new value:
    container.Write(numpy.array([1]), 2)
    print('expect 2.0: %s' % container.Read(numpy.array([1])))
    
    # Write with new value:
    container.Write(numpy.array([1]), 3)
    print('expect 3.0: %s' % container.Read(numpy.array([1])))
    
    # Write with new value with a close state:
    container.Write(numpy.array([1.1]), 4)
    print('expect 4.0: %s' % container.Read(numpy.array([1])))
    print('expect 1: %s' % len(container._states))
    
    # Read far value (no average):
    print('expect 4.0: %s' % container.Read(numpy.array([5.1])))
    print('expect 1: %s' % len(container._states))
    
    # Read far value (with average):
    container.Write(numpy.array([1]), 4)
    container.Write(numpy.array([5]), 2)
    print('expect 3.0: %s' % container.Read(numpy.array([3])))
    print('expect 2: %s' % len(container._states))
    
    # Buffer manipulation: remove close states
    container.Write(numpy.array([1.9]), 5)
    container.Write(numpy.array([2.8]), 6)
    container.Write(numpy.array([3.7]), 7)
    container.Write(numpy.array([4.1]), 8)
    print('expect 8.0: %s' % container.Read(numpy.array([4])))
    print('expect 1: %s' % len(container._states))
    
    # Buffer manipulation: far states
    container._states = []
    container.Write(numpy.array([1]), 1)
    container.Write(numpy.array([3]), 3)
    container.Write(numpy.array([5]), 5)
    container.Write(numpy.array([7]), 7)
    container.Write(numpy.array([9]), 9)
    container.Write(numpy.array([11]), 11)
    print('expect 3.xxx (some average): %s' % container.Read(numpy.array([1])))
    print('expect 5: %s' % len(container._states))    
    
    return container
