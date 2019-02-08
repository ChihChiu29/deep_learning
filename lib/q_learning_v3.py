"""For Q-Learning, version 3.

See: https://en.wikipedia.org/wiki/Q-learning

This version has tighter interface definition, which primarily targets
reinforcement learning with openai gym.
"""
from abc import ABC, abstractmethod
from typing import Callable, Tuple

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


class EnvironmentSignal(Exception):
    pass


class EnvironmentDoneSignal(EnvironmentSignal):
    pass


class Environment(ABC):
    """A generic environment class."""

    def __init__(
            self,
            state_array_size: int,
            action_space_size: int,
    ):
        self.debug_verbosity = 0

        self._state_array_size = state_array_size
        self._action_space = range(action_space_size)

        self._state = numpy.zeros(self._state_array_size)
        self._done = False

    def GetStateArraySize(self) -> int:
        """Gets the size of all state arrays (they are all 1-d)."""
        return self._state_array_size

    def GetActionSpace(self) -> ActionSpace:
        """Gets the action space, which is uniform per environment."""
        return self._action_space

    def GetState(self) -> State:
        """Gets the current state."""
        if self._done:
            raise EnvironmentDoneSignal()
        return self._state

    @abstractmethod
    def TakeAction(self, action: Action) -> Reward:
        """Takes an action, updates state."""
        pass

    def _protected_SetState(self, state: State) -> None:
        """Used by subclasses to set state."""
        self._state = state

    def _protected_SetDone(self, done: bool) -> None:
        """Used by subclasses to set done status."""
        self._done = done


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
    def MakeCopy(self) -> 'QFunction':
        """Creates a copy of self."""
        pass

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

    def GetNewValueFromTransition(
            self,
            state_t: State,
            action_t: Action,
            reward_t: Reward,
            state_t_plus_1: State,
            action_space: ActionSpace,
    ) -> float:
        """Gets a new values caused by a transition.

        Args:
            state_t: the state at t.
            action_t: the action to perform at t.
            reward_t: the direct reward as the result of (s_t, a_t).
            state_t_plus_1: the state to land at after action_t.
            action_space: the possible actions to take at state_t_plus_1.
        """
        estimated_best_future_value = max(
            self.GetValue(state_t_plus_1, action_t_plut_1)
            for action_t_plut_1 in action_space)

        return ((1.0 - self._alpha) * self.GetValue(state_t, action_t) +
                self._alpha * (
                        reward_t + self._gamma * estimated_best_future_value))


class QFunctionPolicy(ABC):
    """The Policy that uses a Q-function to make decisions."""

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


class CallbackFunctionInterface(ABC):

    @abstractmethod
    def Call(
            self,
            env: Environment,
            episode_idx: int,
            total_reward_last_episode: float,
            num_steps_last_episode: int,
    ) -> None:
        pass


def SimpleRun(
        env_factory: Callable[[], Environment],
        qfunc: QFunction,
        policy: QFunctionPolicy,
        num_of_episode: int,
        callback_func: CallbackFunctionInterface = None,
        debug_verbosity: int = 0,
):
    """Runs a simple simulation.

    The simulation runs multiple episodes. For episode, a new environment is
    created, and it is used until it is "done". Feedback is given to the model
    after each step.

    Args:
        env_factory: a factory function returns an environment. It is called
            for each episode.
        qfunc: a Q-Function.
        policy: a policy.
        num_of_episode: how many episodes to run.
        callback_func: a callback function invoked after every episode.
        debug_verbosity: what verbosity to use.
    """
    for episode_idx in range(num_of_episode):
        env = env_factory()
        env.debug_verbosity = debug_verbosity
        qfunc.debug_verbosity = debug_verbosity
        policy.debug_verbosity = debug_verbosity

        step_idx = 0
        total_reward = 0.0
        while True:
            try:
                s = env.GetState()
                a = policy.Decide(qfunc, s, env.GetActionSpace())
                r = env.TakeAction(a)
                s_new = env.GetState()
                total_reward += r

                qfunc.SetValue(s, a, qfunc.GetNewValueFromTransition(
                    s, a, r, s_new, env.GetActionSpace()))
                step_idx += 1
            except EnvironmentDoneSignal:
                break

        callback_func.Call(env, episode_idx, total_reward, step_idx)


def DQNRun(
        env_factory: Callable[[], Environment],
        qfunc: QFunction,
        policy: QFunctionPolicy,
        num_of_episode: int,
        experience_history_capacity: int,
        num_training_samples: int,
        training_every_steps: int = 1,
        callback_func: CallbackFunctionInterface = None,
        debug_verbosity: int = 0,
):
    """Runs a simple simulation.

    The simulation runs multiple episodes. For episode, a new environment is
    created, and it is used until it is "done". Feedback is given to the model
    after each step.

    Args:
        env_factory: a factory function returns an environment. It is called
            for each episode.
        qfunc: a Q-Function.
        policy: a policy.
        num_of_episode: how many episodes to run.
        experience_history_capacity: how large is the experience history.
        num_training_samples: how many events to poll from history to train
            with.
        training_every_steps: how often to give feedback to the Q-functions.
        callback_func: a callback function invoked after every episode.
        debug_verbosity: what verbosity to use.
    """
    experience_history = _ExperienceHistory(experience_history_capacity)
    for episode_idx in range(num_of_episode):
        env = env_factory()
        env.debug_verbosity = debug_verbosity
        qfunc.debug_verbosity = debug_verbosity
        policy.debug_verbosity = debug_verbosity

        step_idx = 0
        total_reward = 0.0
        while True:
            try:
                s = env.GetState()
                a = policy.Decide(qfunc, s, env.GetActionSpace())
                r = env.TakeAction(a)
                s_new = env.GetState()
                total_reward += r
                experience_history.AddEvent(s, a, s_new, r)

                if step_idx % training_every_steps == 0:
                    # Update Q-Function.
                    snapshot = qfunc.MakeCopy()
                    for _ in range(num_training_samples):
                        qfunc.SetValue(s, a, snapshot.GetNewValueFromTransition(
                            s, a, r, s_new, env.GetActionSpace()))

                step_idx += 1
            except EnvironmentDoneSignal:
                break

        callback_func.Call(env, episode_idx, total_reward, step_idx)


class _ExperienceHistory:
    """A fixed size history of experiences."""

    def __init__(self, capacity: int):
        """Constructor.

        Args:
            capacity: how many past events to save. If capacity is full,
                old events are discarded as new events are recorded.
        """
        self._capacity = capacity

        # Events inserted later are placed at tail.
        self._history = []

    def AddEvent(
            self,
            state: State,
            action: Action,
            new_state: State,
            reward: Reward,
    ) -> None:
        """Adds an event to history."""
        self._history.append((state, action, new_state, reward))
        if len(self._history) > self._capacity:
            self._history.pop(0)

    def Sample(
            self,
    ) -> Tuple[
        State,
        Action,
        State,
        Reward,
    ]:
        """Samples an event from the history.

        Returns:
            A tuple of (state_t, action_t, state_t_plus_1, reward).
        """
        return numpy.random.choice(self._history)
