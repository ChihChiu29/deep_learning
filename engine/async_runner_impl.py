"""Implements asynchronous runners."""
import threading

from deep_learning.engine import base
from deep_learning.engine.base import States
from deep_learning.engine.base import Values
from qpylib import t


class AsyncBrain(base.Brain):
  """A thread safe and batch training wrapper for another brain."""

  def __init__(
      self,
      brain: base.Brain,
      batch_size: int = 64,
  ):
    """Ctor.

    Args:
      brain: the brain to wrap around.
      batch_size: the batch size. Training only happens when this number of data
        is collected.
    """
    self._brain = brain
    self._batch_size = batch_size

    self._lock = threading.Lock()
    self._batch_buffer = []  # type: t.List[base.Transition]

  # @Override
  def GetValues(self, states: States) -> Values:
    return self._brain.GetValues(states)

  # @Override
  def UpdateFromTransitions(
      self,
      transitions: t.Iterable[base.Transition],
  ) -> None:
    with self._lock:
      self._batch_buffer.extend(transitions)
      if len(self._batch_buffer) > self._batch_size:
        self._brain.UpdateFromTransitions(self._batch_buffer)
        self._batch_buffer = []

  # @Override
  def Save(self, filepath: t.Text) -> None:
    self._brain.Save(filepath)

  # @Override
  def Load(self, filepath: t.Text) -> None:
    with self._lock:
      self._brain.Load(filepath)


class AsyncEnvRunner:
  """Wrapper for a runner running an environment."""

  def __init__(
      self,
      env: base.Environment,
      runner: base.Runner,
  ):
    """Ctor.

    Args:
      env: the environment to run.
      runner: the runner to wrap.
    """
    super().__init__()
    self._env = env
    self._runner = runner

  @property
  def runner(self):
    return self._runner

  def Run(
      self,
      brain: AsyncBrain,
      policy: base.Policy,
      num_of_episodes: int,
  ):
    """Runs the environment using this given runner."""
    self._runner.Run(self._env, brain, policy, num_of_episodes=num_of_episodes)


class AsyncRunnerExtension(base.RunnerExtension):
  """Makes a runner extension thread safe."""

  def __init__(self, runner_ext: base.RunnerExtension):
    self._ext = runner_ext
    self._lock = threading.Lock()

  def OnEpisodeFinishedCallback(
      self,
      env: base.Environment,
      brain: base.Brain,
      episode_idx: int,
      num_of_episodes: int,
      episode_reward: float,
      steps: int):
    with self._lock:
      self._ext.OnEpisodeFinishedCallback(
        env=env,
        brain=brain,
        episode_idx=episode_idx,
        num_of_episodes=num_of_episodes,
        episode_reward=episode_reward,
        steps=steps)

  def OnCompletionCallback(self):
    # Mute this since this can be called from a async runner.
    # OnAllThreadsCompletion should be called explicitly instead.
    pass

  def OnAllThreadsCompletion(self):
    self._ext.OnCompletionCallback()


class ParallelRunner:
  """Helps to run multiple runners in parallel."""

  def __init__(
      self,
      async_runners: t.Iterable[AsyncEnvRunner],
  ):
    """Ctor.

    Args:
      async_runners: the environments and corresponding runners.
    """
    super().__init__()
    self._env_runners = async_runners

    self._exts = []  # type: t.List[AsyncRunnerExtension]

  def AddCallback(self, ext: AsyncRunnerExtension):
    """Adds a callback which extends Runner's ability."""
    self._exts.append(ext)
    for env_runner in self._env_runners:
      env_runner.runner.AddCallback(ext)

  def ClearCallbacks(self):
    """Removes all registered callbacks."""
    self._exts = []
    for env_runner in self._env_runners:
      env_runner.runner.ClearCallbacks()

  def Run(
      self,
      brain: AsyncBrain,
      policy: base.Policy,
      num_of_episodes: int,
  ):
    """Runs an agent for some episodes.

    Args:
      brain: the async-ready brain.
      policy: make the decision.
      num_of_episodes: number of episodes to run for each runner.
    """
    threads = []
    for env_runner in self._env_runners:
      thread = threading.Thread(
        target=env_runner.Run, args=(brain, policy, num_of_episodes))
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    for ext in self._exts:
      ext.OnAllThreadsCompletion()


class _ThreadSafeHistory:
  """A thread save experience and history recorder."""

  def __init__(
      self,
      capacity: int,
      on_full_callback: t.Callable[[t.List[base.Transition]], None],
      on_episode_finish_callback: t.Callable[[int, base.Reward, int], None],
  ):
    """Ctor.

    Args:
      capacity: the experience capacity.
      on_full_callback: when experience is full, call this call back function.
        Experience will be emptied afterwards.
      on_episode_finish_callback: when one episode finishes, this function
        is called with:
          episode_idx: the global episode index (not the one from individual
            runner).
          episode_reward: the reward for this episode.
          episode_steps: the steps run for this episode.
    """
    self._capacity = capacity
    self._on_full_callback = on_full_callback
    self._on_episode_finish_callback = on_episode_finish_callback

    self._lock = threading.Lock()
    self._experience = []  # type: t.List[base.Transition]
    self._episode_idx = 0

  def AddTransition(
      self,
      transition: base.Transition,
  ) -> None:
    with self._lock:
      self._experience.append(transition)
      if len(self._experience) == self._capacity:
        self._on_full_callback(self._experience)
        self._experience = []

  def AddEpisodeInfo(self, reward: base.Reward, steps: int):
    with self._lock:
      self._episode_idx += 1
      self._on_episode_finish_callback(self._episode_idx, reward, steps)


class _AsyncEnvRunner(threading.Thread):
  """Manages running an env."""
  # Global signal for all runners.
  stop_signal = False

  def __init__(
      self,
      env: base.Environment,
      shared_brain: base.Brain,
      shared_policy: base.Policy,
      shared_experience: _ThreadSafeHistory,
      num_of_episodes: int,
  ):
    """Ctor.

    Args:
      env: the environment to run, should use different instances for
        different runners.
      shared_brain: the brain, should be shared with other async runner.
      shared_policy: the policy that makes decision.
      shared_experience: where to dump experience, should be shared.
      num_of_episodes: runs for this number of episodes.
    """
    threading.Thread.__init__(self)
    self._env = env
    self._brain = shared_brain
    self._policy = shared_policy
    self._experience = shared_experience
    self._num_of_episodes = num_of_episodes

  # @Override
  def run(self):
    state = self._env.Reset()
    episode_idx = 0  # type: int
    episode_reward = 0.0  # type: base.Reward
    steps = 0
    while not self.stop_signal:
      tran = self._env.TakeAction(self._policy.Decide(
        env=self._env,
        brain=self._brain,
        state=state,
        episode_idx=episode_idx,
        num_of_episodes=self._num_of_episodes,
      ))
      self._experience.AddTransition(tran)
      episode_reward += tran.r
      steps += 1
      state = tran.sp
      if tran.sp is None:
        self._experience.AddEpisodeInfo(episode_reward, steps)
        episode_reward = 0.0
        steps = 0
        episode_idx += 1
        if episode_idx == self._num_of_episodes:
          self.Stop()
        state = self._env.Reset()

  def Stop(self):
    self.stop_signal = True


class MultiEnvsParallelBatchedRunner:
  """A runner that runs multiple environments in parallel."""

  def __init__(self, batch_size: int):
    """Ctor.

    Args:
      batch_size: for how many transitions will brain be updated.
    """
    self._callbacks = []
    self._batch_size = batch_size

  def _protected_ProcessTransition(
      self,
      brain: base.Brain,
      transition: base.Transition,
      step_idx: int,
  ) -> None:
    """Processes a new transition; e.g. to train the QFunction."""
    pass

  def AddCallback(self, ext: base.RunnerExtension):
    """Adds a callback which extends Runner's ability."""
    self._callbacks.append(ext)

  def ClearCallbacks(self):
    """Removes all registered callbacks."""
    self._callbacks = []

  def Run(
      self,
      envs: t.Iterable[base.Environment],
      brain: base.Brain,
      policy: base.Policy,
      num_of_episodes: int,
  ):
    """Runs an agent for some episodes.

    For each episode, the environment is reset first, then run until it's
    done. Between episodes, Report function is called to give user feedback.
    """
    envs = list(envs)
    total_num_of_episodes = num_of_episodes * len(envs)

    def ExtCaller(
        episode_idx: int,
        episode_reward: base.Reward,
        episode_steps: int) -> None:
      for reporter in self._callbacks:
        reporter.OnEpisodeFinishedCallback(
          env=None,
          brain=brain,
          episode_idx=episode_idx,
          num_of_episodes=total_num_of_episodes,
          episode_reward=episode_reward,
          steps=episode_steps,
        )

    experience = _ThreadSafeHistory(
      capacity=self._batch_size,
      on_full_callback=brain.UpdateFromTransitions,
      on_episode_finish_callback=ExtCaller,
    )

    runners = []
    for env in envs:
      runner = _AsyncEnvRunner(
        env=env,
        shared_brain=brain,
        shared_policy=policy,
        shared_experience=experience,
        num_of_episodes=num_of_episodes,
      )
      runner.start()
      runners.append(runner)

    for runner in runners:
      runner.join()

    # All runs finished.
    for reporter in self._callbacks:
      reporter.OnCompletionCallback()
