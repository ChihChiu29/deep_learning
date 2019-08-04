"""Init script."""

import gym
import keras
import numpy
from keras import layers
from keras import models
from keras import optimizers

from deep_learning.engine import brain_impl
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import q_base
from deep_learning.engine import runner_extension_impl
from deep_learning.engine import runner_impl
from deep_learning.examples import circular_world_env
from deep_learning.examples import shortcut
from deep_learning.experimental import guided_environments
from deep_learning.experimental import model_builder
from qpylib import logging
from qpylib import numpy_util
from qpylib import running_environment

__will_use = (
  gym,
  keras,
  numpy,

  layers,
  models,
  optimizers,

  logging,
  numpy_util,
  running_environment,

  q_base,
  environment_impl,
  policy_impl,
  brain_impl,
  runner_extension_impl,
  runner_impl,
  shortcut,

  circular_world_env,
  model_builder,
  guided_environments,
)

# Add this to the first cell of your notebook:
# ReloadProject('deep_learning')
