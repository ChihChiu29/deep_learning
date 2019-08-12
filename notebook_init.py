"""Init script."""

import gym
import keras
import numpy
from keras import layers
from keras import models
from keras import optimizers

from deep_learning.engine import a3c_impl
from deep_learning.engine import base
from deep_learning.engine import environment_impl
from deep_learning.engine import policy_impl
from deep_learning.engine import qfunc_impl
from deep_learning.engine import runner_extension_impl
from deep_learning.engine import runner_impl
from deep_learning.examples import circular_world_env
from deep_learning.examples import interval_world_env
from deep_learning.examples import shortcut
from deep_learning.experimental import guided_environments
from deep_learning.experimental import model_builder
from deep_learning.experimental import other_runners
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

  base,
  a3c_impl,
  environment_impl,
  policy_impl,
  qfunc_impl,
  runner_extension_impl,
  runner_impl,

  circular_world_env,
  interval_world_env,
  model_builder,
  guided_environments,
  other_runners,
  shortcut,
)

# Add this to the first cell of your notebook:
# ReloadProject('deep_learning')
