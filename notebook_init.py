"""Init script."""

import gym
import keras
import numpy

from keras import layers
from keras.layers import Activation, Dense, InputLayer
from keras import models
from keras.models import Model, Sequential
from keras import optimizers

from qpylib import logging, numpy_util
from deep_learning.engine import q_base, environment_impl, policy_impl, qfunc_impl, runner_impl
from deep_learning.example import circular_world_env

from deep_learning.experimental import guided_environments

A = numpy.array
T = q_base.Transition

# Add this to the first cell of your notebook:
# ReloadProject('deep_learning')
