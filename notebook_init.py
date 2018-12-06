"""Init script."""

from typing import Dict, List, Tuple

import gym
import keras

from keras.layers import Activation, Dense, InputLayer
from keras.models import Model, Sequential

from lib import q_learning
from lib import q_learning_impl


# Add this to the first cell of your notebook:
# ReloadProject('deep_learning')