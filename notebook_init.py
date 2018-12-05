"""Init script."""

from typing import Dict, List, Tuple

import gym
import keras

from keras.layers import Activation
from keras.layers import Dense
from keras.models import Sequential

from lib import q_learning
from lib import q_learning_impl


# Add this to the first cell of your notebook:
# ReloadProject('deep_learning')