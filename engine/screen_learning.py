"""Implementations specific to learning using screens images.

Ref:
  https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and
  -prioritized-experience-replay/
  https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py
"""
import PIL
import keras
import numpy
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers

from deep_learning.engine import brain_impl
from deep_learning.engine import environment_impl
from deep_learning.engine import q_base
from qpylib import t

IMAGE_STACK = 2
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84


class ScreenGymEnvironment(environment_impl.GymEnvironment):
  """Gym environment for screen based learning.
  
  The main difference from a Gym environment is that instead using the raw
  state returned by a Gym environment, a new image consists of 2 frames is
  used.
  """

  def __init__(self, gym_env):
    super().__init__(gym_env)

    # Initialized in Reset. Contains previous and current image.
    self._current_stacked_img = None  # type: numpy.ndarray

  # @Override
  def GetStateShape(self) -> t.Sequence[int]:
    return IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT

  # @Override
  def Reset(self) -> q_base.State:
    img = self._ProcessStateImage(super().Reset())
    self._current_stacked_img = numpy.array([img, img])
    return self._StackedImg2State(self._current_stacked_img)

  # @Override
  def TakeAction(self, action: q_base.Action) -> q_base.Transition:
    tran = super().TakeAction(action)
    tran.s = self._StackedImg2State(self._current_stacked_img)
    if tran.sp is not None:
      new_stacked_img = numpy.array(
        [self._current_stacked_img[1], self._ProcessStateImage(tran.sp)])
      tran.sp = self._StackedImg2State(new_stacked_img)
      self._current_stacked_img = new_stacked_img
    return tran

  def _StackedImg2State(self, img):
    return img[numpy.newaxis, :]

  # From: https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py
  def _ProcessStateImage(self, img_state: q_base.State):
    """Processes a image state to an image of fixed size."""
    img = img_state[0]
    rgb = numpy.array(
      PIL.Image.fromarray(img).resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT), resample=PIL.Image.BILINEAR))
    # rgb = scipy.misc.imresize(
    #   img, (IMAGE_WIDTH, IMAGE_HEIGHT), interp='bilinear')

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # extract luminance

    o = gray.astype('float32') / 128 - 1  # normalize
    return o


def CreateOriginalConvolutionModel(
    action_space_size: int,
    activation: t.Text = 'relu',
    optimizer: optimizers.Optimizer = None,
) -> keras.Model:
  """Creates a convolution model suitable for screen based learning.

  The model created by this function is very heavy for CPU training.

  This model uses parameters reported in:
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
  """
  if optimizer is None:
    optimizer = brain_impl.CreateDefaultOptimizer()

  model = models.Sequential()
  model.add(layers.Conv2D(
    32, (8, 8),
    strides=(4, 4),
    activation=activation,
    input_shape=(IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT),
    data_format='channels_first'))
  model.add(layers.Conv2D(
    64,
    (4, 4),
    strides=(2, 2),
    activation=activation))
  model.add(layers.Conv2D(
    64,
    (3, 3),
    activation=activation))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=512, activation=activation))
  model.add(layers.Dense(units=action_space_size))

  model.compile(loss='mse', optimizer=optimizer)

  return model


def CreateConvolutionModel(
    action_space_size: int,
    activation: t.Text = 'relu',
    optimizer: optimizers.Optimizer = None,
) -> keras.Model:
  """Creates a convolution model suitable for screen based learning."""
  if optimizer is None:
    optimizer = brain_impl.CreateDefaultOptimizer()

  model = models.Sequential()
  model.add(layers.Conv2D(
    16, (8, 8),
    strides=(4, 4),
    activation=activation,
    input_shape=(IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT),
    data_format='channels_first'))
  model.add(layers.Conv2D(
    16,
    (4, 4),
    strides=(2, 2),
    activation=activation))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=32, activation=activation))
  model.add(layers.Dense(units=action_space_size))

  model.compile(loss='mse', optimizer=optimizer)

  return model
