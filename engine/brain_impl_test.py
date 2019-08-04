"""Unit tests for brain_impl.py."""
import tempfile
import unittest

import numpy

from deep_learning.engine import brain_impl
from deep_learning.engine import q_base
from qpylib import numpy_util


class RandomValueQFunctionTest(unittest.TestCase):

  def test_getValues(self):
    qfunc = brain_impl.RandomBrain(action_space_size=3)

    self.assertEqual(
      (2, 3), qfunc.GetValues(numpy.array([[1, 2, 3], [4, 5, 6]])).shape)

    self.assertEqual(
      (2, 3), qfunc.GetValues(numpy.array([[1, 2], [4, 5]])).shape)


class MemoizationQFunctionTest(unittest.TestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = brain_impl.MemoizationBrain(action_space_size=2)

    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_getSetValues(self):
    self.qfunc._SetValues(self.states, self.values)
    numpy_util.TestUtil.AssertArrayEqual(
      self.values, self.qfunc.GetValues(self.states))

  def test_saveLoad(self):
    tmp_file = tempfile.NamedTemporaryFile().name
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = brain_impl.MemoizationBrain(action_space_size=2)
    qfunc.Load(tmp_file)

    self.assertCountEqual(qfunc._storage.keys(), self.qfunc._storage.keys())
    for k in qfunc._storage.keys():
      numpy_util.TestUtil.AssertArrayEqual(
        qfunc._storage[k], self.qfunc._storage[k])


class DQNTest(unittest.TestCase):
  _multiprocess_can_split_ = True

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = brain_impl.DQN(
      model=brain_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(6, 4),
      ))
    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_getSetValues_convergence(self):
    for _ in range(100):
      self.qfunc._SetValues(self.states, self.values)
    diff1 = numpy.sum(
      numpy.abs(self.values - self.qfunc.GetValues(self.states)))
    for _ in range(100):
      self.qfunc._SetValues(self.states, self.values)
    diff2 = numpy.sum(
      numpy.abs(self.values - self.qfunc.GetValues(self.states)))

    self.assertLess(diff2, diff1)

  def test_saveLoad(self):
    tmp_file = tempfile.NamedTemporaryFile().name
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = brain_impl.DQN(
      model=brain_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(6, 4),
      ))
    qfunc.Load(tmp_file)

    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._model, self.qfunc._model)


class DQN_TargetNetwork_Test(unittest.TestCase):
  _multiprocess_can_split_ = True

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = brain_impl.DQN_TargetNetwork(
      model=brain_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(6, 4),
      ),
      update_target_network_every_num_of_steps=2,
    )
    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_copyWeightsToTargetNetwork(self):
    # Tests that after 1 set values action, target network is different
    # than Q-function model.
    self.qfunc._SetValues(self.states, self.values)
    try:
      numpy_util.TestUtil.AssertModelWeightsEqual(
        self.qfunc._model, self.qfunc._target_network)
      self.fail('model weights should be different after 1 set action')
    except AssertionError:
      pass

    # Tests that after 2 set values actions, target network is the same as
    # the Q-function model.
    self.qfunc._SetValues(self.states, self.values)
    numpy_util.TestUtil.AssertModelWeightsEqual(
      self.qfunc._model, self.qfunc._target_network)


class DDQNTest(unittest.TestCase):
  _multiprocess_can_split_ = True

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = brain_impl.DDQN(
      model_pair=(
        brain_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        ),
        brain_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        )),
      discount_factor=0.9,
    )
    self.states = numpy.array([
      [1, 2, 3],
      [4, 5, 6],
    ])

    self.values = numpy.array([
      [0.5, 0.5],
      [0.3, 0.7],
    ])

  def test_updateValues_swapModels(self):
    q1 = self.qfunc._q1
    q2 = self.qfunc._q2
    self.qfunc.UpdateFromTransitions([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[1, 0]]),
      r=1.0,
      sp=numpy.array([[4, 5, 6]])
    )])

    self.assertEqual(q1, self.qfunc._q2)
    self.assertEqual(q2, self.qfunc._q1)

  def test_convergence(self):
    trans = [q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[1, 0]]),
      r=1.0,
      sp=None,
    )]
    states, actions, target_action_values = None, None, None
    for _ in range(100):
      states, actions, target_action_values = self.qfunc.UpdateFromTransitions(
        trans)

    error1_1 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(self.qfunc.GetValues(states), actions) -
      target_action_values))
    states, actions, target_action_values = self.qfunc.UpdateFromTransitions(
      trans)
    error1_2 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(self.qfunc.GetValues(states), actions) -
      target_action_values))
    # Needs this to swap back q1 and q2.
    states, actions, target_action_values = self.qfunc.UpdateFromTransitions(
      trans)

    # Since an even number of iterations was used in the first loop, an even
    # number must be used here as well to make sure it's the same model that's
    # being compared.
    for _ in range(100):
      states, actions, target_action_values = self.qfunc.UpdateFromTransitions(
        trans)

    error2_1 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(self.qfunc.GetValues(states), actions) -
      target_action_values))
    states, actions, target_action_values = self.qfunc.UpdateFromTransitions(
      trans)
    error2_2 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(self.qfunc.GetValues(states), actions) -
      target_action_values))

    # Only compare errors from the same model.
    self.assertLessEqual(error2_1, error1_1)
    self.assertLessEqual(error2_2, error1_2)

  def test_saveLoad(self):
    tmp_file = tempfile.NamedTemporaryFile().name
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = brain_impl.DDQN(
      model_pair=(
        brain_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        ),
        brain_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        )),
    )
    qfunc.Load(tmp_file)

    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._q1, self.qfunc._model)
    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._q2, self.qfunc._model)
