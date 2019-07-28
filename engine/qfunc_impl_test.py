"""Unit tests for qfunc_impl.py."""
import unittest

import numpy

from deep_learning.engine import q_base
from deep_learning.engine import qfunc_impl
from qpylib import numpy_util


class RandomValueQFunctionTest(unittest.TestCase):

  def test_getValues(self):
    qfunc = qfunc_impl.RandomValueQFunction(action_space_size=3)

    self.assertEqual(
      (2, 3), qfunc.GetValues(numpy.array([[1, 2, 3], [4, 5, 6]])).shape)

    self.assertEqual(
      (2, 3), qfunc.GetValues(numpy.array([[1, 2], [4, 5]])).shape)


class MemoizationQFunctionTest(unittest.TestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.MemoizationQFunction(action_space_size=2)

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
    tmp_file = '/tmp/MemoizationQFunctionTest_savedata.tmp'
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = qfunc_impl.MemoizationQFunction(action_space_size=2)
    qfunc.Load(tmp_file)

    self.assertCountEqual(qfunc._storage.keys(), self.qfunc._storage.keys())
    for k in qfunc._storage.keys():
      numpy_util.TestUtil.AssertArrayEqual(
        qfunc._storage[k], self.qfunc._storage[k])


class DQNTest(unittest.TestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.DQN(
      model=qfunc_impl.CreateModel(
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
    tmp_file = '/tmp/DQNTest_savedata.tmp'
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = qfunc_impl.DQN(
      model=qfunc_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(6, 4),
      ))
    qfunc.Load(tmp_file)

    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._model, self.qfunc._model)


class DQN_TargetNetwork_Test(unittest.TestCase):

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.DQN_TargetNetwork(
      model=qfunc_impl.CreateModel(
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

  def setUp(self) -> None:
    # State space size is 3; Action space size is 2.
    self.qfunc = qfunc_impl.DDQN(
      model_pair=(
        qfunc_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        ),
        qfunc_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        )),
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
    self.qfunc.UpdateValues([q_base.Transition(
      s=numpy.array([[1, 2, 3]]),
      a=numpy.array([[1, 0]]),
      r=1.0,
      sp=numpy.array([[4, 5, 6]])
    )])

    self.assertEqual(q1, self.qfunc._q2)
    self.assertEqual(q2, self.qfunc._q1)

  def test_convergence(self):
    states, actions, target_action_values = None, None, None
    for _ in range(100):
      states, actions, target_action_values = self.qfunc.UpdateValues(
        [q_base.Transition(
          s=numpy.array([[1, 2, 3]]),
          a=numpy.array([[1, 0]]),
          r=1.0,
          sp=numpy.array([[4, 5, 6]])
        )])
    diff1 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(
        self.qfunc.GetActionValues(states), actions) - target_action_values))

    for _ in range(100):
      states, actions, target_action_values = self.qfunc.UpdateValues(
        [q_base.Transition(
          s=numpy.array([[1, 2, 3]]),
          a=numpy.array([[1, 0]]),
          r=1.0,
          sp=numpy.array([[4, 5, 6]])
        )])
    diff2 = numpy.sum(numpy.abs(
      self.qfunc.GetActionValues(
        self.qfunc.GetActionValues(states), actions) - target_action_values))

    self.assertLess(diff2, diff1)

  def test_saveLoad(self):
    tmp_file = '/tmp/DDQNTest_savedata.tmp'
    self.qfunc._SetValues(self.states, self.values)
    self.qfunc.Save(tmp_file)
    qfunc = qfunc_impl.DDQN(
      model_pair=(
        qfunc_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        ),
        qfunc_impl.CreateModel(
          state_shape=(3,),
          action_space_size=2,
          hidden_layer_sizes=(3,),
        )),
    )
    qfunc.Load(tmp_file)

    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._q1, self.qfunc._model)
    numpy_util.TestUtil.AssertModelWeightsEqual(qfunc._q2, self.qfunc._model)
