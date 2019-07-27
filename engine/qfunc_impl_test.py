"""Unit tests for qfunc_impl.py."""
import unittest

import numpy

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

    weights1 = qfunc._model.get_weights()
    weights2 = self.qfunc._model.get_weights()
    self.assertEqual(len(weights1), len(weights2))
    for idx in range(len(weights1)):
      numpy_util.TestUtil.AssertArrayEqual(weights1[idx], weights2[idx])
