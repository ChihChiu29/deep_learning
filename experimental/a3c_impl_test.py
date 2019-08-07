import tempfile
import unittest

import numpy

from deep_learning.engine import base
from deep_learning.experimental import a3c_impl
from qpylib import logging
from qpylib import numpy_util
from qpylib import running_environment

running_environment.ForceCpuForTheRun()


class A3CTest(unittest.TestCase):
  # A3C is essentially a singleton (since it sets Keras backend).
  _multiprocess_can_split_ = False

  def test_convergence(self):
    a3c = a3c_impl.A3C(
      model=a3c_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(3,),
      ),
      # optimizer=a3c_impl.CreateDefaultOptimizer(learning_rate=0.05),
    )
    s = numpy.array([[1, 2, 3]])
    a1 = numpy.array([[1, 0]])
    a2 = numpy.array([[0, 1]])

    for _ in range(10):
      # Needs to train for both actions as one step, otherwise it shows some
      # "staggering" effect.
      a3c.UpdateFromTransitions([
        base.Transition(s=s, a=a1, r=1.0, sp=None),
      ])
      a3c.UpdateFromTransitions([
        base.Transition(s=s, a=a2, r=-1.0, sp=s),
      ])
      logging.printf('%s', a3c.GetValues(s))
    old_value_a1 = a3c.GetActionValues(a3c.GetValues(s), a1)
    # Trains for one step, for both actions.
    a3c.UpdateFromTransitions([
      base.Transition(s=s, a=a1, r=1.0, sp=None),
    ])
    a3c.UpdateFromTransitions([
      base.Transition(s=s, a=a2, r=-1.0, sp=s),
    ])
    self.assertGreaterEqual(
      a3c.GetActionValues(a3c.GetValues(s), a1), old_value_a1)

  def test_saveLoad(self):
    a3c = a3c_impl.A3C(
      model=a3c_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(3,),
      ),
    )
    tmp_file = tempfile.NamedTemporaryFile().name
    s = numpy.array([[1, 2, 3]])
    for _ in range(10):
      a3c.UpdateFromTransitions([
        base.Transition(
          s=s,
          a=numpy.array([[1, 0]]),
          r=1.0,
          sp=numpy.array([[4, 5, 6]])),
      ])
    a3c.Save(tmp_file)
    saved_values = a3c.GetValues(s)

    a3c = a3c_impl.A3C(
      model=a3c_impl.CreateModel(
        state_shape=(3,),
        action_space_size=2,
        hidden_layer_sizes=(3,),
      ),
    )
    a3c.Load(tmp_file)

    numpy_util.TestUtil.AssertArrayEqual(saved_values, a3c.GetValues(s))
