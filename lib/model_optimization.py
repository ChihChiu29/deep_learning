"""Helps to optimize a Keras model.

A Keras model can be evaluated to a score, and functions in this module help
to optimize weights in the model to achieve higher score.
"""

from typing import Iterable

import numpy

class WeightLayer:
    """Helps to manager weights in a layer."""
    
    def __init__(self, keras_layer):
        """Constructor.
        
        Args:
            keras_layer: a Keras layer. It needs to have get/set_weights
                functioins.
        """
        self._layer = keras_layer
        
        # Total number of weights.
        self._size = None  # int
        # Shapes per variable.
        self._dimensions = []  # [[shape, size], ...]
        
        self._SetDimensions()
        
    @property
    def size(self):
        return self._size
        
    def _SetDimensions(self):
        """Sets various dimensions from variables in the layer."""
        self._size = 0
        for variable_ndarray in self._layer.get_weights():
            size = variable_ndarray.size
            self._dimensions.append((variable_ndarray.shape, size))
            self._size += size
            
    def GetWeights(self) -> numpy.ndarray:
        """Gets all weights as a 1-d ndarray."""
        return numpy.concatenate(list(
            variable_ndarray.flatten() for variable_ndarray in
            self._layer.get_weights()))
            
    def SetWeights(self, weight_source: numpy.ndarray) -> None:
        """Sets weights.
        
        Args:
            weight_array: the source of the weights. It needs to have more
                elements than required by this layer.
        """
        cursor = 0
        new_weights = []
        for shape, size in self._dimensions:
            new_weights.append(
                weight_source[cursor:cursor+size].reshape(shape))
            cursor += size
        self._layer.set_weights(new_weights)


class WeigthFunction:
    """A weight function treats a Keras model as a function of weights."""
    
    def __init__(self, keras_model):
        """Constructor.
        
        Args:
            keras_model: a Keras model. It needs to have `layers` attribute.
        """
        self._model = keras_model
        
        # Total number of weights in the model.
        self._total_weight_size = None