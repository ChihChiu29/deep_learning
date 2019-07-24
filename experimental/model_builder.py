"""Implements some other types of models."""
from qpylib import t


def CreateSingleHeadModel(
    state_shape: t.Tuple[int],
    action_space_size: int,
    hidden_layer_sizes: t.Iterable[int],
    activation: t.Text,
):
  """Creates a single model with RMSProp optimizer.

  Following reference:
    https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/

  Args:
    state_shape: the shape of the state ndarray.
    action_space_size: the size of the action space.
    hidden_layer_sizes: a list of number of nodes in the hidden layers,
      staring with the input layer.
    activation: the activation, for example "relu".
    rmsprop_learning_rate: the learning rate used by the RMSprop optimizer.
  """
  hidden_layer_sizes = tuple(hidden_layer_sizes)
  model = models.Sequential()
  if len(state_shape) > 1:
    model.add(layers.Flatten(input_shape=state_shape))
    for num_nodes in hidden_layer_sizes:
      model.add(layers.Dense(units=num_nodes, activation=activation))
  else:
    model.add(
      layers.Dense(
        units=hidden_layer_sizes[0],
        activation=activation,
        input_dim=state_shape[0]))
    for num_nodes in hidden_layer_sizes[1:]:
      model.add(layers.Dense(units=num_nodes, activation=activation))
  model.add(layers.Dense(
    units=action_space_size, activation='linear'))

  opt = optimizers.RMSprop(lr=rmsprop_learning_rate)
  model.compile(loss='mse', optimizer=opt)

  return model
