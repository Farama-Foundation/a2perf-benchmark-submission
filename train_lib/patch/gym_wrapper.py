import collections
from typing import Dict
from typing import Optional
from typing import Text

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tf_agents import specs


def spec_from_gym_space(
    space: gym.Space,
    dtype_map: Optional[Dict[gym.Space, np.dtype]] = None,
    simplify_box_bounds: bool = True,
    name: Optional[Text] = None,
) -> specs.BoundedArraySpec:
  """Converts gym spaces into array specs.

  Gym does not properly define dtypes for spaces. By default all spaces set
  their type to float64 even though observations do not always return this type.
  See:
  https://github.com/openai/gym/issues/527

  To handle this we allow a dtype_map for setting default types for mapping
  spaces to specs.

  TODO(oars): Support using different dtypes for different parts of the
  observations. Not sure that we have a need for this yet.

  Args:
    space: gym.Space to turn into a spec.
    dtype_map: A dict from spaces to dtypes to use as the default dtype.
    simplify_box_bounds: Whether to replace bounds of Box space that are arrays
      with identical values with one number and rely on broadcasting.
    name: Name of the spec.

  Returns:
    A BoundedArraySpec nest mirroring the given space structure.
  Raises:
    ValueError: If there is an unknown space type.
  """
  if dtype_map is None:
    dtype_map = {}

  # We try to simplify redundant arrays to make logging and debugging less
  # verbose and easier to read since the printed spec bounds may be large.
  def try_simplify_array_to_value(np_array):
    """If given numpy array has all the same values, returns that value."""
    first_value = np_array.item(0)
    if np.all(np_array == first_value):
      return np.array(first_value, dtype=np_array.dtype)
    else:
      return np_array

  def nested_spec(spec, child_name):
    """Returns the nested spec with a unique name."""
    nested_name = name + '/' + child_name if name else child_name
    return spec_from_gym_space(
        spec, dtype_map, simplify_box_bounds, nested_name
    )

  if isinstance(space, gym.spaces.Discrete):
    # Discrete spaces span the set {0, 1, ... , n-1} while Bounded Array specs
    # are inclusive on their bounds.
    maximum = space.n - 1
    # TODO(oars): change to use dtype in space once Gym is updated.
    dtype = dtype_map.get(gym.spaces.Discrete, np.int64)
    return specs.BoundedArraySpec(
        shape=(), dtype=dtype, minimum=0, maximum=maximum, name=name
    )
  elif isinstance(space, gym.spaces.MultiDiscrete):
    dtype = dtype_map.get(gym.spaces.MultiDiscrete, np.int32)
    maximum = try_simplify_array_to_value(
        np.asarray(space.nvec - 1, dtype=dtype)
    )
    return specs.BoundedArraySpec(
        shape=space.shape, dtype=dtype, minimum=0, maximum=maximum, name=name
    )
  elif isinstance(space, gym.spaces.MultiBinary):
    dtype = dtype_map.get(gym.spaces.MultiBinary, np.int32)
    # Can remove this once we update gym.
    if isinstance(space.n, int):
      shape = (space.n,)
    else:
      shape = tuple(space.n)
    return specs.BoundedArraySpec(
        shape=shape, dtype=dtype, minimum=0, maximum=1, name=name
    )
  elif isinstance(space, gym.spaces.Box):
    if hasattr(space, 'dtype') and gym.spaces.Box not in dtype_map:
      dtype = space.dtype
    else:
      dtype = dtype_map.get(gym.spaces.Box, np.float32)
    if dtype == tf.string:
      return specs.ArraySpec(shape=space.shape, dtype=dtype, name=name)
    minimum = np.asarray(space.low, dtype=dtype)
    maximum = np.asarray(space.high, dtype=dtype)
    if simplify_box_bounds:
      simple_minimum = try_simplify_array_to_value(minimum)
      simple_maximum = try_simplify_array_to_value(maximum)
      # Can only simplify if both bounds are simplified. Otherwise
      # broadcasting doesn't work from non-simplified to simplified.
      if simple_minimum.shape == simple_maximum.shape:
        minimum = simple_minimum
        maximum = simple_maximum
    return specs.BoundedArraySpec(
        shape=space.shape,
        dtype=dtype,
        minimum=minimum,
        maximum=maximum,
        name=name,
    )
  elif isinstance(space, gym.spaces.Tuple):
    return tuple(
        [nested_spec(s, 'tuple_%d' % i) for i, s in enumerate(space.spaces)]
    )
  elif isinstance(space, gym.spaces.Dict):
    return collections.OrderedDict(
        [(key, nested_spec(s, key)) for key, s in space.spaces.items()]
    )
  else:
    raise ValueError(
        'The gym space {} is currently not supported.'.format(space)
    )


from tf_agents.environments import gym_wrapper

gym_wrapper.spec_from_gym_space = spec_from_gym_space
