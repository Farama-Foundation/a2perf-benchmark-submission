import functools
import sys
from typing import Optional
from typing import Text

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.networks import value_network
from tf_agents.typing import types

from a2perf.domains.web_navigation.gwob.CoDE import networks as web_networks


def _create_critic_net(
    env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
) -> critic_network.CriticNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return critic_network.CriticNetwork(
        (observation_tensor_spec, action_tensor_spec),
        joint_fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    raise ValueError(
        'SAC cannot be used for WebNavigation due to discrete action space'
    )
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_value_net(
    env_name: Text, observation_tensor_spec: types.NestedTensorSpec, **kwargs
) -> value_network.ValueNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return value_network.ValueNetwork(
        observation_tensor_spec,
        fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size')
    latent_dim = kwargs.get('latent_dim')
    profile_value_dropout = kwargs.get('profile_value_dropout')
    embedding_dim = kwargs.get('embedding_dim')

    return web_networks.WebLSTMValueNetwork(
        input_tensor_spec=observation_tensor_spec,
        lstm_kwargs=dict(
            vocab_size=max_vocab_size,
            latent_dim=latent_dim,
            profile_value_dropout=profile_value_dropout,
            embedding_dim=embedding_dim,
        ),
    )
  elif env_name == 'CircuitTraining-v0':
    seed = kwargs.get('seed')
    seed_stream = tfp.util.SeedStream(
        seed=seed, salt='value_net_weight_init_seed'
    )
    init = tf.keras.initializers.GlorotUniform(seed=seed_stream() % sys.maxsize)

    dense = functools.partial(
        tf.keras.layers.Dense, activation='relu', kernel_initializer=init
    )
    fc_layer_units = [64, 64, 64, 64]

    def value_layer():
      return tf.keras.layers.Dense(
          1, activation=None, kernel_initializer=init, name='value'
      )

    def drop_mask(observation_and_mask):
      return observation_and_mask['graph_embedding']

    def squeeze_value_dim(value):
      # Make value_prediction's shape from [B, T, 1] to [B, T].
      return tf.squeeze(value, -1, name='squeeze_value_net')

    return sequential.Sequential(
        [tf.keras.layers.Lambda(drop_mask)]
        + [tf.keras.layers.Flatten()]
        + [dense(num_units) for num_units in fc_layer_units]
        + [value_layer(), tf.keras.layers.Lambda(squeeze_value_dim)],
        input_spec=observation_tensor_spec,
        name='value_network',
    )
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_q_net(
    env_name: Text, seed: Optional[int] = None, **kwargs
) -> critic_network.CriticNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    raise ValueError(
        'DDQN cannot be used for QuadrupedLocomotion due to continuous action'
        ' space'
    )

  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size')
    latent_dim = kwargs.get('latent_dim')
    profile_value_dropout = kwargs.get('profile_value_dropout')
    embedding_dim = kwargs.get('embedding_dim')
    return networks.WebLSTMQNetwork(
        vocab_size=max_vocab_size,
        latent_dim=latent_dim,
        profile_value_dropout=profile_value_dropout,
        embedding_dim=embedding_dim,
    )
  elif env_name == 'CircuitTraining-v0':
    pass
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_actor_net(
    env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    seed: Optional[int] = None,
    **kwargs,
) -> actor_network.ActorNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return actor_network.ActorNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size')
    latent_dim = kwargs.get('latent_dim')
    profile_value_dropout = kwargs.get('profile_value_dropout')
    embedding_dim = kwargs.get('embedding_dim')

    return networks.WebLSTMActorNetwork(
        input_tensor_spec=observation_tensor_spec,
        output_tensor_spec=action_tensor_spec,
        lstm_kwargs=dict(
            vocab_size=max_vocab_size,
            latent_dim=latent_dim,
            profile_value_dropout=profile_value_dropout,
            embedding_dim=embedding_dim,
        ),
    )
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_actor_distribution_net(
    env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    seed: Optional[int] = None,
    **kwargs,
) -> actor_distribution_network.ActorDistributionNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return actor_distribution_network.ActorDistributionNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size')
    latent_dim = kwargs.get('latent_dim')
    profile_value_dropout = kwargs.get('profile_value_dropout')
    embedding_dim = kwargs.get('embedding_dim')

    return web_networks.WebLSTMActorDistributionNetwork(
        input_tensor_spec=observation_tensor_spec,
        output_tensor_spec=action_tensor_spec,
        lstm_kwargs=dict(
            vocab_size=max_vocab_size,
            latent_dim=latent_dim,
            profile_value_dropout=profile_value_dropout,
            embedding_dim=embedding_dim,
        ),
    )
  elif env_name == 'CircuitTraining-v0':
    seed_stream = tfp.util.SeedStream(
        seed=seed, salt='actor_net_weight_init_seed'
    )
    init = tf.keras.initializers.GlorotUniform(seed=seed_stream() % sys.maxsize)

    dense = functools.partial(
        tf.keras.layers.Dense, activation='relu', kernel_initializer=init
    )
    fc_layer_units = [64, 64, 64, 64]

    def no_op_layer():
      return tf.keras.layers.Lambda(lambda x: x)

    def projection_layer():
      return tf.keras.layers.Dense(
          np.unique(
              action_tensor_spec.maximum - action_tensor_spec.minimum + 1),
          activation=None,
          kernel_initializer=init,
          name='projection_layer',
      )

    def create_dist(features):
      # Apply mask onto the logits such that infeasible actions will not be taken.
      _, logits, mask = features.values()

      if mask.shape.rank < logits.shape.rank:
        mask = tf.expand_dims(mask, -2)

      # Overwrite the logits for invalid (!= 1) actions to a very large negative
      # number. We do not use -inf because it produces NaNs in many tfp
      # functions.
      # Currently keep aligned with Menger. Eventually move to logits.dtype.min.
      almost_neg_inf = tf.ones_like(logits) * (-(2.0 ** 32) + 1)
      logits = tf.where(tf.equal(mask, 1), logits, almost_neg_inf)

      return tfp.distributions.Categorical(
          logits=logits, dtype=action_tensor_spec.dtype
      )

    return sequential.Sequential(
        [
            nest_map.NestMap({
                'augmented_features': no_op_layer(),
                'graph_embedding': tf.keras.Sequential(
                    [tf.keras.layers.Flatten()]
                    + [dense(num_units) for num_units in fc_layer_units]
                    + [projection_layer()]
                ),
                'mask': no_op_layer(),
            })
        ]
        +
        # Create the output distribution from the mean and standard deviation.
        [tf.keras.layers.Lambda(create_dist)],
        input_spec=observation_tensor_spec,
        name='actor_network',
    )
  else:
    raise ValueError(f'No network defined for {env_name}')
