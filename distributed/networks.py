from typing import Optional
from typing import Text

from a2perf.domains.web_navigation.gwob.CoDE import networks as web_networks
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.typing import types


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
  else:
    raise ValueError(f'No network defined for {env_name}')
