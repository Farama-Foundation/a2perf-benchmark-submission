from typing import Optional
from typing import Text

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import q_network
from tf_agents.networks import value_network
from tf_agents.typing import types

from a2perf.domains.web_navigation.gwob.CoDE import networks as web_networks
from .circuit_training import static_feature_cache
from .models import GrlModel
from .models import GrlPolicyModel
from .models import create_circuit_training_dqn_models_fn


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
) -> q_network.QNetwork:
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
    return web_networks.WebLSTMQNetwork(
        vocab_size=max_vocab_size,
        latent_dim=latent_dim,
        profile_value_dropout=profile_value_dropout,
        embedding_dim=embedding_dim,
    )
  elif env_name == 'CircuitTraining-v0':
    static_features = kwargs.get('static_features', None)
    observation_tensor_spec = kwargs.get('observation_tensor_spec')
    action_tensor_spec = kwargs.get('action_tensor_spec')

    cache = static_feature_cache.StaticFeatureCache()
    cache.add_static_feature(static_features)
    return create_circuit_training_dqn_models_fn(
        rl_architecture='generalization',
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        static_features=cache.get_all_static_features(),
        use_model_tpu=False,
        seed=seed,
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
  elif env_name == 'CircuitTraining-v0':
    # Create ppo models but only use actor
    static_features = kwargs.get('static_features', None)
    cache = static_feature_cache.StaticFeatureCache()
    cache.add_static_feature(static_features)
    grl_shared_net = GrlModel(
        observation_tensor_spec,
        action_tensor_spec,
        all_static_features=cache.get_all_static_features(),
        use_model_tpu=False,
        is_augmented=False,
        seed=seed,
    )
    grl_actor_net = GrlPolicyModel(
        grl_shared_net, observation_tensor_spec, action_tensor_spec
    )
    return grl_actor_net

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
