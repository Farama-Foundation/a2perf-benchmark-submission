from typing import Optional
from typing import Text

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.td3 import td3_agent
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from .networks import _create_actor_distribution_net
from .networks import _create_actor_net
from .networks import _create_critic_net
from .networks import _create_q_net
from .networks import _create_value_net


def _create_ddpg_agent(
    env_name: Text,
    train_step: tf.Variable,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    learning_rate: float,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    gradient_clipping: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> tf_agent.TFAgent:
  critic_net = _create_critic_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      env_name=env_name,
  )
  actor_net = _create_actor_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      env_name=env_name,
  )

  return ddpg_agent.DdpgAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      gradient_clipping=gradient_clipping,
      train_step_counter=train_step,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
  )


def _create_td3_agent(
    env_name: Text,
    train_step: tf.Variable,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    learning_rate: float,
    exploration_noise_std: float,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    gradient_clipping: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> tf_agent.TFAgent:
  critic_net = _create_critic_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      env_name=env_name,
  )
  actor_net = _create_actor_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      env_name=env_name,
  )
  return td3_agent.Td3Agent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      gradient_clipping=gradient_clipping,
      train_step_counter=train_step,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      exploration_noise_std=exploration_noise_std,
  )


def _create_ppo_agent(
    env_name: Text,
    train_step: tf.Variable,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    learning_rate: float,
    entropy_regularization: float,
    gradient_clipping: Optional[float],
    use_gae: bool,
    debug_summaries: bool,
    summarize_grads_and_vars: bool,
    # webnav kwargs
    max_vocab_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    profile_value_dropout: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> tf_agent.TFAgent:
  """Creates a PPO agent."""
  actor_net = _create_actor_distribution_net(
      env_name=env_name,
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      max_vocab_size=max_vocab_size,
      latent_dim=latent_dim,
      profile_value_dropout=profile_value_dropout,
      embedding_dim=embedding_dim,
  )

  value_net = _create_value_net(
      env_name=env_name,
      observation_tensor_spec=observation_tensor_spec,
      seed=seed,
      max_vocab_size=max_vocab_size,
      latent_dim=latent_dim,
      profile_value_dropout=profile_value_dropout,
      embedding_dim=embedding_dim,
  )

  optimizer = tf.keras.optimizers.Adam(
      learning_rate=learning_rate, epsilon=1e-5
  )

  return ppo_clip_agent.PPOClipAgent(
      action_spec=action_tensor_spec,
      actor_net=actor_net,
      compute_value_and_advantage_in_train=False,
      debug_summaries=debug_summaries,
      entropy_regularization=entropy_regularization,
      gradient_clipping=gradient_clipping,
      greedy_eval=False,
      importance_ratio_clipping=0.2,
      normalize_observations=True,
      normalize_rewards=True,
      num_epochs=1,  # Legacy argument, should always be 1
      optimizer=optimizer,
      summarize_grads_and_vars=summarize_grads_and_vars,
      time_step_spec=time_step_tensor_spec,
      train_step_counter=train_step,
      update_normalizers_in_train=False,
      use_gae=use_gae,
      use_td_lambda_return=True,
      value_net=value_net,
  )


def _create_ddqn_agent(
    env_name: Text,
    train_step: tf.Variable,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    epsilon_greedy: float,
    learning_rate: float,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    gradient_clipping: Optional[float] = None,
    # webnav kwargs
    max_vocab_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    profile_value_dropout: Optional[float] = None,
    embedding_dim: Optional[int] = None,
    seed: Optional[int] = None,
) -> tf_agent.TFAgent:
  """Creates an agent."""
  q_net = _create_q_net(
      env_name=env_name,
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      max_vocab_size=max_vocab_size,
      latent_dim=latent_dim,
      profile_value_dropout=profile_value_dropout,
      embedding_dim=embedding_dim,
  )

  return dqn_agent.DdqnAgent(
      time_step_spec=time_step_tensor_spec,
      action_spec=action_tensor_spec,
      q_network=q_net,
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      td_errors_loss_fn=tf.math.squared_difference,
      train_step_counter=train_step,
      epsilon_greedy=epsilon_greedy,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      gradient_clipping=gradient_clipping,
  )


def _create_sac_agent(
    env_name: Text,
    train_step: tf.Variable,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    learning_rate: float,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    gradient_clipping: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> tf_agent.TFAgent:
  """Creates an agent."""

  critic_net = _create_critic_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      env_name=env_name,
  )
  actor_net = _create_actor_distribution_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      env_name=env_name,
  )
  return sac_agent.SacAgent(
      time_step_tensor_spec,
      action_tensor_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      alpha_optimizer=tf.keras.optimizers.Adam(
          learning_rate=learning_rate, epsilon=1e-5
      ),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      gradient_clipping=gradient_clipping,
      train_step_counter=train_step,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
  )
