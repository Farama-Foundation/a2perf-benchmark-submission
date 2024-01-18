# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Sample training with distributed collection using a variable container.

See README for launch instructions.
"""
import functools
import os
from typing import Callable
from typing import Optional
from typing import Text

from a2perf.domains import quadruped_locomotion
from a2perf.domains.web_navigation.gwob.CoDE import networks
from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import reverb
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.'
)
_LEARNER_ITERATIONS_PER_CALL = flags.DEFINE_integer(
    'learner_iterations_per_call', None,
    'Number of iterations per learner call.')
_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', 1000, 'Log interval.')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_POLICY_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'policy_checkpoint_interval', 1000, 'Policy checkpoint interval.'
)
_TIMESTEPS_PER_ACTORBATCH = flags.DEFINE_integer(
    'timesteps_per_actorbatch', None, 'Number of timesteps per actorbatch.')

_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_MOTION_FILE_PATH = flags.DEFINE_string(
    'motion_file_path', None, 'Path to the motion file.'
)
_TRAIN_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'train_checkpoint_interval', 1000, 'Train checkpoint interval.'
)
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', None,
                                     'Number of websites to use.')
_DIFFICULTY_LEVEL = flags.DEFINE_integer('difficulty_level', None,
                                         'Difficulty of the task.')

_USE_TPU = flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or not.')
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')
_MAX_TRAIN_STEP = flags.DEFINE_integer('max_train_steps', None,
                                       'Number of iterations.')
_GRADIENT_CLIPPING = flags.DEFINE_float('gradient_clipping', None,
                                        'Gradient clipping.')
_DEBUG_SUMMARIES = flags.DEFINE_bool('debug_summaries', False,
                                     'Whether to use debug summaries.')
_SUMMARIZE_GRADS_AND_VARS = flags.DEFINE_bool('summarize_grads_and_vars', False,
                                              'Whether to summarize grads and vars.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', None, 'Learning rate.')
FLAGS = flags.FLAGS


class PrefixedLogFormatter(logging.PythonFormatter):
  def format(self, record):
    original = super(PrefixedLogFormatter, self).format(record)
    return f'Train: {original}'


def _create_actor_net(env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    seed: Optional[int] = None, **kwargs
) -> actor_distribution_network.ActorDistributionNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return actor_distribution_network.ActorDistributionNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=(512, 256),
        seed=seed
    )
  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size', None)
    latent_dim = kwargs.get('latent_dim', None)
    profile_value_dropout = kwargs.get('profile_value_dropout', None)
    embedding_dim = kwargs.get('embedding_dim', None)
    if not all(
        [max_vocab_size, latent_dim, profile_value_dropout, embedding_dim]):
      raise ValueError('Missing arguments for WebLSTMActorDistributionNetwork')
    return networks.WebLSTMActorDistributionNetwork(
        input_tensor_spec=observation_tensor_spec,
        output_tensor_spec=action_tensor_spec,
        lstm_kwargs=dict(
            vocab_size=max_vocab_size,
            latent_dim=latent_dim,
            profile_value_dropout=profile_value_dropout,
            embedding_dim=embedding_dim,
        ))
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_critic_net(env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
    seed: Optional[int] = None, **kwargs
) -> critic_network.CriticNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return critic_network.CriticNetwork(
        (observation_tensor_spec, action_tensor_spec),
        joint_fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    raise ValueError(
        'SAC cannot be used for WebNavigation due to discrete action space')
  else:
    raise ValueError(f'No network defined for {env_name}')


def _create_agent(
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

) -> tf_agent.TFAgent:
  """Creates an agent."""

  critic_net = _create_critic_net(
      observation_tensor_spec=observation_tensor_spec,
      action_tensor_spec=action_tensor_spec,
      seed=seed,
      env_name=env_name
  )
  actor_net = _create_actor_net(
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
      actor_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                               epsilon=1e-5),
      critic_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                epsilon=1e-5),
      alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                               epsilon=1e-5),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.math.squared_difference,
      gamma=0.99,
      gradient_clipping=gradient_clipping,
      train_step_counter=train_step,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
  )


@gin.configurable
def train(
    root_dir: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    debug_summaries: bool = False,
    gradient_clipping: Optional[float] = None,
    learning_rate: float = 3e-4,
    log_interval: int = 1000,
    max_train_step: Optional[int] = None,
    learner_iterations_per_call: int = 1,
    batch_size: int = 0,
    policy_checkpoint_interval: int = 1000,
    timesteps_per_actorbatch: int = 0,
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment] = suite_mujoco.load,
    summarize_grads_and_vars: bool = False,
    train_checkpoint_interval: int = 1000,
    seed: Optional[int] = None,
) -> None:
  """Trains a SAC agent."""
  # Get the specs from the environment.
  logging.info('Training SAC with learning rate: %f', learning_rate)
  logging.info('Timesteps per actorbatch: %s', timesteps_per_actorbatch)

  env = suite_load_fn(environment_name)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env)
  )

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = _create_agent(
        env_name=environment_name,
        train_step=train_step,
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        learning_rate=learning_rate,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        gradient_clipping=gradient_clipping,
        seed=seed,
    )

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weights.
  saved_model_dir = os.path.join(root_dir, 'policies')
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir, agent, train_step,
      interval=policy_checkpoint_interval,
      async_saving=False,
      save_greedy_policy=True,
      save_collect_policy=True,
  )

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.push(values=variables,
                          table=reverb_variable_container.DEFAULT_TABLE)

  # Create the replay buffer client.
  reverb_client = reverb.Client(replay_buffer_server_address)

  # Create the replay buffer.
  reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      sequence_length=2,
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      server_address=replay_buffer_server_address,
  )

  # Close and delete the environment if it's no longer needed.
  env.close()
  del env
  logging.info('Closed and deleted environment.')

  def experience_dataset_fn():
    with strategy.scope():
      return reverb_replay_train.as_dataset(
          sample_batch_size=batch_size,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          num_steps=2).prefetch(3)

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=log_interval),
  ]

  sac_learner = learner.Learner(
      root_dir=root_dir,
      train_step=train_step,
      agent=agent,
      experience_dataset_fn=experience_dataset_fn,
      triggers=learning_triggers,
      max_checkpoints_to_keep=1,
      # only need a single checkpoint to resume training
      strategy=strategy,
      summary_interval=log_interval,
      checkpoint_interval=train_checkpoint_interval,
  )
  logging.info('Created learner.')

  logging.info('Training. Train step: %d out of %d', train_step.numpy(),
               max_train_step)
  while train_step < max_train_step:
    sac_learner.run(iterations=learner_iterations_per_call)
    variable_container.push(variables)
  logging.info('Training finished.')


def main(_):
  tf.compat.v1.enable_v2_behavior()

  if _DEBUG.value:
    logging.set_verbosity(logging.DEBUG)

  # Set the random seeds
  tf.random.set_seed(_SEED.value)
  np.random.seed(_SEED.value)

  # Add a prefix to our absl logger so we know which collect job this is
  absl_handler = logging.get_absl_handler()
  absl_handler.setFormatter(PrefixedLogFormatter())

  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value,
                                      finalize_config=False
                                      # a2perf environments have more configs to add
                                      )
  strategy = strategy_utils.get_strategy(tpu=_USE_TPU.value,
                                         use_gpu=FLAGS.use_gpu
                                         # Defined in tensorflow strategies
                                         )
  # Define the default dictionary for gym_kwargs
  if _ENV_NAME.value == 'QuadrupedLocomotion-v0':
    default_gym_kwargs = dict(motion_files=[_MOTION_FILE_PATH.value],
                              num_parallel_envs=_ENV_BATCH_SIZE.value)
    suite_load_function = functools.partial(
        suite_pybullet.load,
        gym_kwargs=default_gym_kwargs
    )
  elif _ENV_NAME.value == 'WebNavigation-v0':
    default_gym_kwargs = dict(
        use_legacy_step=True,
        use_legacy_reset=True,
        difficulty=_DIFFICULTY_LEVEL.value,
        num_websites=_NUM_WEBSITES.value,
        seed=0,
        browser_args=dict(
            threading=False,
            chrome_options={
                '--headless',
                '--no-sandbox'
            }
        )
    )
    suite_load_function = functools.partial(
        suite_gym.load,
        gym_kwargs=default_gym_kwargs
    )
  else:
    raise ValueError(f'Unknown environment: {_ENV_NAME.value}')

  train(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      strategy=strategy,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      debug_summaries=_DEBUG_SUMMARIES.value,
      gradient_clipping=_GRADIENT_CLIPPING.value,
      learning_rate=_LEARNING_RATE.value,
      log_interval=_LOG_INTERVAL.value,
      max_train_step=_MAX_TRAIN_STEP.value,
      learner_iterations_per_call=_LEARNER_ITERATIONS_PER_CALL.value,
      policy_checkpoint_interval=_POLICY_CHECKPOINT_INTERVAL.value,
      suite_load_fn=suite_load_function,
      summarize_grads_and_vars=_SUMMARIZE_GRADS_AND_VARS.value,
      timesteps_per_actorbatch=_TIMESTEPS_PER_ACTORBATCH.value,
      train_checkpoint_interval=_TRAIN_CHECKPOINT_INTERVAL.value,
      batch_size=_BATCH_SIZE.value,
      seed=_SEED.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'env_batch_size',
      'max_train_steps',
      'timesteps_per_actorbatch',
      'learner_iterations_per_call',
      'learning_rate',
      'batch_size',
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
