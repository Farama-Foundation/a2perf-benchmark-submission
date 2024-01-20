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

import gin
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import ppo_learner as ppo_learner_lib
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion
from a2perf.domains.web_navigation.gwob.CoDE import networks

_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'shuffle_buffer_size', None,
    'Size of the shuffle buffer for the training dataset.'
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length', None,
    'Length of sequences to sample from the replay buffer.'
)
_SEED = flags.DEFINE_integer('seed', None, 'Random seed.')
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.'
)
_NUM_WEBSITES = flags.DEFINE_integer('num_websites', None,
                                     'Number of websites to use.')
_DIFFICULTY_LEVEL = flags.DEFINE_integer('difficulty_level', None,
                                         'Difficulty of the task.')
_DEBUG = flags.DEFINE_bool('debug', None, 'Debug mode')
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', None, 'Log interval.')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_POLICY_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'policy_checkpoint_interval', None, 'Policy checkpoint interval.'
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
    'train_checkpoint_interval', None, 'Train checkpoint interval.'
)

_USE_TPU = flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or not.')
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')
_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_ENTROPY_REGULARIZATION = flags.DEFINE_float('entropy_regularization', None,
                                             'Entropy regularization.')
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
_USE_GAE = flags.DEFINE_bool('use_gae', None, 'Whether to use GAE or not.')
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


def _create_value_net(env_name: Text,
    observation_tensor_spec: types.NestedTensorSpec,
    seed: Optional[int] = None,
    **kwargs
) -> value_network.ValueNetwork:
  if env_name == 'QuadrupedLocomotion-v0':
    return value_network.ValueNetwork(
        observation_tensor_spec,
        fc_layer_params=(512, 256),
    )
  elif env_name == 'WebNavigation-v0':
    max_vocab_size = kwargs.get('max_vocab_size', None)
    latent_dim = kwargs.get('latent_dim', None)
    profile_value_dropout = kwargs.get('profile_value_dropout', None)
    embedding_dim = kwargs.get('embedding_dim', None)

    if not all(
        [max_vocab_size, latent_dim, profile_value_dropout, embedding_dim]):
      raise ValueError('Missing arguments for WebLSTMActorDistributionNetwork')

    return networks.WebLSTMValueNetwork(
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


def _create_agent(
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
    seed: Optional[int] = None,
) -> tf_agent.TFAgent:
  """Creates a PPO agent."""
  actor_net = _create_actor_net(env_name=env_name,
                                observation_tensor_spec=observation_tensor_spec,
                                action_tensor_spec=action_tensor_spec,
                                seed=seed, )

  value_net = _create_value_net(env_name=env_name,
                                observation_tensor_spec=observation_tensor_spec,
                                seed=seed, )

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                       epsilon=1e-5)

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
      num_epochs=1,  # this is a legacy argument and should always be 1
      optimizer=optimizer,
      summarize_grads_and_vars=summarize_grads_and_vars,
      time_step_spec=time_step_tensor_spec,
      train_step_counter=train_step,
      update_normalizers_in_train=False,
      use_gae=use_gae,
      use_td_lambda_return=True,
      value_net=value_net,
  )


@gin.configurable
def train(
    root_dir: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    debug_summaries: bool = False,
    entropy_regularization: float = 0.0,
    gradient_clipping: Optional[float] = None,
    learner_iterations_per_call: int = 1,
    learning_rate: float = 3e-4,
    log_interval: int = 1000,
    max_train_step: Optional[int] = None,
    num_epochs: int = 0,
    batch_size: int = 0,
    shuffle_buffer_size: int = 0,
    policy_checkpoint_interval: int = 1000,
    sequence_length: int = 0,
    timesteps_per_actorbatch: int = 0,
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment] = suite_mujoco.load,
    summarize_grads_and_vars: bool = False,
    train_checkpoint_interval: int = 1000,
    use_gae: bool = True,
    env_batch_size: int = 1,
    seed: Optional[int] = None,
) -> None:
  """Trains a PPO agent."""
  logging.info('Sequence length train: %s', sequence_length)
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
        entropy_regularization=entropy_regularization,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        use_gae=use_gae,
        gradient_clipping=gradient_clipping,
        seed=seed,
    )
    logging.info('Created agent.')

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

    # Create the replay buffer.
    reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=sequence_length,
        table_name='training_table',
        server_address=replay_buffer_server_address,
    )
    reverb_replay_normalization = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=sequence_length,
        table_name='normalization_table',
        server_address=replay_buffer_server_address,
    )

    # Close and delete the environment if it's no longer needed.
    env.close()
    del env

    # Initialize the datasets. The normalization and training dataset are kept in
    # sync (contain the same data). We leverage two tables to perform
    # deterministic sampling, so that normalization and training use the same
    # collected data.
    def experience_dataset_fn():
      with strategy.scope():
        return reverb_replay_train.as_dataset(
            sample_batch_size=1,
            sequence_preprocess_fn=agent.preprocess_sequence
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def normalization_dataset_fn():
      with strategy.scope():
        return reverb_replay_normalization.as_dataset(
            sample_batch_size=1,
            sequence_preprocess_fn=agent.preprocess_sequence
        ).prefetch(tf.data.experimental.AUTOTUNE)

    # Create the learner.
    learning_triggers = [
        save_model_trigger,
        triggers.StepPerSecondLogTrigger(train_step, interval=log_interval),
    ]

    # Add an `after_train_step_fn` with metrics on how on-policy the data is.
    train_steps_per_policy_update = int(
        learner_iterations_per_call * sequence_length * num_epochs / batch_size
    )
    logging.info('Train steps per policy update: %d',
                 train_steps_per_policy_update)
    after_train_strategy_step_fn = (
        train_utils.create_staleness_metrics_after_train_step_fn(
            train_step=train_step,
            train_steps_per_policy_update=train_steps_per_policy_update,
        )
    )

    ppo_learner = ppo_learner_lib.PPOLearner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn=experience_dataset_fn,
        normalization_dataset_fn=normalization_dataset_fn,
        num_samples=env_batch_size,
        num_epochs=num_epochs,
        minibatch_size=batch_size,
        checkpoint_interval=train_checkpoint_interval,
        shuffle_buffer_size=shuffle_buffer_size,
        summary_interval=log_interval,
        triggers=learning_triggers,
        strategy=strategy,
        after_train_strategy_step_fn=after_train_strategy_step_fn,
    )

    # Run the training loop.
    while train_step < max_train_step:
      logging.info('Training. Train step: %d', train_step.numpy())
      logging.info('\tThe max train step is: %d', max_train_step)
      ppo_learner.run()
      variable_container.push(variables)
      logging.info('\tPushed variables to variable container.')

      reverb_replay_train.clear()
      logging.info('\tCleared training replay buffer.')

      reverb_replay_normalization.clear()
      logging.info('\tCleared normalization replay buffer.')
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
      entropy_regularization=_ENTROPY_REGULARIZATION.value,
      gradient_clipping=_GRADIENT_CLIPPING.value,
      learner_iterations_per_call=1,
      learning_rate=_LEARNING_RATE.value,
      log_interval=_LOG_INTERVAL.value,
      max_train_step=_MAX_TRAIN_STEP.value,
      num_epochs=_NUM_EPOCHS.value,
      policy_checkpoint_interval=_POLICY_CHECKPOINT_INTERVAL.value,
      sequence_length=_SEQUENCE_LENGTH.value,
      suite_load_fn=suite_load_function,
      summarize_grads_and_vars=_SUMMARIZE_GRADS_AND_VARS.value,
      timesteps_per_actorbatch=_TIMESTEPS_PER_ACTORBATCH.value,
      train_checkpoint_interval=_TRAIN_CHECKPOINT_INTERVAL.value,
      use_gae=_USE_GAE.value,
      batch_size=_BATCH_SIZE.value,
      shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE.value,
      env_batch_size=_ENV_BATCH_SIZE.value,
      seed=_SEED.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'env_batch_size',
      'sequence_length',
      'seed',
      'timesteps_per_actorbatch',
      'batch_size',
      'debug',
      'num_epochs',
      'max_train_steps',
      'learning_rate',
      'entropy_regularization',
      'log_interval',
      'train_checkpoint_interval',
      'policy_checkpoint_interval',
      'shuffle_buffer_size',

  ])
  multiprocessing.handle_main(lambda _: app.run(main))
