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
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner as learner_lib
from tf_agents.train import ppo_learner as ppo_learner_lib
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

# noinspection PyUnresolvedReferences
from a2perf.domains import circuit_training
# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion
# noinspection PyUnresolvedReferences
from a2perf.domains import web_navigation
from .agents import _create_ddpg_agent
from .agents import _create_ddqn_agent
from .agents import _create_ppo_agent
from .agents import _create_sac_agent
from .agents import _create_td3_agent

_MAX_VOCAB_SIZE = flags.DEFINE_integer(
    'max_vocab_size', None, 'Maximum vocabulary size.'
)
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', None, 'Latent dimension of the LSTM.'
)
_PROFILE_VALUE_DROPOUT = flags.DEFINE_float(
    'profile_value_dropout', None, 'Profile value dropout.'
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', None, 'Embedding dimension of the LSTM.'
)

_LEARNER_ITERATIONS_PER_CALL = flags.DEFINE_integer(
    'learner_iterations_per_call',
    None,
    'Number of iterations per learner call.',
)

_EPSILON_GREEDY = flags.DEFINE_float(
    'epsilon_greedy', None, 'Epsilon greedy value.'
)
_EXPLORATION_NOISE_STD = flags.DEFINE_float(
    'exploration_noise_std', None, 'Exploration noise std.'
)

_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'shuffle_buffer_size',
    None,
    'Size of the shuffle buffer for the training dataset.',
)

_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    None,
    'Length of sequences to sample from the replay buffer.',
)
_SEED = flags.DEFINE_integer('seed', None, 'Random seed.')
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_NUM_WEBSITES = flags.DEFINE_integer(
    'num_websites', None, 'Number of websites to use.'
)
_DIFFICULTY_LEVEL = flags.DEFINE_integer(
    'difficulty_level', None, 'Difficulty of the task.'
)
_DEBUG = flags.DEFINE_bool('debug', None, 'Debug mode')
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', None, 'Log interval.')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_POLICY_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'policy_checkpoint_interval', None, 'Policy checkpoint interval.'
)

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
    'Variable container server address.',
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')
_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'Paths to the gin-config files.'
)
_ENTROPY_REGULARIZATION = flags.DEFINE_float(
    'entropy_regularization', None, 'Entropy regularization.'
)
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', None, 'Gin binding parameters.'
)
_MAX_TRAIN_STEP = flags.DEFINE_integer(
    'max_train_steps', None, 'Number of iterations.'
)
_ALGORITHM = flags.DEFINE_string(
    'algorithm',
    None,
    'Algorithm to use. Must be one of "ppo" or "sac".',
)
_GRADIENT_CLIPPING = flags.DEFINE_float(
    'gradient_clipping', None, 'Gradient clipping.'
)
_DEBUG_SUMMARIES = flags.DEFINE_bool(
    'debug_summaries', False, 'Whether to use debug summaries.'
)
_SUMMARIZE_GRADS_AND_VARS = flags.DEFINE_bool(
    'summarize_grads_and_vars', False, 'Whether to summarize grads and vars.'
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', None, 'Learning rate.')
_USE_GAE = flags.DEFINE_bool('use_gae', None, 'Whether to use GAE or not.')
FLAGS = flags.FLAGS


@gin.configurable
def train(
    root_dir: Text,
    algorithm: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    debug_summaries: bool = False,
    entropy_regularization: float = 0.0,
    exploration_noise_std: float = 0.1,
    epsilon_greedy: float = 0.1,
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
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment
    ] = suite_mujoco.load,
    summarize_grads_and_vars: bool = False,
    train_checkpoint_interval: int = 1000,
    use_gae: bool = True,
    env_batch_size: int = 1,
    seed: Optional[int] = None,
    max_vocab_size: Optional[int] = None,
    latent_dim: Optional[int] = None,
    profile_value_dropout: Optional[float] = None,
    embedding_dim: Optional[int] = None,
) -> None:
  env = suite_load_fn(environment_name)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env)
  )

  if algorithm == 'ppo':
    algo_kwargs = {
        'entropy_regularization': entropy_regularization,
        'use_gae': use_gae,
        'learning_rate': learning_rate,
    }

    create_agent_fn = _create_ppo_agent

  elif algorithm == 'sac':
    algo_kwargs = {
        'learning_rate': learning_rate,
    }
    create_agent_fn = _create_sac_agent
  elif algorithm == 'ddqn':
    algo_kwargs = {
        'epsilon_greedy': epsilon_greedy,
        'learning_rate': learning_rate,
    }
    create_agent_fn = _create_ddqn_agent
  elif algorithm == 'td3':
    algo_kwargs = {
        'learning_rate': learning_rate,
        'exploration_noise_std': exploration_noise_std,
    }
    create_agent_fn = _create_td3_agent
  elif algorithm == 'ddpg':
    algo_kwargs = {
        'learning_rate': learning_rate,
    }
    create_agent_fn = _create_ddpg_agent
  else:
    raise ValueError(f'Unknown algorithm: {algorithm}')

  # Create the agent.
  with strategy.scope():
    num_replicas = strategy.num_replicas_in_sync
    train_step = train_utils.create_train_step()

    agent = create_agent_fn(
        env_name=environment_name,
        train_step=train_step,
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        gradient_clipping=gradient_clipping,
        seed=seed,
        max_vocab_size=max_vocab_size,
        latent_dim=latent_dim,
        profile_value_dropout=profile_value_dropout,
        embedding_dim=embedding_dim,
        **algo_kwargs,
    )
    logging.info('Created agent.')

    # Create the policy saver which saves the initial model now, then it
    # periodically checkpoints the policy weights.
    saved_model_dir = os.path.join(root_dir, 'policies')
    save_model_trigger = triggers.PolicySavedModelTrigger(
        saved_model_dir,
        agent,
        train_step,
        interval=policy_checkpoint_interval,
        async_saving=False,
        save_greedy_policy=True,
        save_collect_policy=True,
    )

    variables = {
        reverb_variable_container.POLICY_KEY: agent.collect_policy.variables(),
        reverb_variable_container.TRAIN_STEP_KEY: train_step,
    }

    variable_container = reverb_variable_container.ReverbVariableContainer(
        variable_container_server_address,
        table_names=[reverb_variable_container.DEFAULT_TABLE],
    )
    variable_container.push(
        values=variables, table=reverb_variable_container.DEFAULT_TABLE
    )

    # Create the learner.
    learning_triggers = [
        save_model_trigger,
        triggers.StepPerSecondLogTrigger(train_step, interval=log_interval),
    ]

    if algorithm in ('ppo',):
      # More replicas results in fewer train steps per epoch.
      max_train_step //= num_replicas
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

      def experience_dataset_fn():
        return reverb_replay_train.as_dataset(
            sample_batch_size=env_batch_size,
            num_steps=sequence_length,
            sequence_preprocess_fn=agent.preprocess_sequence,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

      def normalization_dataset_fn():
        return reverb_replay_train.as_dataset(
            sample_batch_size=env_batch_size,
            num_steps=sequence_length,
            sequence_preprocess_fn=agent.preprocess_sequence,
            num_parallel_calls=tf.data.AUTOTUNE,
        ).prefetch(tf.data.AUTOTUNE)

      # Add an `after_train_step_fn` with metrics on how on-policy the data is.
      timesteps_per_actorbatch = env_batch_size * sequence_length
      train_steps_per_policy_update = (
          timesteps_per_actorbatch * num_epochs // num_replicas // batch_size
      )
      logging.info(
          'Train steps per policy update: %d', train_steps_per_policy_update
      )
      after_train_strategy_step_fn = (
          train_utils.create_staleness_metrics_after_train_step_fn(
              train_step=train_step,
              train_steps_per_policy_update=train_steps_per_policy_update,
          )
      )

      create_learner_fn = functools.partial(
          ppo_learner_lib.PPOLearner,
          root_dir=root_dir,
          train_step=train_step,
          agent=agent,
          experience_dataset_fn=experience_dataset_fn,
          normalization_dataset_fn=normalization_dataset_fn,
          num_samples=1,
          num_epochs=num_epochs,
          checkpoint_interval=train_checkpoint_interval,
          shuffle_buffer_size=shuffle_buffer_size,
          minibatch_size=batch_size,
          summary_interval=log_interval,
          triggers=learning_triggers,
          strategy=strategy,
          after_train_strategy_step_fn=after_train_strategy_step_fn,
      )
    elif algorithm in ('sac', 'ddqn', 'td3', 'ddpg'):
      reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
          agent.collect_data_spec,
          sequence_length=2,
          table_name=reverb_replay_buffer.DEFAULT_TABLE,
          server_address=replay_buffer_server_address,
      )
      reverb_replay_normalization = None

      def experience_dataset_fn():
        return reverb_replay_train.as_dataset(
            sample_batch_size=batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            num_steps=2,
        ).prefetch(tf.data.AUTOTUNE)

      create_learner_fn = functools.partial(
          learner_lib.Learner,
          root_dir=root_dir,
          train_step=train_step,
          agent=agent,
          experience_dataset_fn=experience_dataset_fn,
          checkpoint_interval=train_checkpoint_interval,
          summary_interval=log_interval,
          triggers=learning_triggers,
          strategy=strategy,
      )
    else:
      raise ValueError(f'Unknown algorithm: {algorithm}')

    # Close and delete the environment since we have the spec.
    env.close()
    del env

    learner = create_learner_fn()
    if algorithm == 'ppo':

      def _learner_run_fn():
        learner.run()

    else:

      def _learner_run_fn():
        learner.run(iterations=learner_iterations_per_call)

    # Run the training loop.
    while train_step < max_train_step:
      logging.info('Training. Train step: %d', train_step.numpy())
      logging.info('\tThe max train step is: %d', max_train_step)
      _learner_run_fn()
      variable_container.push(variables)
      logging.info('\tPushed variables to variable container.')
      if algorithm == 'ppo':
        reverb_replay_train.clear()
        logging.info('\tCleared training replay buffer.')

        reverb_replay_normalization.clear()
        logging.info('\tCleared normalization replay buffer.')

  logging.info('Training finished.')

  # Cleanup so process exits
  del agent
  del learner
  del variable_container
  del strategy

  # Create root_dir/training_complete file to signal training completion
  with open(os.path.join(root_dir, 'training_complete'), 'w') as f:
    f.write('Training complete.')
  

def main(_):
  if _DEBUG.value:
    logging.set_verbosity(logging.DEBUG)

  # Set the random seeds
  tf.random.set_seed(_SEED.value)
  np.random.seed(_SEED.value)

  # A2Perf environments may have more configs to add, so don't finalize
  gin.parse_config_files_and_bindings(
      _GIN_FILE.value,
      _GIN_BINDINGS.value,
      finalize_config=False,
  )

  # FLAGS.use_gpu is defined in tensorflow strategy import
  strategy = strategy_utils.get_strategy(
      tpu=_USE_TPU.value,
      use_gpu=FLAGS.use_gpu,
  )
  if _ENV_NAME.value == 'QuadrupedLocomotion-v0':
    default_gym_kwargs = dict(
        motion_files=[_MOTION_FILE_PATH.value],
        num_parallel_envs=_ENV_BATCH_SIZE.value,
    )
    suite_load_function = functools.partial(
        suite_pybullet.load, gym_kwargs=default_gym_kwargs
    )
  elif _ENV_NAME.value == 'WebNavigation-v0':
    # Set the budget for TF data autotuning
    default_gym_kwargs = dict(
        use_legacy_step=True,
        use_legacy_reset=True,
        difficulty=_DIFFICULTY_LEVEL.value,
        num_websites=_NUM_WEBSITES.value,
        seed=0,
        browser_args=dict(
            threading=False, chrome_options={'--headless', '--no-sandbox'}
        ),
    )
    suite_load_function = functools.partial(
        suite_gym.load, gym_kwargs=default_gym_kwargs
    )
  else:
    raise ValueError(f'Unknown environment: {_ENV_NAME.value}')
  logging.info('Args passed to gym: %s', default_gym_kwargs)

  train(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      strategy=strategy,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      debug_summaries=_DEBUG_SUMMARIES.value,
      entropy_regularization=_ENTROPY_REGULARIZATION.value,
      gradient_clipping=_GRADIENT_CLIPPING.value,
      learner_iterations_per_call=_LEARNER_ITERATIONS_PER_CALL.value,
      learning_rate=_LEARNING_RATE.value,
      log_interval=_LOG_INTERVAL.value,
      max_train_step=_MAX_TRAIN_STEP.value,
      num_epochs=_NUM_EPOCHS.value,
      policy_checkpoint_interval=_POLICY_CHECKPOINT_INTERVAL.value,
      sequence_length=_SEQUENCE_LENGTH.value,
      suite_load_fn=suite_load_function,
      summarize_grads_and_vars=_SUMMARIZE_GRADS_AND_VARS.value,
      train_checkpoint_interval=_TRAIN_CHECKPOINT_INTERVAL.value,
      use_gae=_USE_GAE.value,
      batch_size=_BATCH_SIZE.value,
      shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE.value,
      env_batch_size=_ENV_BATCH_SIZE.value,
      exploration_noise_std=_EXPLORATION_NOISE_STD.value,
      seed=_SEED.value,
      algorithm=_ALGORITHM.value,
      max_vocab_size=_MAX_VOCAB_SIZE.value,
      latent_dim=_LATENT_DIM.value,
      profile_value_dropout=_PROFILE_VALUE_DROPOUT.value,
      embedding_dim=_EMBEDDING_DIM.value,
      epsilon_greedy=_EPSILON_GREEDY.value,
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
      'batch_size',
      'debug',
      'num_epochs',
      'max_train_steps',
      'learning_rate',
      'entropy_regularization',
      'log_interval',
      'train_checkpoint_interval',
      'policy_checkpoint_interval',
      'learner_iterations_per_call',
      'shuffle_buffer_size',
      'algorithm',
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
