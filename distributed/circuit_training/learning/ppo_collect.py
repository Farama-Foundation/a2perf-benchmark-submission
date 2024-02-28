# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collection job using a variable container for policy updates."""

import functools
import os

import gin
from absl import app
from absl import flags
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.system import system_multiprocessing as multiprocessing

from . import ppo_collect_lib

_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode.')
_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'Paths to the gin-config files.'
)
_INITIAL_COLLECT_STEPS = flags.DEFINE_integer(
    'initial_collect_steps', 1000, 'Initial number of steps to collect.'
)
_ALGORITHM = flags.DEFINE_string(
    'algorithm',
    None,
    'Algorithm to use. Must be one of "ppo" or "sac".',
)
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin binding parameters.'
)
_NETLIST_FILE = flags.DEFINE_string('netlist_file', '',
                                    'File path to the netlist file.')
_INIT_PLACEMENT = flags.DEFINE_string(
    'init_placement', '', 'File path to the init placement file.'
)
_STD_CELL_PLACER_MODE = flags.DEFINE_string(
    'std_cell_placer_mode',
    'dreamplace',
    (
        'Options for fast std cells placement: `fd` (uses the '
        'force-directed algorithm), `dreamplace` (uses DREAMPlace '
        'algorithm).'
    ),
)

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_SUMMARY_INTERVAL = flags.DEFINE_integer(
    'summary_interval', 100, 'Interval for writing summaries.'
)

_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)

_TASK = flags.DEFINE_integer(
    'task', 0, 'Identifier of the collect task. Must be unique in a job.'
)
_GLOBAL_SEED = flags.DEFINE_integer(
    'seed',
    111,
    'Used in env and weight initialization, does not impact action sampling.',
)
_NETLIST_INDEX = flags.DEFINE_integer(
    'netlist_index', 0, 'Index of the netlist in the agent policy model.'
)

_POLICY_SAVED_MODEL_DIR = flags.DEFINE_string(
    'policy_saved_model_dir', None, 'If set, load the pretrained policy model.'
)
_POLICY_CHECKPOINT_DIR = flags.DEFINE_string(
    'policy_checkpoint_dir', None, 'If set, load the pretrained policy model.'
)
_MAX_VOCAB_SIZE = flags.DEFINE_integer(
    'max_vocab_size', None, 'Maximum vocabulary size.'
)
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', None, 'Latent dimension of the LSTM.'
)
_PROFILE_VALUE_DROPOUT = flags.DEFINE_float(
    'profile_value_dropout', None, 'Profile value dropout.'
)
_NUM_REPLICAS = flags.DEFINE_integer(
    'num_replicas', None, 'Number of replicas.'
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', None, 'Embedding dimension of the LSTM.'
)

_EPSILON_GREEDY = flags.DEFINE_float(
    'epsilon_greedy', None, 'Epsilon greedy value.'
)
_EXPLORATION_NOISE_STD = flags.DEFINE_float(
    'exploration_noise_std', None, 'Exploration noise std.'
)

_MAX_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'max_sequence_length',
    None,
    'Length of sequences to sample from the replay buffer.',
)
_NUM_WEBSITES = flags.DEFINE_integer(
    'num_websites', None, 'Number of websites to use.'
)
_DIFFICULTY_LEVEL = flags.DEFINE_integer(
    'difficulty_level', None, 'Difficulty of the task.'
)
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', None, 'Log interval.')

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

_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')

_ENTROPY_REGULARIZATION = flags.DEFINE_float(
    'entropy_regularization', None, 'Entropy regularization.'
)
_MAX_TRAIN_STEP = flags.DEFINE_integer(
    'max_train_steps', None, 'Number of iterations.'
)
_VOCABULARY_MANAGER_AUTH_KEY = flags.DEFINE_string(
    'vocabulary_manager_auth_key',
    None,
    'Authentication key for the manager server.',
)
_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', None, 'Address for the vocabulary manager.'
)
_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', None, 'Vocabulary server port.'
)


def main(_):
  gin.parse_config_files_and_bindings(
      _GIN_FILE.value, _GIN_BINDINGS.value, skip_unknown=True,
      finalize_config=False
  )
  root_dir = _ROOT_DIR.value

  gym_kwargs = dict(
      netlist_file=_NETLIST_FILE.value,
      init_placement=_INIT_PLACEMENT.value,
      global_seed=_GLOBAL_SEED.value,
      std_cell_placer_mode=_STD_CELL_PLACER_MODE.value,
      netlist_index=_NETLIST_INDEX.value,
  )
  create_env_fn = functools.partial(
      suite_gym.load,
      gym_kwargs=gym_kwargs,
      env_wrappers=[wrappers.ActionClipWrapper],

  )

  ppo_collect_lib.collect(
      task=_TASK.value,
      root_dir=root_dir,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      create_env_fn=create_env_fn,
      sequence_length=_MAX_SEQUENCE_LENGTH.value,
      netlist_index=_NETLIST_INDEX.value,
      max_train_steps=_MAX_TRAIN_STEP.value,
      summary_interval=_SUMMARY_INTERVAL.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'replay_buffer_server_address',
      'variable_container_server_address',
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
