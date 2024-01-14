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
import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.agents import tf_agent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train import ppo_learner as ppo_learner_lib
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from a2perf.domains.web_navigation.gwob.CoDE import networks
from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node

_EMBEDDING_DIM = flags.DEFINE_integer('embedding_dim', None, 'Embedding dim.')
_LATENT_DIM = flags.DEFINE_integer('latent_dim', None, 'Latent dim.')
_PROFILE_VALUE_DROPOUT = flags.DEFINE_float('profile_value_dropout', None,
                                            'Profile value dropout.')
_MAX_VOCAB_SIZE = flags.DEFINE_integer('max_vocab_size', None,
                                       'Max vocab size.')
_EPSILON_GREEDY = flags.DEFINE_float('epsilon_greedy', None, 'Epsilon greedy.')
_LEARNER_ITERATIONS_PER_CALL = flags.DEFINE_integer(
    'learner_iterations_per_call', None,
    'Number of iterations per learner call.')
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
    'timesteps_per_actorbatch', 2048, 'Number of timesteps per actorbatch.')

_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_TRAIN_CHECKPOINT_INTERVAL = flags.DEFINE_integer(
    'train_checkpoint_interval', 1000, 'Train checkpoint interval.'
)

_USE_TPU = flags.DEFINE_bool('use_tpu', False, 'Whether to use TPU or not.')
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.'
)
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'Batch size.')
_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')
_MAX_TRAIN_STEP = flags.DEFINE_integer('max_train_steps', 2000000,
                                       'Number of iterations.')
_GRADIENT_CLIPPING = flags.DEFINE_float('gradient_clipping', None,
                                        'Gradient clipping.')
_DEBUG_SUMMARIES = flags.DEFINE_bool('debug_summaries', False,
                                     'Whether to use debug summaries.')
_SUMMARIZE_GRADS_AND_VARS = flags.DEFINE_bool('summarize_grads_and_vars', False,
                                              'Whether to summarize grads and vars.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
FLAGS = flags.FLAGS


class PrefixedLogFormatter(logging.PythonFormatter):
  def format(self, record):
    original = super(PrefixedLogFormatter, self).format(record)
    return f'Train: {original}'


def _create_agent(
    train_step: tf.Variable,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: ts.TimeStep,
    learning_rate: float,
    epsilon_greedy: float,
    gradient_clipping: Optional[float],
    latent_dim: int,
    profile_value_dropout: float,
    embedding_dim: int,
    debug_summaries: bool,
    summarize_grads_and_vars: bool,
    max_vocab_size: int,
) -> tf_agent.TFAgent:
  """Creates a PPO agent."""

  q_net = networks.WebLSTMQNetwork(
      vocab_size=max_vocab_size,
      latent_dim=latent_dim,
      profile_value_dropout=profile_value_dropout,
      q_min=None,
      q_max=None,
      embedding_dim=embedding_dim,
      return_state_value=False
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                       epsilon=1e-5)

  return dqn_agent.DdqnAgent(
      time_step_spec=time_step_tensor_spec,
      action_spec=action_tensor_spec,
      optimizer=optimizer,
      train_step_counter=train_step,
      debug_summaries=debug_summaries,
      gradient_clipping=gradient_clipping,
      epsilon_greedy=epsilon_greedy,
      summarize_grads_and_vars=summarize_grads_and_vars,
      q_network=q_net,
  )


@gin.configurable
def train(
    root_dir: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    debug_summaries: bool = False,
    epsilon_greedy: float = 0.1,
    gradient_clipping: Optional[float] = None,
    learner_iterations_per_call: int = 1,
    learning_rate: float = 3e-4,
    log_interval: int = 1000,
    max_train_step: Optional[int] = None,
    batch_size: int = 0,
    policy_checkpoint_interval: int = 1000,
    timesteps_per_actorbatch: int = 0,
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment] = suite_mujoco.load,
    summarize_grads_and_vars: bool = False,
    train_checkpoint_interval: int = 1000,
    embedding_dim: int = 32,
    latent_dim: int = 32,
    profile_value_dropout: float = 0.0,
    max_vocab_size: Optional[int] = None,
) -> None:
  """Trains a PPO agent."""
  logging.info('Timesteps per actorbatch: %s', timesteps_per_actorbatch)

  env = suite_load_fn(environment_name)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env)
  )

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = _create_agent(
        train_step=train_step,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        learning_rate=learning_rate,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        gradient_clipping=gradient_clipping,
        epsilon_greedy=epsilon_greedy,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        profile_value_dropout=profile_value_dropout,
        max_vocab_size=max_vocab_size,
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

    # Create the replay buffer.
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

    # Initialize the datasets. The normalization and training dataset are kept in
    # sync (contain the same data). We leverage two tables to perform
    # deterministic sampling, so that normalization and training use the same
    # collected data.
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

    dqn_learner = learner.Learner(
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
      dqn_learner.run(iterations=learner_iterations_per_call)
      variable_container.push(variables)

    logging.info('Training finished.')


def main(_):
  tf.compat.v1.enable_v2_behavior()

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
  default_gym_kwargs = dict(
      difficulty=_DIFFICULTY_LEVEL.value,
      num_websites=_NUM_WEBSITES.value,
      seed=0,
      browser_args=dict(
          threading=False,
          chrome_options={
              '--no-sandbox',
          }
      )
  )

  # Create the partial function with the default dictionary
  suite_load_function = functools.partial(
      suite_gym.load,
      gym_kwargs=default_gym_kwargs
  )

  train(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      strategy=strategy,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      debug_summaries=_DEBUG_SUMMARIES.value,
      epsilon_greedy=_EPSILON_GREEDY.value,
      gradient_clipping=_GRADIENT_CLIPPING.value,
      learner_iterations_per_call=1,
      learning_rate=_LEARNING_RATE.value,
      log_interval=_LOG_INTERVAL.value,
      max_train_step=_MAX_TRAIN_STEP.value,
      policy_checkpoint_interval=_POLICY_CHECKPOINT_INTERVAL.value,
      suite_load_fn=suite_load_function,
      summarize_grads_and_vars=_SUMMARIZE_GRADS_AND_VARS.value,
      timesteps_per_actorbatch=_TIMESTEPS_PER_ACTORBATCH.value,
      train_checkpoint_interval=_TRAIN_CHECKPOINT_INTERVAL.value,
      batch_size=_BATCH_SIZE.value,
      # env_batch_size=_ENV_BATCH_SIZE.value,
      embedding_dim=_EMBEDDING_DIM.value,
      latent_dim=_LATENT_DIM.value,
      profile_value_dropout=_PROFILE_VALUE_DROPOUT.value,
      max_vocab_size=_MAX_VOCAB_SIZE.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'seed',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'env_batch_size',
      'num_websites',
      'difficulty_level',
      'embedding_dim',
      'latent_dim',
      'profile_value_dropout',
      'learner_iterations_per_call',
      'max_vocab_size',
      'epsilon_greedy',
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
