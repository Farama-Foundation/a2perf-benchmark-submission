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
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from a2perf.domains import quadruped_locomotion

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.'
)
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
    'env_batch_size', 1, 'Number of environments to run in parallel.'
)
_MOTION_FILE_PATH = flags.DEFINE_string(
    'motion_file_path', None, 'Path to the motion file.'
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
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 1, 'Number of epochs.')
_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_ENTROPY_REGULARIZATION = flags.DEFINE_float('entropy_regularization', 0.0,
                                             'Entropy regularization.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')
_NUM_ITERATIONS = flags.DEFINE_integer('num_iterations', 2000000,
                                       'Number of iterations.')
_GRADIENT_CLIPPING = flags.DEFINE_float('gradient_clipping', None,
                                        'Gradient clipping.')
_DEBUG_SUMMARIES = flags.DEFINE_bool('debug_summaries', False,
                                     'Whether to use debug summaries.')
_SUMMARIZE_GRADS_AND_VARS = flags.DEFINE_bool('summarize_grads_and_vars', False,
                                              'Whether to summarize grads and vars.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
_USE_GAE = flags.DEFINE_bool('use_gae', True, 'Whether to use GAE or not.')
FLAGS = flags.FLAGS


def _create_agent(
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
    num_epochs: int = 25,
) -> tf_agent.TFAgent:
  """Creates a PPO agent."""
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_tensor_spec,
      action_tensor_spec,
      fc_layer_params=(512, 256),
  )

  value_net = value_network.ValueNetwork(
      observation_tensor_spec,
      fc_layer_params=(512, 256),
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  return ppo_clip_agent.PPOClipAgent(
      num_epochs=num_epochs,
      time_step_spec=time_step_tensor_spec,
      action_spec=action_tensor_spec,
      optimizer=optimizer,
      actor_net=actor_net,
      value_net=value_net,
      entropy_regularization=entropy_regularization,
      gradient_clipping=gradient_clipping,
      use_gae=use_gae,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step,
  )


@gin.configurable
def train(
    root_dir: Text,
    environment_name: Text,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment
    ] = suite_mujoco.load,
    # Training params
    learning_rate: float = 3e-4,
    num_iterations: int = 2000000,
    learner_iterations_per_call: int = 1,
    use_gae: bool = True,
    gradient_clipping: Optional[float] = None,
    debug_summaries: bool = False,
    train_checkpoint_interval: int = 1000,
    timesteps_per_actorbatch: int = 0,
    policy_checkpoint_interval: int = 1000,
    env_batch_size: int = 0,
    summarize_grads_and_vars: bool = False,
    summary_dir: Optional[Text] = None,
    log_interval: int = 1000,
    entropy_regularization: float = 0.0,
) -> None:
  """Trains a PPO agent."""
  # Get the specs from the environment.
  logging.info('Training PPO with learning rate: %f', learning_rate)
  env = suite_load_fn(environment_name)
  observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
      spec_utils.get_tensor_specs(env)
  )

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    agent = _create_agent(
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
    )

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weigths.
  saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir, agent, train_step,
      interval=policy_checkpoint_interval
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
  variable_container.push(variables)

  # Create the replay buffer with a fixed maximum size.
  sequence_length = timesteps_per_actorbatch
  reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
      data_spec=agent.collect_data_spec,
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      sequence_length=sequence_length,
      server_address=replay_buffer_server_address,
      dataset_buffer_size=None,  # Modify if slow
      max_cycle_length=env_batch_size,
      num_workers_per_iterator=tf.data.experimental.AUTOTUNE,
  )

  # Close and delete the environment if it's no longer needed.
  env.close()
  del env

  # Initialize the dataset.
  def experience_dataset_fn():
    with strategy.scope():
      # Gathering all data from the replay buffer.
      dataset = reverb_replay.as_dataset(
          sample_batch_size=None,  # Set to None to gather all data
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          sequence_preprocess_fn=None,
          single_deterministic_pass=False,
      )

      # Rebatch the dataset.
      # rebatched_dataset = dataset.batch(env_batch_size)

      # Prefetch data for efficiency.
      # return rebatched_dataset.prefetch(tf.data.experimental.AUTOTUNE)
      return dataset.prefetch(tf.data.experimental.AUTOTUNE)

  # Use the experience_dataset_fn to get your dataset
  dataset = experience_dataset_fn()

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=log_interval),
  ]
  ppo_learner = learner.Learner(
      root_dir=root_dir,
      train_step=train_step,
      agent=agent,
      experience_dataset_fn=experience_dataset_fn,
      after_train_strategy_step_fn=None,
      triggers=learning_triggers,
      checkpoint_interval=train_checkpoint_interval,
      summary_interval=log_interval,
      max_checkpoints_to_keep=1,
      use_kwargs_in_agent_train=False,
      strategy=strategy,
      run_optimizer_variable_init=True,
      use_reverb_v2=False,
      direct_sampling=False,
      experience_dataset_options=None,
      strategy_run_options=None,
      summary_root_dir=summary_dir
  )

  # Run the training loop.
  while train_step.numpy() < num_iterations:
    # Since this is PPO, each learner iteration is a single epoch.
    ppo_learner.run(iterations=learner_iterations_per_call)
    variable_container.push(variables)
  logging.info('Training finished.')


def main(_):
  tf.config.run_functions_eagerly(True)
  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value,
                                      finalize_config=False
                                      # a2perf environments have more configs to add
                                      )
  strategy = strategy_utils.get_strategy(tpu=_USE_TPU.value,
                                         use_gpu=FLAGS.use_gpu
                                         # Defined in tensorflow strategies
                                         )
  # Define the default dictionary for gym_kwargs
  default_gym_kwargs = dict(motion_files=[_MOTION_FILE_PATH.value],
                            num_parallel_envs=_ENV_BATCH_SIZE.value)

  # Create the partial function with the default dictionary
  suite_load_function = functools.partial(
      suite_pybullet.load,
      gym_kwargs=default_gym_kwargs
  )

  train(
      debug_summaries=_DEBUG_SUMMARIES.value,
      entropy_regularization=_ENTROPY_REGULARIZATION.value,
      environment_name=_ENV_NAME.value,
      gradient_clipping=_GRADIENT_CLIPPING.value,
      learner_iterations_per_call=1,
      learning_rate=_LEARNING_RATE.value,
      log_interval=_LOG_INTERVAL.value,
      num_iterations=_NUM_ITERATIONS.value,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      root_dir=_ROOT_DIR.value,
      timesteps_per_actorbatch=_TIMESTEPS_PER_ACTORBATCH.value,
      env_batch_size=_ENV_BATCH_SIZE.value,
      strategy=strategy,
      train_checkpoint_interval=_TRAIN_CHECKPOINT_INTERVAL.value,
      policy_checkpoint_interval=_POLICY_CHECKPOINT_INTERVAL.value,
      suite_load_fn=suite_load_function,
      summary_dir=os.path.join(_ROOT_DIR.value, 'summaries'),
      summarize_grads_and_vars=_SUMMARIZE_GRADS_AND_VARS.value,
      use_gae=_USE_GAE.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
