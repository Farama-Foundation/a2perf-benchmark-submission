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
import time
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Text

from a2perf.domains import circuit_training
from a2perf.domains import quadruped_locomotion
from a2perf.domains import web_navigation
from absl import app
from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.environments import wrappers
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from . import agents
from . import learners

_TASK_INDEX = flags.DEFINE_integer(
    'task_index', 0, 'Index of the netlist in the agent policy model.'
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
_TASK_NAME = flags.DEFINE_string('task_name', None, 'Name of the task.')

_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations', None, 'Number of iterations.'
)
_SHUFFLE_BUFFER_SIZE = flags.DEFINE_integer(
    'shuffle_buffer_size',
    None,
    'Size of the shuffle buffer for the training dataset.',
)
_NUM_EPISODES_PER_ITERATION = flags.DEFINE_integer(
    'num_episodes_per_iteration',
    None,
    'Number of episodes per iteration.',
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

_NETLIST_FILE = flags.DEFINE_string(
    'netlist_file', '', 'File path to the netlist file.'
)
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
_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode')
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
_SUMMARIZE_GRADS_AND_VARS = flags.DEFINE_bool(
    'summarize_grads_and_vars', False, 'Whether to summarize grads and vars.'
)
_LEARNING_RATE = flags.DEFINE_float('learning_rate', None, 'Learning rate.')
_USE_GAE = flags.DEFINE_bool('use_gae', None, 'Whether to use GAE or not.')
FLAGS = flags.FLAGS


def create_replay_buffers(tf_agent, tasks, replay_buffer_server_address):
  reverb_replay_trains = []
  for i, index in enumerate(tasks):
    reverb_replay_trains += [
        reverb_replay_buffer.ReverbReplayBuffer(
            tf_agent.collect_data_spec,
            sequence_length=None,
            table_name=f'training_table_{i}',
            server_address=replay_buffer_server_address,
        )
    ]
  return reverb_replay_trains


def compute_init_iteration_on_policy(
    init_train_step: int,
    sequence_length: int,
    num_episodes_per_iteration: int,
    num_epochs: int,
    per_replica_batch_size: int,
    num_replicas_in_sync: int,
) -> int:
  """Computes the initial iterations number.

  In case of restarting, the init_train_step might not be zero. We need to
  compute the initial iteration number to offset the total number of iterations.

  Args:
    init_train_step: Initial train step.
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_replicas_in_sync: The number of replicas training in sync.

  Returns:
    The initial iteration number.
  """
  return int(
      init_train_step
      * per_replica_batch_size
      * num_replicas_in_sync
      / sequence_length
      / num_episodes_per_iteration
      / num_epochs
  )


def train_on_policy(
    train_step,
    max_train_step,
    debug_summaries,
    learner_obj,
    model_id,
    variable_container,
    variables,
    reverb_replay_trains,
    init_iteration,
    num_iterations,
):
  for i in range(init_iteration, num_iterations):
    step_val = train_step.numpy()
    logging.info('Training. Iteration: %d', i)
    start_time = time.time()
    if debug_summaries:
      # `wait_for_data` is not necessary and is added only to measure the data
      # latency. It takes one batch of data from dataset and print it. So, it
      # waits until the data is ready to consume.
      learner_obj.wait_for_data()
      data_wait_time = time.time() - start_time
      logging.info('Data wait time sec: %s', data_wait_time)
    learner_obj.run()
    run_time = time.time() - start_time
    num_steps = train_step.numpy() - step_val
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    logging.info('%d train steps out of %d', train_step.numpy(), max_train_step)
    variable_container.push(variables)
    logging.info('clearing replay buffers')
    for reverb_replay_train in reverb_replay_trains:
      reverb_replay_train.clear()
    with (
        learner_obj.train_summary_writer.as_default(),
        common.soft_device_placement(),
        tf.summary.record_if(lambda: True),
    ):
      with tf.name_scope('RunTime/'):
        tf.summary.scalar(
            name='step_per_sec', data=num_steps / run_time, step=train_step
        )
        if debug_summaries:
          tf.summary.scalar(
              name='data_wait_time_sec', data=data_wait_time, step=train_step
          )


def train_off_policy(
    train_step,
    learner_obj,
    model_id,
    variable_container,
    variables,
    init_iteration,
    num_iterations,
    learner_iterations_per_call,
):
  for i in range(init_iteration, num_iterations):
    step_val = train_step.numpy()
    logging.info('Training. Iteration: %d', i)
    start_time = time.time()
    learner_obj.run(learner_iterations_per_call)
    model_id.assign_add(1)
    run_time = time.time() - start_time
    num_steps = train_step.numpy() - step_val
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    variable_container.push(variables)
    with (
        learner_obj.train_summary_writer.as_default(),
        common.soft_device_placement(),
        tf.summary.record_if(lambda: True),
    ):
      with tf.name_scope('RunTime/'):
        tf.summary.scalar(
            name='step_per_sec', data=num_steps / run_time, step=train_step
        )
      with tf.name_scope('LearningRate/'):
        optimizer = getattr(learner_obj._agent, '_optimizer', None)
        if optimizer is not None:
          tf.summary.scalar(
              name='learning_rate',
              data=learner_obj._agent._optimizer.learning_rate,
              step=train_step,
          )


@gin.configurable
def train(
    root_dir: Text,
    algorithm: Text,
    task_name: Text,
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
    policy_checkpoint_interval: int = 1000,
    sequence_length: int = 0,
    suite_load_fn: Callable[
        [Text], py_environment.PyEnvironment
    ] = suite_mujoco.load,
    summarize_grads_and_vars: bool = False,
    train_checkpoint_interval: int = 1000,
    use_gae: bool = True,
    # Set to a very large number so the learning rate remains the same, and
    # also the deadline stops the training rather than this param.
    shuffle_buffer_size: int = 3,
    num_iterations: int = 1_000_000_000,
    # This is the number of episodes we train on in each iteration.
    # num_episodes_per_iteration * epsisode_length * num_epochs =
    # global_step (number of gradient updates) * per_replica_batch_size *
    # num_replicas.
    num_episodes_per_iteration: int = 256,
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

  if environment_name == 'CircuitTraining-v0':
    # Add static features
    static_features = env.wrapped_env().get_static_obs()
    env_kwargs = {
        'static_features': static_features,
    }

    # Also supply
  elif environment_name == 'WebNavigation-v0':
    env_kwargs = {}
  elif environment_name == 'QuadrupedLocomotion-v0':
    env_kwargs = {}
  else:
    raise ValueError(f'Unknown environment: {environment_name}')

  env.close()
  del env

  if algorithm == 'ppo':
    algo_kwargs = {
        'entropy_regularization': entropy_regularization,
        'use_gae': use_gae,
        'learning_rate': learning_rate,
    }
  elif algorithm == 'sac':
    algo_kwargs = {
        'learning_rate': learning_rate,
    }
  elif algorithm == 'ddqn':
    algo_kwargs = {
        'epsilon_greedy': epsilon_greedy,
        'learning_rate': learning_rate,
    }
  elif algorithm == 'td3':
    algo_kwargs = {
        'learning_rate': learning_rate,
        'exploration_noise_std': exploration_noise_std,
    }
  elif algorithm == 'ddpg':
    algo_kwargs = {
        'learning_rate': learning_rate,
    }
  else:
    raise ValueError(f'Unknown algorithm: {algorithm}')

  # Create the agent.
  with strategy.scope():
    train_step = train_utils.create_train_step()
    model_id = common.create_variable('model_id')

    agent = agents.create_agent(
        environment_name=environment_name,
        algorithm=algorithm,
        train_step=train_step,
        max_train_step=max_train_step,
        observation_tensor_spec=observation_tensor_spec,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        gradient_clipping=gradient_clipping,
        seed=seed,
        max_vocab_size=max_vocab_size,
        latent_dim=latent_dim,
        strategy=strategy,
        profile_value_dropout=profile_value_dropout,
        embedding_dim=embedding_dim,
        algo_kwargs=algo_kwargs,
        **env_kwargs,
    )

    logging.info('Created agent.')

    # Create th e policy saver which saves the initial model now, then it
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
        'model_id': model_id,
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

    learner_obj = learners.create_learner(
        algorithm=algorithm,
        model_id=model_id,
        agent=agent,
        sequence_length=sequence_length,
        replay_buffer_server_address=replay_buffer_server_address,
        num_episodes_per_iteration=num_episodes_per_iteration,
        num_epochs=num_epochs,
        batch_size=batch_size,
        train_step=train_step,
        root_dir=root_dir,
        train_checkpoint_interval=train_checkpoint_interval,
        log_interval=log_interval,
        learning_triggers=learning_triggers,
        strategy=strategy,
    )
    reverb_replay_trains = create_replay_buffers(
        tf_agent=agent,
        replay_buffer_server_address=replay_buffer_server_address,
        tasks=[task_name],
    )

    if algorithm == 'ppo':
      init_iteration = compute_init_iteration_on_policy(
          train_step,
          sequence_length,
          num_episodes_per_iteration,
          num_epochs,
          batch_size,
          strategy.num_replicas_in_sync,
      )
      logging.info(
          'Initialize iteration at: init_iteration %s.', init_iteration
      )
      model_id.assign(init_iteration)

      # Push the variables to the variable container before starting the
      # training loop. This is to stop the learner from hanging since it will
      # initially see sequences from model id 0.
      variable_container.push(variables)
      train_on_policy(
          train_step=train_step,
          max_train_step=max_train_step,
          debug_summaries=debug_summaries,
          learner_obj=learner_obj,
          model_id=model_id,
          variable_container=variable_container,
          variables=variables,
          reverb_replay_trains=reverb_replay_trains,
          num_iterations=num_iterations,
          init_iteration=init_iteration,
      )

    else:
      init_iteration = train_step.numpy()
      logging.info(
          'Initialize iteration at: init_iteration %s.', init_iteration
      )
      model_id.assign(train_step)

      variable_container.push(variables)
      train_off_policy(
          train_step=train_step,
          learner_obj=learner_obj,
          model_id=model_id,
          variable_container=variable_container,
          variables=variables,
          num_iterations=num_iterations,
          init_iteration=init_iteration,
          learner_iterations_per_call=learner_iterations_per_call,
      )

    # Create root_dir/training_complete file to signal training completion
    with open(os.path.join(root_dir, 'training_complete'), 'w') as f:
      f.write('Training complete.')


def main(_):
  if _DEBUG.value:
    logging.set_verbosity(logging.DEBUG)
    # tf.config.run_functions_eagerly(True)
    # tf.data.experimental.enable_debug_mode()

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
  elif _ENV_NAME.value == 'CircuitTraining-v0':
    default_gym_kwargs = dict(
        netlist_file=_NETLIST_FILE.value,
        init_placement=_INIT_PLACEMENT.value,
        global_seed=_SEED.value,
        std_cell_placer_mode=_STD_CELL_PLACER_MODE.value,
        netlist_index=_TASK_INDEX.value,
    )
    suite_load_function = functools.partial(
        suite_gym.load,
        gym_kwargs=default_gym_kwargs,
        env_wrappers=[wrappers.ActionClipWrapper],
    )

  else:
    raise ValueError(f'Unknown environment: {_ENV_NAME.value}')
  logging.info('Args passed to gym: %s', default_gym_kwargs)

  train(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      strategy=strategy,
      task_name=_TASK_NAME.value,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      debug_summaries=_DEBUG.value,
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
      exploration_noise_std=_EXPLORATION_NOISE_STD.value,
      seed=_SEED.value,
      algorithm=_ALGORITHM.value,
      num_episodes_per_iteration=_NUM_EPISODES_PER_ITERATION.value,
      max_vocab_size=_MAX_VOCAB_SIZE.value,
      latent_dim=_LATENT_DIM.value,
      profile_value_dropout=_PROFILE_VALUE_DROPOUT.value,
      embedding_dim=_EMBEDDING_DIM.value,
      epsilon_greedy=_EPSILON_GREEDY.value,
      num_iterations=_NUM_ITERATIONS.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'num_episodes_per_iteration',
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
      'task_name',
      'algorithm',
  ])
  multiprocessing.handle_main(lambda _: app.run(main))
