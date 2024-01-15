"""Sample collection Job using a variable container for policy updates."""
import functools
import os

import gin
import numpy as np
import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_SEED = flags.DEFINE_integer('seed', None, 'Random seed.')
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)
_MOTION_FILE_PATH = flags.DEFINE_string(
    'motion_file_path',
    None,
    'Path to the motion file. '
    'The motion file dog_pace is in the a2perf package.',
)
_SUMMARY_INTERVAL = flags.DEFINE_integer(
    'summary_interval',
    None,
    'Interval at which to record summaries.',
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    None,
    'Size of reverb buffer to sample.',
)
_MAX_TRAIN_STEPS = flags.DEFINE_integer(
    'max_train_steps',
    None,
    'Maximum number of training steps to run.',
)
_INITIAL_COLLECT_STEPS = flags.DEFINE_integer(
    'initial_collect_steps',
    None,
    'Number of steps to run the initial collect policy.',
)
_TASK = flags.DEFINE_integer('task', None,
                             'Identifier of a collect task. Must be unique.')

_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')


class PrefixedLogFormatter(logging.PythonFormatter):
  def format(self, record):
    original = super(PrefixedLogFormatter, self).format(record)
    return f'Collect {_TASK.value}: {original}'


def collect(environment_name: str,
    collect_policy: py_tf_eager_policy.PyTFEagerPolicyBase,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    root_dir: str,
    task: int,
    summary_interval: int,
    sequence_length: int,
    initial_collect_steps: int,

    gym_kwargs=None) -> None:
  summary_dir = os.path.join(root_dir, 'summaries', str(task))

  # Create the partial function with the default dictionary
  suite_load_function = functools.partial(
      suite_pybullet.load,
      gym_kwargs=gym_kwargs,
  )
  collect_env = suite_load_function(environment_name)

  # Create the variable container.
  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.update(variables)
  reverb_client = reverb.Client(replay_buffer_server_address)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_client,
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      sequence_length=2,
      stride_length=1)

  random_policy = random_py_policy.RandomPyPolicy(
      collect_env.time_step_spec(), collect_env.action_spec()
  )
  initial_collect_actor = actor.Actor(
      collect_env,
      random_policy,
      train_step,
      steps_per_run=initial_collect_steps,
      observers=[rb_observer],
  )
  logging.info('Doing initial collect.')
  initial_collect_actor.run()
  logging.info('Done initial collect.')

  # Create the collect actor.
  env_step_metric = py_metrics.EnvironmentSteps()
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=sequence_length,
      metrics=actor.collect_metrics(10),
      summary_dir=summary_dir,
      summary_interval=summary_interval,
      observers=[rb_observer, env_step_metric],
  )

  # Run the experience collection loop.
  prev_num_steps_collected = 0
  while True:
    collect_actor.run()
    variable_container.update(variables)
    logging.info('Collecting with policy at step: %d', train_step.numpy())

    num_steps_collected = env_step_metric.result()
    logging.info('\tCollected %d steps', num_steps_collected)
    logging.info('\tCollected %d steps this iteration',
                 num_steps_collected - prev_num_steps_collected)
    prev_num_steps_collected = num_steps_collected


def run_collect(root_dir: str,
    environment_name: str,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    motion_file_path: str,
    env_batch_size: int,
    task: int,
    initial_collect_steps: int,
    summary_interval: int,
    sequence_length: int) -> None:
  """Wait for the collect policy to be ready and run collect job."""
  gym_kwargs = dict(motion_files=[motion_file_path],
                    num_parallel_envs=env_batch_size)
  collect_policy_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR,
                                    learner.COLLECT_POLICY_SAVED_MODEL_DIR)
  collect_policy = train_utils.wait_for_policy(collect_policy_dir,
                                               load_specs_from_pbtxt=True)

  collect(environment_name=environment_name,
          collect_policy=collect_policy,
          replay_buffer_server_address=replay_buffer_server_address,
          variable_container_server_address=variable_container_server_address,
          root_dir=root_dir,
          task=task,
          summary_interval=summary_interval,
          sequence_length=sequence_length,
          gym_kwargs=gym_kwargs,
          initial_collect_steps=initial_collect_steps,
          )


def main(_):
  tf.compat.v1.enable_v2_behavior()

  # Set the random seeds
  tf.random.set_seed(_SEED.value)
  np.random.seed(_SEED.value)

  absl_handler = logging.get_absl_handler()
  absl_handler.setFormatter(PrefixedLogFormatter())
  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value,
                                      finalize_config=False)

  run_collect(root_dir=_ROOT_DIR.value,
              environment_name=_ENV_NAME.value,
              replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
              variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
              motion_file_path=_MOTION_FILE_PATH.value,
              env_batch_size=_ENV_BATCH_SIZE.value,
              task=_TASK.value,
              summary_interval=_SUMMARY_INTERVAL.value,
              sequence_length=_SEQUENCE_LENGTH.value,
              initial_collect_steps=_INITIAL_COLLECT_STEPS.value,
              )


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['root_dir', 'env_name', 'replay_buffer_server_address',
       'variable_container_server_address',
       'sequence_length', 'env_batch_size', 'motion_file_path', 'task',
       'summary_interval', 'max_train_steps', 'initial_collect_steps', 'seed'])
  multiprocessing.handle_main(functools.partial(app.run, main))
