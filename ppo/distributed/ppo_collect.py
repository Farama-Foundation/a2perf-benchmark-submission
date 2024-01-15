"""Sample collection Job using a variable container for policy updates."""

import functools
import os
from typing import Text
import numpy as np
import gin
import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

# noinspection PyUnresolvedReferences
from a2perf.domains import quadruped_locomotion

_DIFFICULTY_LEVEL = flags.DEFINE_integer(
    'difficulty_level',
    None,
    'Difficulty level of the environment.',
)
_NUM_WEBSITES = flags.DEFINE_integer(
    'num_websites',
    None,
    'Number of websites to use in the environment.',
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', None, 'Number of environments to run in parallel.'
)
_SEED = flags.DEFINE_integer('seed', None, 'Random seed.')
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
    0,
    'Interval at which to record summaries.',
)
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length',
    None,
    'Size of reverb buffer to sample.',
)
_TASK = flags.DEFINE_integer('task', 0,
                             'Identifier of a collect task. Must be unique.')

_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')


class PrefixedLogFormatter(logging.PythonFormatter):
  def format(self, record):
    original = super(PrefixedLogFormatter, self).format(record)
    return f'Collect {_TASK.value}: {original}'


@gin.configurable
def collect(
    environment_name: Text,
    collect_policy: py_tf_eager_policy.PyTFEagerPolicyBase,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    sequence_length: int = 0,
    summary_interval: int = 0,
    suite_load_function: callable = suite_pybullet.load,
) -> None:
  """Collects experience using a policy updated after every episode."""
  logging.info('Sequence length collect: %s', sequence_length)
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

  experience_observer = reverb_utils.ReverbTrajectorySequenceObserver(
      reverb_client,
      table_name=[
          'training_table',
          'normalization_table',
      ],
      sequence_length=sequence_length,
      stride_length=sequence_length,
      priority=train_step,
  )

  # Create the collect actor.
  env_step_metric = py_metrics.EnvironmentSteps()
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=sequence_length,
      metrics=actor.collect_metrics(10),
      summary_interval=summary_interval,
      summary_dir=os.path.join(_ROOT_DIR.value, 'summaries',
                               str(_TASK.value)),
      observers=[experience_observer, env_step_metric],
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


def run_collect(
    root_dir: str,
    environment_name: str,
    sequence_length: int,
    summary_interval: int,
    suite_load_fn: callable,
) -> None:
  """Wait for the collect policy to be ready and run collect job."""
  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = os.path.join(
      root_dir,
      learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR,
  )
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True
  )

  collect(
      environment_name=environment_name,
      collect_policy=collect_policy,
      summary_interval=summary_interval,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      sequence_length=sequence_length,
      suite_load_fn=suite_load_fn,
  )


def main(_):
  # Add a prefix to our absl logger so we know which collect job this is
  absl_handler = logging.get_absl_handler()
  absl_handler.setFormatter(PrefixedLogFormatter())

  tf.compat.v1.enable_v2_behavior()

  # Set the random seeds
  tf.random.set_seed(_SEED.value)
  np.random.seed(_SEED.value)

  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value,
                                      finalize_config=False)

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

  run_collect(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      sequence_length=_SEQUENCE_LENGTH.value,
      summary_interval=_SUMMARY_INTERVAL.value,
      suite_load_fn=suite_load_function,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'sequence_length',
      'env_batch_size',
      'seed',
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
