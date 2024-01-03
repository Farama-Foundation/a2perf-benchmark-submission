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

r"""Sample collection Job using a variable container for policy updates.

See README for launch instructions.
"""
import functools
import os
from typing import Callable
from typing import Text

import gin
import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.environments import py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

from a2perf.domains import quadruped_locomotion

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_MAX_TRAIN_STEPS = flags.DEFINE_integer(
    'max_train_steps', 2000000, 'Max number of training steps.'
)
_MAX_ENV_STEPS = flags.DEFINE_integer(
    'max_env_steps', 1000, 'Max number of steps per environment.'
)
_STEPS_PER_RUN = flags.DEFINE_integer(
    'steps_per_run', 1, 'Number of environment steps to take per run.'
)
_MOTION_FILE_PATH = flags.DEFINE_string(
    'motion_file_path', None, 'Path to the motion file.'
)
_ENV_BATCH_SIZE = flags.DEFINE_integer(
    'env_batch_size', 1, 'Number of environments to run in parallel.'
)

_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)
_SUMMARY_INTERVAL = flags.DEFINE_integer(
    'summary_interval', 1000, 'Interval for writing summaries.'
)
_TASK = flags.DEFINE_integer(
    'task', 0, 'Identifier of the collect task. Must be unique in a job.'
)
_GIN_FILE = flags.DEFINE_multi_string('gin_file', None,
                                      'Paths to the gin-config files.')
_GIN_BINDINGS = flags.DEFINE_multi_string('gin_bindings', None,
                                          'Gin binding parameters.')


@gin.configurable
def collect(
    summary_dir: Text,
    environment_name: Text,
    collect_policy: py_tf_eager_policy.PyTFEagerPolicyBase,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    summary_interval: int = 1000,
    suite_load_fn: Callable[
      [Text], py_environment.PyEnvironment
    ] = suite_mujoco.load,
    max_train_steps: int = 0,
    steps_per_run: int = 0,
) -> None:
  """Collects experience using a policy updated after every episode."""
  # Create the environment. For now support only single environment collection.
  collect_env = suite_load_fn(environment_name)

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

  # Create replay buffer observer that uses steps
  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      py_client=reverb.Client(replay_buffer_server_address),
      table_name=reverb_replay_buffer.DEFAULT_TABLE,
      sequence_length=steps_per_run,
      stride_length=steps_per_run,
  )

  env_step_metric = py_metrics.EnvironmentSteps()
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=steps_per_run,
      episodes_per_run=None,
      observers=[rb_observer, env_step_metric],
      transition_observers=None,
      info_observers=None,
      metrics=actor.collect_metrics(10),
      reference_metrics=None,
      image_metrics=None,
      summary_dir=summary_dir,
      summary_interval=summary_interval,
      end_episode_on_boundary=True,
      name='Actor_{}'.format(_TASK.value),
  )

  # Run the experience collection loop.
  while train_step.numpy() < max_train_steps:
    logging.info('Collecting with policy at step: %d', train_step.numpy())
    collect_actor.run()
    variable_container.update(variables)

  logging.info('Finished collecting!')


def main(_):
  tf.config.run_functions_eagerly(True)
  gin.parse_config_files_and_bindings(_GIN_FILE.value, _GIN_BINDINGS.value,
                                      finalize_config=False
                                      # a2perf domains add additional configs
                                      )

  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = os.path.join(
      _ROOT_DIR.value,
      learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR,
  )
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True
  )

  # Prepare summary directory.
  summary_dir = os.path.join(_ROOT_DIR.value, learner.TRAIN_DIR,
                             str(_TASK.value))

  # Define the default dictionary for gym_kwargs
  default_gym_kwargs = dict(motion_files=[_MOTION_FILE_PATH.value],
                            num_parallel_envs=_ENV_BATCH_SIZE.value)

  # Create the partial function with the default dictionary
  suite_load_function = functools.partial(
      suite_pybullet.load,
      gym_kwargs=default_gym_kwargs
  )
  # Perform collection.
  collect(
      summary_dir=summary_dir,
      summary_interval=_SUMMARY_INTERVAL.value,
      suite_load_fn=suite_load_function,
      environment_name=_ENV_NAME.value,
      collect_policy=collect_policy,
      steps_per_run=_STEPS_PER_RUN.value,
      max_train_steps=_MAX_TRAIN_STEPS.value,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
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
