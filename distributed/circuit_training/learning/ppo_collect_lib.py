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
"""Library for PPO collect job."""
import os
import time
from typing import Any
from typing import Callable
from typing import Optional

import gin
import reverb
from absl import logging
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

# noinspection PyUnresolvedReferences
from a2perf.domains import circuit_training

# If we have not collected in this many seconds, run another episode. This
# prevents the training loop from being stuck when using a collector
# max_episodes_per_model limit, since various workers (including the Reverb
# server) can be preempted.
COLLECT_AT_LEAST_EVERY_SECONDS = 10 * 60
ACTOR_COLLECT_METRICS_BUFFER_SIZE = 10


@gin.configurable(allowlist=['max_timesteps_per_model'])
def collect(
    task: int,
    root_dir: str,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    create_env_fn: Callable[..., Any],
    sequence_length: int,
    summary_dir: Optional[str] = None,
    netlist_index: int = 0,
    max_timesteps_per_model: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    summary_interval: Optional[int] = None,
):
  """Collects experience using a policy updated after every episode."""
  train_step = train_utils.create_train_step()
  collect_env = create_env_fn(
      'CircuitTraining-v0')

  """Wait for the collect policy to be ready and run collect job."""
  collect_policy_dir = os.path.join(
      root_dir,
      '../../',  # two levels because collect/<hostname>/ is the root_dir
      learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR,
  )
  logging.info('Looking for collect policy in %s', collect_policy_dir)

  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True
  )
  logging.info('Loaded collect policy from %s', collect_policy_dir)

  # Create the variable container.
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.update(variables)

  # Create the replay buffer observer for collect jobs.
  env_step_metric = py_metrics.EnvironmentSteps()
  observers = [
      reverb_utils.ReverbTrajectorySequenceObserver(
          reverb.Client(replay_buffer_server_address),
          table_name=f'training_table_{netlist_index}',
          sequence_length=sequence_length,
          stride_length=sequence_length,
          priority=model_id,
      ),
      env_step_metric,
  ]

  # Create the collect actor.
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=sequence_length,
      metrics=actor.collect_metrics(
          ACTOR_COLLECT_METRICS_BUFFER_SIZE) if task == 0 else [],
      summary_dir=summary_dir if task == 0 else None,
      summary_interval=summary_interval,
      observers=observers,
  )

  # Run the experience collection loop.
  model_to_num_timesteps = {}
  last_collection_ts = 0
  prev_num_steps_collected = 0
  while train_step < max_train_steps:
    if model_id.numpy() not in model_to_num_timesteps:
      model_to_num_timesteps[model_id.numpy()] = 0

    if (
        max_timesteps_per_model is None
        or model_to_num_timesteps[model_id.numpy()] < max_timesteps_per_model
        or time.time() - last_collection_ts > COLLECT_AT_LEAST_EVERY_SECONDS
    ):
      logging.info('Collecting at model_id: %d', model_id.numpy())
      last_collection_ts = time.time()
      start_time = time.time()
      collect_actor.run()
      end_time = time.time()
      # Clear old models.
      for k in list(model_to_num_timesteps):
        if k != model_id.numpy():
          del model_to_num_timesteps[k]

      model_to_num_timesteps[model_id.numpy()] += 1
      logging.info('\tCollection took %.3f seconds', end_time - start_time)
    variable_container.update(variables)
    logging.info('Collecting with policy at step: %d', train_step.numpy())
    logging.info('\tMax train step: %d', max_train_steps)
    logging.info('\tCollected %d steps', env_step_metric.result())
    logging.info(
        '\tCollected %d steps this iteration',
        env_step_metric.result() - prev_num_steps_collected,
    )
    prev_num_steps_collected = env_step_metric.result()
