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
"""Sample training with distributed collection using a variable container."""

import os
import time

from absl import logging
import gin
import reverb
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.networks import network
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils
from tf_agents.typing import types
from tf_agents.utils import common

from . import agent
from . import learner as learner_lib


@gin.configurable(
    allowlist=[
        'per_replica_batch_size',
        'num_epochs',
        'init_learning_rate',
    ]
)
def train(
    root_dir: str,
    strategy: tf.distribute.Strategy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    action_tensor_spec: types.NestedTensorSpec,
    time_step_tensor_spec: types.NestedTensorSpec,
    sequence_length: int,
    actor_net: network.Network,
    value_net: network.Network,
    per_replica_batch_size: int = 128,
    num_epochs: int = 4,
    max_train_steps: int = 1_000_000,
    timesteps_per_actorbatch: int = 256,
    init_learning_rate: float = 0.004,
    num_netlists: int = 1,
    debug_summaries: bool = False,
    summary_interval: int = 200,  # in terms of train steps
    entropy_regularization: float = 0.0,
    use_gae: bool = False,
    shuffle_buffer_size: int = -1,
    policy_checkpoint_interval: int = 1000,

) -> None:
  """Trains a PPO agent.

  Args:
    root_dir: Main directory path where checkpoints, saved_models, and summaries
      will be written to.
    strategy: `tf.distribute.Strategy` to use during training.
    replay_buffer_server_address: Address of the reverb replay server.
    variable_container_server_address: The address of the Reverb server for
      ReverbVariableContainer.
    action_tensor_spec: Action tensor_spec.
    time_step_tensor_spec: Time step tensor_spec.
    sequence_length: Fixed sequence length for elements in the dataset. Used for
      calculating how many iterations of minibatches to use for training.
    actor_net: TF-Agents actor network.
    value_net: TF-Agents value network.
    init_train_step: Initial train step.
    per_replica_batch_size: The minibatch size for learner. The dataset used for
      training is shaped `[minibatch_size, 1, ...]`. If None, full sequences
      will be fed into the agent. Please set this parameter to None for RNN
      networks which requires full sequences.
    num_epochs: The number of iterations to go through the same sequences. The
      num_episodes_per_iteration are repeated for num_epochs times in a
      particular learner run.
    num_iterations: The number of iterations to run the training.
    num_episodes_per_iteration: This is the number of episodes we train in each
      epoch.
    init_learning_rate: Initial learning rate.
    num_netlists: Number of netlits to train used for normalizing advantage. If
      larger than 1, the advantage will be normalize first across the netlists
      then on the entire batch.
    debug_summaries: If enable summray extra information.
  """

  # Create the agent.
  with strategy.scope():
    num_replicas = strategy.num_replicas_in_sync
    train_step = train_utils.create_train_step()
    model_id = common.create_variable('model_id')

    tf_agent = agent.create_circuit_ppo_agent(
        train_step=train_step,
        action_tensor_spec=action_tensor_spec,
        time_step_tensor_spec=time_step_tensor_spec,
        actor_net=actor_net,
        value_net=value_net,
        strategy=strategy,
        learning_rate=init_learning_rate,
        entropy_regularization=entropy_regularization,
        use_gae=use_gae,
    )
    tf_agent.initialize()

  # Create the policy saver which saves the initial model now, then it
  # periodically checkpoints the policy weights.
  saved_model_dir = os.path.join(root_dir,
                                 'policies')
  save_model_trigger = triggers.PolicySavedModelTrigger(
      saved_model_dir,
      tf_agent,
      train_step,
      interval=policy_checkpoint_interval,
      async_saving=False,
      save_collect_policy=True,
      save_greedy_policy=True,
  )

  # Create the variable container.
  variables = {
      reverb_variable_container.POLICY_KEY: tf_agent.collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container = reverb_variable_container.ReverbVariableContainer(
      variable_container_server_address,
      table_names=[reverb_variable_container.DEFAULT_TABLE],
  )
  variable_container.push(variables)

  # Create the replay buffer.
  reverb_replay_trains = []
  for index in range(num_netlists):
    reverb_replay_trains += [
        reverb_replay_buffer.ReverbReplayBuffer(
            tf_agent.collect_data_spec,
            sequence_length=None,
            table_name=f'training_table_{index}',
            server_address=replay_buffer_server_address,
        )
    ]

  # Initialize the dataset.
  def experiences_dataset_fn():
    get_dtype = lambda x: x.dtype
    get_shape = lambda x: (None,) + x.shape
    shapes = tf.nest.map_structure(get_shape, tf_agent.collect_data_spec)
    dtypes = tf.nest.map_structure(get_dtype, tf_agent.collect_data_spec)

    def broadcast_info(info_traj):
      # Assumes that the first element of traj is shaped
      # (sequence_length, ...); and we extract this length.
      info, traj = info_traj
      first_elem = tf.nest.flatten(traj)[0]
      length = first_elem.shape[0] or tf.shape(first_elem)[0]
      info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
      return reverb.ReplaySample(info, traj)

    datasets = []
    for index in range(num_netlists):
      dataset = reverb.TrajectoryDataset(
          server_address=replay_buffer_server_address,
          table=f'training_table_{index}',
          dtypes=dtypes,
          shapes=shapes,
          # Menger uses learner_iterations_per_call (256). Using 8 here instead
          # because we do not need that much data in the buffer (they have to be
          # filtered out for the next iteration anyways). The rule of thumb is
          # 2-3x batch_size.
          max_in_flight_samples_per_worker=8,
          num_workers_per_iterator=-1,
          max_samples_per_stream=-1,
          rate_limiter_timeout_ms=-1,
      )
      logging.info('Created dataset for training_table_%s', index)

      datasets += [dataset.map(broadcast_info)]

    return datasets

  # Create the learner.
  learning_triggers = [
      save_model_trigger,
      triggers.StepPerSecondLogTrigger(train_step, interval=summary_interval),
  ]

  def per_sequence_fn(sample):
    # At this point, each sample data contains a sequence of trajectories.
    data, info = sample.data, sample.info
    data = tf_agent.preprocess_sequence(data)
    return data, info

  learner = learner_lib.CircuittrainingPPOLearner(
      root_dir,
      train_step=train_step,
      model_id=model_id,
      agent=tf_agent,
      experience_datasets_fn=experiences_dataset_fn,
      minibatch_size=per_replica_batch_size,
      shuffle_buffer_size=shuffle_buffer_size,
      triggers=learning_triggers,
      strategy=strategy,
      num_epochs=num_epochs,
      per_sequence_fn=per_sequence_fn,
      sequence_length=sequence_length,
      timesteps_per_actorbatch=timesteps_per_actorbatch,
  )

  # Run the training loop.ation, num_iterations):
  while train_step < max_train_steps:
    logging.info('Training. Train step: %d', train_step.numpy())
    logging.info('\tThe max train step is: %d', max_train_steps)
    step_val = train_step.numpy()
    start_time = time.time()
    if debug_summaries:
      # `wait_for_data` is not necessary and is added only to measure the data
      # latency. It takes one batch of data from dataset and print it. So, it
      # waits until the data is ready to consume.
      learner.wait_for_data()
      data_wait_time = time.time() - start_time
      logging.info('Data wait time sec: %s', data_wait_time)
    learner.run()
    run_time = time.time() - start_time
    num_steps = train_step.numpy() - step_val
    logging.info('Steps per sec: %s', num_steps / run_time)
    logging.info('Pushing variables at model_id: %d', model_id.numpy())
    variable_container.push(variables)
    logging.info('clearing replay buffers')
    for reverb_replay_train in reverb_replay_trains:
      reverb_replay_train.clear()
    with (
      learner.train_summary_writer.as_default(),
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
  logging.info('Training finished.')
