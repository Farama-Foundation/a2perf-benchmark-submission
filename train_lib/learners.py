import functools
from typing import Callable
from typing import Tuple

import gin
import reverb
import tensorflow as tf
from absl import logging
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import learner as learner_lib
from tf_agents.typing import types

from . import learner_lib as learner

# A function which processes a tuple of a nested tensor representing a TF-Agent
# Trajectory and Reverb SampleInfo.
_SequenceParamsType = Tuple[types.NestedTensor, types.ReverbSampleInfo]
_SequenceFnType = Callable[[_SequenceParamsType], _SequenceParamsType]

GIGABYTES = 1073741824  # 1 GB in bytes


def dataset_options():
  options = tf.data.Options()
  autotune_options = tf.data.experimental.AutotuneOptions()
  autotune_options.ram_budget = 2 * GIGABYTES
  options.autotune = autotune_options
  return options


def broadcast_info(info_traj):
  # Assumes that the first element of traj is shaped
  # (sequence_length, ...); and we extract this length.
  info, traj = info_traj
  first_elem = tf.nest.flatten(traj)[0]
  length = first_elem.shape[0] or tf.shape(first_elem)[0]
  info = tf.nest.map_structure(lambda t: tf.repeat(t, [length]), info)
  return reverb.ReplaySample(info, traj)


@gin.configurable(allowlist=['shuffle_buffer_episode_len'])
def get_shuffle_buffer_size(
    sequence_length: int,
    shuffle_buffer_episode_len: int = 3,
) -> int:
  """Returns shuffle buffer size.

  Args:
    sequence_length: The sequence length.
    shuffle_buffer_episode_len: The size of buffer for shuffle operation in
      dataset. The buffer size should be between 1-3 episode len.

  Returns:
    The shuffle buffer size.
  """
  return sequence_length * shuffle_buffer_episode_len


def create_off_policy_experience_dataset_fn(tf_agent, tasks, batch_size,
    replay_buffer_server_address):
  def experience_dataset_fn():
    reverb_replay_train = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name='training_table_0',
        server_address=replay_buffer_server_address,
    )

    dataset = reverb_replay_train.as_dataset(
        sample_batch_size=batch_size,
        sequence_preprocess_fn=tf_agent.preprocess_sequence,
        num_steps=2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        single_deterministic_pass=False,
    ).prefetch(3).with_options(dataset_options())
    logging.info('Created dataset for training_table_0')

    return dataset

  return experience_dataset_fn


def create_on_policy_experience_dataset_fn(tf_agent, tasks,
    replay_buffer_server_address):
  def experience_dataset_fn():
    get_dtype = lambda x: x.dtype
    get_shape = lambda x: (None,) + x.shape
    shapes = tf.nest.map_structure(get_shape, tf_agent.collect_data_spec)
    dtypes = tf.nest.map_structure(get_dtype, tf_agent.collect_data_spec)

    datasets = []
    for i, index in enumerate(tasks):
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

  return experience_dataset_fn


def create_per_sequence_fn(tf_agent):
  def per_sequence_fn(sample):
    # At this point, each sample data contains a sequence of trajectories.
    data, info = sample.data, sample.info
    data = tf_agent.preprocess_sequence(data)
    return data, info

  return per_sequence_fn


def create_ppo_learner(agent,
    sequence_length,
    replay_buffer_server_address,
    model_id,
    num_episodes_per_iteration,
    num_epochs,
    batch_size,
    train_step,
    root_dir, train_checkpoint_interval,
    learning_triggers,
    log_interval,
    strategy):
  experience_dataset_fn = create_on_policy_experience_dataset_fn(tf_agent=agent,
                                                                 tasks=[0],
                                                                 replay_buffer_server_address=replay_buffer_server_address)
  per_sequence_fn = create_per_sequence_fn(agent)
  return learner.PPOLearner(
      root_dir,
      train_step,
      model_id,
      agent,
      experience_dataset_fn,
      sequence_length=sequence_length,
      num_episodes_per_iteration=num_episodes_per_iteration,
      minibatch_size=batch_size,
      shuffle_buffer_size=get_shuffle_buffer_size(sequence_length),
      triggers=learning_triggers,
      strategy=strategy,
      num_epochs=num_epochs,
      per_sequence_fn=per_sequence_fn,
      summary_interval=log_interval,
      checkpoint_interval=train_checkpoint_interval,
  )


def create_off_policy_learner(
    agent,
    replay_buffer_server_address,
    batch_size,
    train_step, root_dir, train_checkpoint_interval,
    log_interval, learning_triggers, strategy):
  experience_dataset_fn = create_off_policy_experience_dataset_fn(
      tf_agent=agent,
      batch_size=batch_size,
      tasks=[0],
      replay_buffer_server_address=replay_buffer_server_address
  )

  return learner_lib.Learner(
      root_dir=root_dir,
      train_step=train_step,
      agent=agent,
      experience_dataset_fn=experience_dataset_fn,
      checkpoint_interval=train_checkpoint_interval,
      summary_interval=log_interval,
      triggers=learning_triggers,
      strategy=strategy,
  )


def create_learner(
    algorithm, agent, model_id, sequence_length,
    replay_buffer_server_address,
    num_episodes_per_iteration, num_epochs, batch_size,
    train_step, root_dir, train_checkpoint_interval,
    log_interval, learning_triggers, strategy):
  if algorithm in ('ppo',):
    return create_ppo_learner(
        agent=agent,
        model_id=model_id,
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
        strategy=strategy
    )
  elif algorithm in ('sac', 'ddqn', 'td3', 'ddpg'):
    return create_off_policy_learner(
        agent=agent,
        replay_buffer_server_address=replay_buffer_server_address,
        batch_size=batch_size,
        train_step=train_step,
        root_dir=root_dir,
        train_checkpoint_interval=train_checkpoint_interval,
        log_interval=log_interval,
        learning_triggers=learning_triggers,
        strategy=strategy
    )
  else:
    raise ValueError(f'Unknown algorithm: {algorithm}')
