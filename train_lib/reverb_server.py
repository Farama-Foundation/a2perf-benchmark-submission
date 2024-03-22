"""Main binary to launch a stand alone Reverb RB server."""

import os
import time

import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

_ALGORITHM = flags.DEFINE_string(
    'algorithm',
    None,
    'Algorithm to use. Must be one of "ppo" or "sac".',
)
_MIN_TABLE_SIZE_BEFORE_SAMPLING = flags.DEFINE_integer(
    'min_table_size_before_sampling',
    None,
    'Minimum size of the table before sampling.',
)
_TASK_NAMES = flags.DEFINE_multi_string(
    'task_name',
    None,
    'Names of the tasks to train on. For example, "dog_pace", "dog_trot", "dog_spin"', )

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer(
    'replay_buffer_capacity', None, 'Capacity of the replay buffer table.'
)
_PORT = flags.DEFINE_integer('port', None, 'Port to start the server on.')

PROCESS_WAIT_INTERVAL = 10


def run_reverb_server(root_dir):
  collect_policy_saved_model_path = os.path.join(
      root_dir,
      learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR,
  )
  saved_model_pb_path = os.path.join(
      collect_policy_saved_model_path, 'saved_model.pb'
  )
  policy_specs_pbtxt_path = os.path.join(
      collect_policy_saved_model_path, 'policy_specs.pbtxt'
  )
  fingerprint_pb_path = os.path.join(
      collect_policy_saved_model_path, 'fingerprint.pb'
  )
  try:
    # Wait for the collect policy to be output by learner (timeout after 2
    # days), then load it.
    train_utils.wait_for_file(
        saved_model_pb_path, sleep_time_secs=2, num_retries=86400
    )
    train_utils.wait_for_file(
        policy_specs_pbtxt_path, sleep_time_secs=2, num_retries=86400
    )
    train_utils.wait_for_file(
        fingerprint_pb_path, sleep_time_secs=2, num_retries=86400
    )
    collect_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        collect_policy_saved_model_path, load_specs_from_pbtxt=True
    )
  except TimeoutError as e:
    # If the collect policy does not become available during the wait time of
    # the call `wait_for_file`, that probably means the learner is not running.
    logging.error('Could not get the file %s. Exiting.', saved_model_pb_path)
    raise e

  # Create the signature for the variable container holding the policy weights.
  train_step = train_utils.create_train_step()
  model_id = common.create_variable('model_id')
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
      'model_id': model_id,
  }
  variable_container_signature = tf.nest.map_structure(
      lambda variable: tf.TensorSpec(variable.shape, dtype=variable.dtype),
      variables,
  )
  logging.info('Signature of variables: \n%s', variable_container_signature)

  # Create the signature for the replay buffer holding observed experience.
  replay_buffer_signature = tensor_spec.from_spec(
      collect_policy.collect_data_spec
  )
  replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
  logging.info('Signature of experience: \n%s', replay_buffer_signature)

  # The remover does not matter because we clear the table at the end
  # of each global step. We assume that the table is large enough to
  # contain the data collected from one step (otherwise some data will
  # be dropped).
  training_tables = []
  for index in range(len(_TASK_NAMES.value)):

    if _ALGORITHM.value == 'ppo':
      training_tables += [
          reverb.Table(  # Replay buffer storing experience for training.
              name=f'training_table_{index}',
              sampler=reverb.selectors.MaxHeap(),
              remover=reverb.selectors.MinHeap(),
              # Menger sets this to 8, but empirically 1 learns better
              # consistently.
              rate_limiter=reverb.rate_limiters.MinSize(
                  _MIN_TABLE_SIZE_BEFORE_SAMPLING.value
              ),
              max_size=_REPLAY_BUFFER_CAPACITY.value,
              max_times_sampled=1,
              signature=replay_buffer_signature,
          )
      ]

    elif _ALGORITHM.value in ('sac', 'ddqn', 'td3', 'dqn', 'ddpg'):
      training_tables += [
          reverb.Table(  # Replay buffer storing experience for training.
              name=f'training_table_{index}',
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=_REPLAY_BUFFER_CAPACITY.value,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          )]
    else:
      raise ValueError(f'Unsupported algorithm: {_ALGORITHM.value}')
    server = reverb.Server(
        tables=training_tables
               + [
                   reverb.Table(
                       # Variable container storing policy parameters.
                       name=reverb_variable_container.DEFAULT_TABLE,
                       sampler=reverb.selectors.Fifo(),
                       remover=reverb.selectors.Fifo(),
                       rate_limiter=reverb.rate_limiters.MinSize(1),
                       max_size=1,
                       max_times_sampled=0,
                       signature=variable_container_signature,
                   ),
               ],
        port=_PORT.value,
    )

  while not os.path.exists(os.path.join(_ROOT_DIR.value, 'training_complete')):
    time.sleep(PROCESS_WAIT_INTERVAL)


def main(_):
  run_reverb_server(_ROOT_DIR.value)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'port',
      'min_table_size_before_sampling',
      'replay_buffer_capacity',
      'algorithm',
      'task_name'
  ])
  app.run(main)
