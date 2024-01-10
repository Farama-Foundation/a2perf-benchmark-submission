"""Main binary to launch a stand alone Reverb RB server."""

import os

import reverb
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer(
    'replay_buffer_capacity', None, 'Capacity of the replay buffer table.'
)
_PORT = flags.DEFINE_integer('port', None, 'Port to start the server on.')

_SAMPLES_PER_INSERT = flags.DEFINE_integer(
    'samples_per_insert', None, 'Number of samples to insert per insert.'
)
_MIN_TABLE_SIZE_BEFORE_SAMPLING = flags.DEFINE_integer(
    'min_table_size_before_sampling',
    None,
    'Minimum number of items in the table before sampling.',
)
_SAMPLES_PER_INSERT_TOLERANCE_RATIO = flags.DEFINE_float(
    'samples_per_insert_tolerance_ratio',
    None,
    'Tolerance ratio for the samples per insert.',
)


class PrefixedLogFormatter(logging.PythonFormatter):
  def format(self, record):
    original = super(PrefixedLogFormatter, self).format(record)
    return f'Reverb Server: {original}'


def run_reverb_server(root_dir):
  """Start the server after the initial policy becomes available."""
  # Wait for the collect policy to become available, then load it.
  collect_policy_dir = os.path.join(
      root_dir,
      learner.POLICY_SAVED_MODEL_DIR,
      learner.COLLECT_POLICY_SAVED_MODEL_DIR,
  )
  collect_policy = train_utils.wait_for_policy(
      collect_policy_dir, load_specs_from_pbtxt=True
  )

  logging.info('Loaded collect policy from %s', collect_policy_dir)

  # Create the signature for the variable container holding the policy weights.
  train_step = train_utils.create_train_step()
  variables = {
      reverb_variable_container.POLICY_KEY: collect_policy.variables(),
      reverb_variable_container.TRAIN_STEP_KEY: train_step,
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

  # Create and start the replay buffer and variable container server.
  server = reverb.Server(
      tables=[
          reverb.Table(  # Replay buffer storing experience.
              name=reverb_replay_buffer.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=_REPLAY_BUFFER_CAPACITY.value,
              max_times_sampled=0,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
              name=reverb_variable_container.DEFAULT_TABLE,
              sampler=reverb.selectors.Uniform(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=1,
              max_times_sampled=0,
              signature=variable_container_signature,
          ),
      ],
      port=_PORT.value,
  )

  logging.info(
      f'Started Reverb server on port {_PORT.value} with capacity {_REPLAY_BUFFER_CAPACITY.value}')
  server.wait()


def main(_):
  # tf.compat.v1.enable_v2_behavior()

  # Add a prefix to our absl logger so we know which collect job this is
  absl_handler = logging.get_absl_handler()
  absl_handler.setFormatter(PrefixedLogFormatter())

  run_reverb_server(_ROOT_DIR.value)


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'port',
                                'replay_buffer_capacity'
                                # replay buffer size important for off-policy learning
                                ])
  app.run(main)
