"""Main binary to launch a stand alone Reverb RB server."""

import os

from absl import app
from absl import flags
from absl import logging
import reverb
import tensorflow as tf
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.specs import tensor_spec
from tf_agents.train import learner
from tf_agents.train.utils import train_utils

_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_CAPACITY = flags.DEFINE_integer(
    'replay_buffer_capacity', 1000000, 'Capacity of the replay buffer table.'
)
_PORT = flags.DEFINE_integer('port', None, 'Port to start the server on.')


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

  # Crete and start the replay buffer and variable container server.
  # TODO(b/159130813): Optionally turn the reverb server pieces into a library.
  server = reverb.Server(
      tables=[
          # Note that the training table and the normalization table are
          # synchronized and contain identical values. Because the collectors
          # keep running, we use FIFO samplers to ensure that the data used
          # for normalization is the same as the data we use for training.
          #
          # The remover does not matter because we clear the table and the end
          # of each global step. We assume that the table is large enough to
          # contain the data collected from one step.
          reverb.Table(  # Replay buffer storing experience for training.
              name='training_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=_REPLAY_BUFFER_CAPACITY.value,
              max_times_sampled=1,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Replay buffer storing experience for normalization.
              name='normalization_table',
              sampler=reverb.selectors.Fifo(),
              remover=reverb.selectors.Fifo(),
              rate_limiter=reverb.rate_limiters.MinSize(1),
              max_size=_REPLAY_BUFFER_CAPACITY.value,
              max_times_sampled=1,
              signature=replay_buffer_signature,
          ),
          reverb.Table(  # Variable container storing policy parameters.
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

  logging.info(
      f'Started Reverb server on port {_PORT.value} with capacity {_REPLAY_BUFFER_CAPACITY.value}')
  server.wait()


def main(_):
  # Add a prefix to our absl logger so we know which collect job this is
  absl_handler = logging.get_absl_handler()
  absl_handler.setFormatter(PrefixedLogFormatter())

  tf.compat.v1.enable_v2_behavior()
  run_reverb_server(_ROOT_DIR.value)


if __name__ == '__main__':
  flags.mark_flags_as_required(['root_dir', 'port'])
  app.run(main)
