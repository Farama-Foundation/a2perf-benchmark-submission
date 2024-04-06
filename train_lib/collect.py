"""Sample collection Job using a variable container for policy updates."""

import functools
from multiprocessing.managers import BaseManager
import os
import time
from typing import Optional
from typing import Text

from a2perf.domains import circuit_training
from a2perf.domains import quadruped_locomotion
from a2perf.domains.utils import suite_gym
from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node
from absl import app
from absl import flags
from absl import logging
import gin
import reverb
import tensorflow as tf
import tf_agents
from tf_agents.environments import wrappers
from tf_agents.experimental.distributed import reverb_variable_container
from tf_agents.metrics import py_metrics
from tf_agents.policies import py_epsilon_greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import tf_policy
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

_DEBUG = flags.DEFINE_bool('debug', False, 'Debug mode.')
_GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'Paths to the gin-config files.'
)
_INITIAL_COLLECT_STEPS = flags.DEFINE_integer(
    'initial_collect_steps', None, 'Initial number of steps to collect.'
)
_ALGORITHM = flags.DEFINE_string(
    'algorithm',
    None,
    'Algorithm to use. Must be one of "ppo" or "sac".',
)
_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', [], 'Gin binding parameters.'
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
_NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations', None, 'Number of iterations.'
)
_ROOT_DIR = flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
_REPLAY_BUFFER_SERVER_ADDRESS = flags.DEFINE_string(
    'replay_buffer_server_address', None, 'Replay buffer server address.'
)
_SUMMARY_INTERVAL = flags.DEFINE_integer(
    'summary_interval', None, 'Interval for writing summaries.'
)

_VARIABLE_CONTAINER_SERVER_ADDRESS = flags.DEFINE_string(
    'variable_container_server_address',
    None,
    'Variable container server address.',
)

_TASK = flags.DEFINE_integer(
    'task', None, 'Identifier of the collect task. Must be unique in a job.'
)
_GLOBAL_SEED = flags.DEFINE_integer(
    'seed',
    111,
    'Used in env and weight initialization, does not impact action sampling.',
)
_NETLIST_INDEX = flags.DEFINE_integer(
    'netlist_index', 0, 'Index of the netlist in the agent policy model.'
)

_POLICY_SAVED_MODEL_DIR = flags.DEFINE_string(
    'policy_saved_model_dir', None, 'If set, load the pretrained policy model.'
)
_POLICY_CHECKPOINT_DIR = flags.DEFINE_string(
    'policy_checkpoint_dir', None, 'If set, load the pretrained policy model.'
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
_NUM_REPLICAS = flags.DEFINE_integer(
    'num_replicas', None, 'Number of replicas.'
)
_EMBEDDING_DIM = flags.DEFINE_integer(
    'embedding_dim', None, 'Embedding dimension of the LSTM.'
)

_EPSILON_GREEDY = flags.DEFINE_float(
    'epsilon_greedy', None, 'Epsilon greedy value.'
)
_EXPLORATION_NOISE_STD = flags.DEFINE_float(
    'exploration_noise_std', None, 'Exploration noise std.'
)

_MAX_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'max_sequence_length',
    None,
    'Length of sequences to sample from the replay buffer.',
)
_NUM_WEBSITES = flags.DEFINE_integer(
    'num_websites', None, 'Number of websites to use.'
)
_DIFFICULTY_LEVEL = flags.DEFINE_integer(
    'difficulty_level', None, 'Difficulty of the task.'
)
_ENV_NAME = flags.DEFINE_string('env_name', None, 'Name of the environment')
_LOG_INTERVAL = flags.DEFINE_integer('log_interval', None, 'Log interval.')

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

_BATCH_SIZE = flags.DEFINE_integer('batch_size', None, 'Batch size.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', None, 'Number of epochs.')

_ENTROPY_REGULARIZATION = flags.DEFINE_float(
    'entropy_regularization', None, 'Entropy regularization.'
)
_MAX_TRAIN_STEP = flags.DEFINE_integer(
    'max_train_steps', None, 'Number of iterations.'
)
_VOCABULARY_MANAGER_AUTH_KEY = flags.DEFINE_string(
    'vocabulary_manager_auth_key',
    None,
    'Authentication key for the manager server.',
)
_VOCABULARY_SERVER_ADDRESS = flags.DEFINE_string(
    'vocabulary_server_address', None, 'Address for the vocabulary manager.'
)
_VOCABULARY_SERVER_PORT = flags.DEFINE_integer(
    'vocabulary_server_port', None, 'Vocabulary server port.'
)

# If we have not collected in this many seconds, run another episode. This
# prevents the training loop from being stuck when using a collector
# max_episodes_per_model limit, since various workers (including the Reverb
# server) can be preempted.
COLLECT_AT_LEAST_EVERY_SECONDS = 10 * 60
ACTOR_COLLECT_METRICS_BUFFER_SIZE = 10

# Maximum number of retries for connecting to the vocabulary server.
MAX_RETRIES = 8640  # 24 hours
RETRY_DELAY = 10

EPSILON_DECAY_END_VALUE = 1e-2


def mask_circuit_training_actions(circuit_env, observation):
  mask = circuit_env.unwrapped._get_mask()
  return observation, mask


def collect_off_policy(
    collect_env: tf_agents.environments.TFPyEnvironment,
    collect_policy: tf_policy.TFPolicy,
    random_policy: tf_policy.TFPolicy,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    root_dir: str,
    task: int,
    max_train_step: int,
    summary_interval: int,
    initial_collect_steps: int,
) -> None:
  # We run collect jobs in replicas when using kubernetes,
  # so check if JOB_COMPLETION_INDEX is set.
  # If it is, we make sure to only record summaries from task 0, replica 0.
  summary_dir = None
  actor_collect_metrics = actor.collect_metrics(
      ACTOR_COLLECT_METRICS_BUFFER_SIZE
  )
  if 'JOB_COMPLETION_INDEX' in os.environ:
    job_completion_index = int(os.environ['JOB_COMPLETION_INDEX'])
    if job_completion_index == 0:
      summary_dir = os.path.join(root_dir, 'summaries', str(task))

  else:
    summary_dir = os.path.join(root_dir, 'summaries', str(task))

  logging.info('Summary dir: %s', summary_dir)

  # Create the variable container.
  train_step = train_utils.create_train_step()
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

  reverb_client = reverb.Client(replay_buffer_server_address)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
      reverb_client,
      table_name='training_table_0',
      sequence_length=2,
      stride_length=1,
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

  env_step_metric = py_metrics.EnvironmentSteps()
  collect_actor = actor.Actor(
      collect_env,
      collect_policy,
      train_step,
      steps_per_run=1,
      metrics=actor_collect_metrics,
      summary_dir=summary_dir,
      summary_interval=summary_interval,
      observers=[rb_observer, env_step_metric],
  )

  # Run the experience collection loop.
  training_done_file = os.path.join(root_dir, '../../', 'training_complete')
  prev_num_steps_collected = 0
  while train_step < max_train_step and not os.path.exists(training_done_file):
    start_time = time.time()
    collect_actor.run()
    end_time = time.time()
    variable_container.update(variables)
    logging.info(
        'Collecting with policy at step: %d out of %d',
        train_step.numpy(),
        max_train_step,
    )
    logging.info('\tCollected %d steps', env_step_metric.result())
    logging.info(
        '\tCollected %d steps this iteration',
        env_step_metric.result() - prev_num_steps_collected,
    )
    logging.info('\tCollection took %.3f seconds', end_time - start_time)
    prev_num_steps_collected = env_step_metric.result()

    with (
        collect_actor.summary_writer.as_default(),
        tf.summary.record_if(
            lambda: tf.math.equal(train_step % summary_interval, 0)
        ),
    ):
      if getattr(collect_policy, '_get_epsilon', None) is not None:
        tf.summary.scalar(
            'Metrics/EpsilonGreedy',
            collect_policy._get_epsilon(),
            step=train_step,
        )

  logging.info('Done collecting.')


@gin.configurable
def collect_on_policy(
    collect_env: tf_agents.environments.TFPyEnvironment,
    collect_policy: py_tf_eager_policy.PyTFEagerPolicyBase,
    replay_buffer_server_address: Text,
    variable_container_server_address: Text,
    root_dir: str,
    task: int,
    max_train_step: int,
    summary_interval: int,
    sequence_length: int,
    max_timesteps_per_model: Optional[int] = None,
    **kwargs,
) -> None:
  """Collects experience using a policy updated after every episode."""
  summary_dir = None
  if 'JOB_COMPLETION_INDEX' in os.environ:
    job_completion_index = int(os.environ['JOB_COMPLETION_INDEX'])
    if job_completion_index == 0:
      summary_dir = os.path.join(root_dir, 'summaries', str(task))

  else:
    summary_dir = os.path.join(root_dir, 'summaries', str(task))

  # Create the variable container.
  train_step = train_utils.create_train_step()
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
          table_name=f'training_table_{0}',
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
      metrics=actor.collect_metrics(ACTOR_COLLECT_METRICS_BUFFER_SIZE),
      summary_dir=summary_dir,
      summary_interval=summary_interval,
      observers=observers,
  )

  training_done_file = os.path.join(root_dir, '../../', 'training_complete')
  # Run the experience collection loop.
  model_to_num_timesteps = {}
  last_collection_ts = 0
  prev_num_steps_collected = 0
  while train_step < max_train_step and not os.path.exists(training_done_file):
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
    logging.info('\tMax train step: %d', max_train_step)
    logging.info('\tCollected %d steps', env_step_metric.result())
    logging.info(
        '\tCollected %d steps this iteration',
        env_step_metric.result() - prev_num_steps_collected,
    )
    prev_num_steps_collected = env_step_metric.result()


def run_collect(
    root_dir: str,
    environment_name: str,
    replay_buffer_server_address: str,
    variable_container_server_address: str,
    task: int,
    max_train_step: int,
    suite_load_fn: callable,
    algorithm: str,
    summary_interval: int,
    sequence_length: int,
    initial_collect_steps: int,
) -> None:
  """Wait for the collect policy to be ready and run collect job."""
  collect_env = suite_load_fn(environment_name)
  root_policy_path = os.path.join(
      root_dir,
      '../../',  # two levels because collect/<hostname>/ is the root_dir
      learner.POLICY_SAVED_MODEL_DIR,
  )
  if algorithm in ('sac', 'ddqn', 'td3', 'dqn', 'ddpg'):
    observation_and_action_constraint_splitter_fn = None
    if environment_name == 'CircuitTraining-v0':
      observation_and_action_constraint_splitter_fn = functools.partial(
          mask_circuit_training_actions, collect_env
      )

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(),
        collect_env.action_spec(),
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter_fn,
    )

    if algorithm in ('dqn', 'ddqn'):
      # The TF Agent creates a collect policy that handles epsilon greedy,
      # but some environments use a mask on valid/invalid actions. Loading the raw
      # policy allows us to apply the mask ourselves.
      greedy_policy_dir = os.path.join(
          root_policy_path, learner.GREEDY_POLICY_SAVED_MODEL_DIR
      )
      greedy_policy = train_utils.wait_for_policy(
          greedy_policy_dir, load_specs_from_pbtxt=True
      )
      logging.info('Loaded greedy policy from %s', greedy_policy_dir)

      # Now wrap the policy in an epsilon greedy policy.
      epsilon_greedy_policy_obj = py_epsilon_greedy_policy.EpsilonGreedyPolicy(
          greedy_policy=greedy_policy,
          random_policy=random_policy,
          epsilon=_EPSILON_GREEDY.value,
          random_seed=_GLOBAL_SEED.value,
          epsilon_decay_end_value=EPSILON_DECAY_END_VALUE,
          # Adjust the decay end count as needed. Set to approximately
          # the total number of steps to be collected by this collect job.
          epsilon_decay_end_count=_NUM_ITERATIONS.value,
      )
      epsilon_greedy_policy_obj.variables = greedy_policy.variables
      policy = epsilon_greedy_policy_obj
    elif algorithm in ('sac', 'td3', 'ddpg'):
      collect_policy_dir = os.path.join(
          root_policy_path, learner.COLLECT_POLICY_SAVED_MODEL_DIR
      )
      policy = train_utils.wait_for_policy(
          collect_policy_dir, load_specs_from_pbtxt=True
      )
      logging.info('Loaded collect policy from %s', collect_policy_dir)
  else:
    collect_policy_dir = os.path.join(
        root_policy_path, learner.COLLECT_POLICY_SAVED_MODEL_DIR
    )

    logging.info('Looking for collect policy in %s', collect_policy_dir)

    policy = train_utils.wait_for_policy(
        collect_policy_dir, load_specs_from_pbtxt=True
    )
    random_policy = None
    logging.info('Loaded collect policy from %s', collect_policy_dir)

  if algorithm in ('sac', 'ddqn', 'td3', 'dqn', 'ddpg'):
    collect_off_policy(
        collect_env=collect_env,
        collect_policy=policy,
        initial_collect_steps=initial_collect_steps,
        max_train_step=max_train_step,
        random_policy=random_policy,
        replay_buffer_server_address=replay_buffer_server_address,
        root_dir=root_dir,
        summary_interval=summary_interval,
        task=task,
        variable_container_server_address=variable_container_server_address,
    )
  else:
    collect_on_policy(
        collect_env=collect_env,
        collect_policy=policy,
        replay_buffer_server_address=replay_buffer_server_address,
        variable_container_server_address=variable_container_server_address,
        root_dir=root_dir,
        task=task,
        max_train_step=max_train_step,
        summary_interval=summary_interval,
        sequence_length=sequence_length,
    )


def setup_web_navigation_env_for_collect():
  # Connect to the global vocabulary. This vocabulary is shared across all collect jobs.
  class VocabularyManager(BaseManager):
    pass

  VocabularyManager.register('get_shared_dict')
  VocabularyManager.register('get_shared_lock')

  # Initialize the manager outside of the loop
  manager = VocabularyManager(
      address=(
          _VOCABULARY_SERVER_ADDRESS.value,
          _VOCABULARY_SERVER_PORT.value,
      ),
      authkey=_VOCABULARY_MANAGER_AUTH_KEY.value.encode(),
  )

  connected = False
  for attempt in range(MAX_RETRIES):
    try:
      manager.connect()
      connected = True
      print(
          'Successfully connected to the vocab server on attempt'
          f' {attempt + 1}.'
      )
      break
    except ConnectionRefusedError:
      if attempt < MAX_RETRIES - 1:
        print(
            f'Attempt {attempt + 1} failed to connect to the vocab server. '
            f'Retrying in {RETRY_DELAY} seconds...'
        )
        time.sleep(RETRY_DELAY)
      else:
        print('Failed to connect to the manager server.')

  if not connected:
    raise ConnectionRefusedError(
        'Unable to connect to the vocabulary server after maximum retries.'
    )

  manager.connect()

  shared_dict = manager.get_shared_dict()
  shared_lock = manager.get_shared_lock()

  global_vocabulary = vocabulary_node.LockedMultiprocessingVocabulary(
      shared_lock=shared_lock,
      shared_dict=shared_dict,
  )

  default_gym_kwargs = dict(
      global_vocabulary=global_vocabulary,
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
      suite_gym.load,
      gym_kwargs=default_gym_kwargs,
      env_wrappers=[wrappers.ActionClipWrapper],
  )
  return suite_load_function


def setup_quadruped_locomotion_env_for_collect():
  default_gym_kwargs = dict(
      motion_files=[_MOTION_FILE_PATH.value],
      num_parallel_envs=_ENV_BATCH_SIZE.value,
  )
  suite_load_function = functools.partial(
      suite_gym.load,
      gym_kwargs=default_gym_kwargs,
      env_wrappers=[wrappers.ActionClipWrapper],
  )
  return suite_load_function


def setup_circuit_training_env_for_collect():
  gym_kwargs = dict(
      netlist_file=_NETLIST_FILE.value,
      init_placement=_INIT_PLACEMENT.value,
      global_seed=_GLOBAL_SEED.value,
      std_cell_placer_mode=_STD_CELL_PLACER_MODE.value,
      netlist_index=_NETLIST_INDEX.value,
  )
  suite_load_function = functools.partial(
      suite_gym.load,
      gym_kwargs=gym_kwargs,
      env_wrappers=[wrappers.ActionClipWrapper],
  )

  return suite_load_function


def setup_env_for_collect():
  if _ENV_NAME.value == 'QuadrupedLocomotion-v0':
    return setup_quadruped_locomotion_env_for_collect()
  elif _ENV_NAME.value == 'WebNavigation-v0':
    return setup_web_navigation_env_for_collect()
  elif _ENV_NAME.value == 'CircuitTraining-v0':
    return setup_circuit_training_env_for_collect()
  else:
    raise ValueError(f'Unknown environment: {_ENV_NAME.value}')


def main(_):
  if _DEBUG.value:
    tf.config.experimental_run_functions_eagerly(True)

  gin.parse_config_files_and_bindings(
      _GIN_FILE.value, _GIN_BINDINGS.value, finalize_config=False
  )

  suite_load_function = setup_env_for_collect()

  run_collect(
      root_dir=_ROOT_DIR.value,
      environment_name=_ENV_NAME.value,
      replay_buffer_server_address=_REPLAY_BUFFER_SERVER_ADDRESS.value,
      variable_container_server_address=_VARIABLE_CONTAINER_SERVER_ADDRESS.value,
      task=_TASK.value,
      algorithm=_ALGORITHM.value,
      summary_interval=_SUMMARY_INTERVAL.value,
      sequence_length=_MAX_SEQUENCE_LENGTH.value,
      suite_load_fn=suite_load_function,
      max_train_step=_MAX_TRAIN_STEP.value,
      initial_collect_steps=_INITIAL_COLLECT_STEPS.value,
  )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'root_dir',
      'env_name',
      'replay_buffer_server_address',
      'variable_container_server_address',
      'vocabulary_server_address',
      'vocabulary_server_port',
      'vocabulary_manager_auth_key',
      'max_sequence_length',
      'env_batch_size',
      'task',
      'summary_interval',
      'max_train_steps',
      'initial_collect_steps',
      'algorithm',
      'num_replicas',
  ])
  multiprocessing.handle_main(functools.partial(app.run, main))
