import os
import subprocess
import time

from absl import app
from absl import logging
import numpy as np

PROCESS_WAIT_INTERVAL = 10


def create_and_manage_process(command, process_list, env_vars=None):
  if env_vars is None:
    env_vars = os.environ.copy()

  process = subprocess.Popen(
      command,
      env=env_vars,
      text=True,
  )

  process_list.append(process)
  return process


def train():
  mode = os.environ.get('MODE', None)
  job_type = os.environ.get('JOB_TYPE', None)
  if job_type is None:
    raise ValueError('Job type must be set.')
  algorithm = os.environ.get('ALGORITHM', None)
  seed = int(os.environ.get('SEED', -1))
  use_gae = bool(os.environ.get('USE_GAE', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  num_epochs = int(os.environ.get('NUM_EPOCHS', -1))
  replay_buffer_capacity = int(os.environ.get('RB_CAPACITY', -1))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', -1))
  batch_size = int(os.environ.get('BATCH_SIZE', -1))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', -1))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', -1))
  entropy_regularization = float(os.environ.get('ENTROPY_REGULARIZATION', -1))
  exploration_noise_std = float(os.environ.get('EXPLORATION_NOISE_STD', -1))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', -1)
  )
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', -1)
  )
  log_interval = int(os.environ.get('LOG_INTERVAL', -1))
  learning_rate = float(os.environ.get('LEARNING_RATE', -1))
  timesteps_per_actorbatch = int(os.environ.get('TIMESTEPS_PER_ACTORBATCH', -1))
  num_collect_steps_per_actor = int(
      os.environ.get('NUM_COLLECT_STEPS_PER_ACTOR', -1)
  )
  env_name = os.environ.get('ENV_NAME', None)
  netlist_path = os.environ.get('NETLIST_PATH', None)
  init_placement_path = os.environ.get('INIT_PLACEMENT_PATH', None)
  motion_file_path = os.environ.get('MOTION_FILE_PATH', None)
  vocab_port = int(os.environ.get('VOCAB_PORT', '50000'))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', -1))
  num_websites = int(os.environ.get('NUM_WEBSITES', -1))
  debug = bool(os.environ.get('DEBUG', None))
  max_vocab_size = int(os.environ.get('MAX_VOCAB_SIZE', -1))
  embedding_dim = int(os.environ.get('EMBEDDING_DIM', -1))
  latent_dim = int(os.environ.get('LATENT_DIM', -1))
  epsilon_greedy = float(os.environ.get('EPSILON_GREEDY', -1))
  profile_value_dropout = float(os.environ.get('PROFILE_VALUE_DROPOUT', -1))
  std_cell_placer_mode = os.environ.get('STD_CELL_PLACER_MODE', None)

  # Networking params
  num_collect_jobs = int(os.environ.get('NUM_COLLECT_JOBS', 1))
  vocabulary_manager_auth_key = os.environ.get(
      'VOCABULARY_MANAGER_AUTH_KEY', ''
  )
  replay_buffer_server_address = os.environ.get(
      'REPLAY_BUFFER_SERVER_ADDRESS', None
  )
  variable_container_server_address = os.environ.get(
      'VARIABLE_CONTAINER_SERVER_ADDRESS', None
  )
  replay_buffer_server_port = int(
      os.environ.get('REPLAY_BUFFER_SERVER_PORT', -1)
  )
  variable_container_server_port = int(
      os.environ.get('VARIABLE_CONTAINER_SERVER_PORT', -1)
  )
  vocabulary_server_address = os.environ.get('VOCABULARY_SERVER_ADDRESS', None)
  vocabulary_server_port = int(os.environ.get('VOCABULARY_SERVER_PORT', -1))

  logging.info('replay_buffer_server_address: %s', replay_buffer_server_address)
  logging.info('replay_buffer_server_port: %s', replay_buffer_server_port)
  logging.info(
      'variable_container_server_address: %s', variable_container_server_address
  )
  logging.info(
      'variable_container_server_port: %s', variable_container_server_port
  )
  logging.info('batch_size: %s', batch_size)
  logging.info('debug: %s', debug)
  logging.info('entropy_regularization: %s', entropy_regularization)
  logging.info('env_batch_size: %s', env_batch_size)
  logging.info('env_name: %s', env_name)
  logging.info('eval_interval: %s', eval_interval)
  logging.info('learning_rate: %s', learning_rate)
  logging.info('log_interval: %s', log_interval)
  logging.info('num_epochs: %s', num_epochs)
  logging.info('policy_checkpoint_interval: %s', policy_checkpoint_interval)
  logging.info('root_dir: %s', root_dir)
  logging.info('seed: %s', seed)
  logging.info('timesteps_per_actorbatch: %s', timesteps_per_actorbatch)
  logging.info('total_env_steps: %s', total_env_steps)
  logging.info('train_checkpoint_interval: %s', train_checkpoint_interval)

  if env_name == 'CircuitTraining-v0':
    logging.info('netlist_path: %s', netlist_path)
    logging.info('init_placement_path: %s', init_placement_path)
    logging.info('std_cell_placer_mode: %s', std_cell_placer_mode)
  elif env_name == 'QuadrupedLocomotion-v0':
    logging.info('motion_file_path: %s', motion_file_path)
  elif env_name == 'WebNavigation-v0':
    logging.info('vocabulary_server_address: %s', vocabulary_server_address)
    logging.info('vocabulary_server_port: %s', vocabulary_server_port)
    logging.info('difficulty_level: %s', difficulty_level)
    logging.info('num_websites: %s', num_websites)
    logging.info('embedding_dim: %s', embedding_dim)
    logging.info('latent_dim: %s', latent_dim)
    logging.info('epsilon_greedy: %s', epsilon_greedy)
    logging.info('profile_value_dropout: %s', profile_value_dropout)
    logging.info('max_vocab_size: %s', max_vocab_size)
  else:
    raise ValueError(f'Unsupported environment: {env_name}')

  # Check if the selected algorithm is Proximal Policy Optimization (PPO)
  if algorithm in ('ppo',):
    # One step per minibatch. There are `timesteps_per_actorbatch` timesteps
    # per iteration, then multiplied by the number of epochs.
    train_steps_per_iteration = int(
        timesteps_per_actorbatch / batch_size * num_epochs
    )

    # Shuffle buffer just needs to be enough to uncorrelate samples within a
    # single actorbatch
    shuffle_buffer_size = timesteps_per_actorbatch

    # Only a single iteration is performed per call to the learner. We set the
    # `num_samples` argument to `env_batch_size` to ensure that the learner
    # processes all the data collected by the actors in a single call.
    learner_iterations_per_call = 1

    # No need to collect data initially in PPO.
    initial_collect_steps = 0
    num_iterations = np.maximum(
        1, total_env_steps / timesteps_per_actorbatch
    ).astype(int)
  elif algorithm in ('sac', 'ddqn', 'td3', 'ddpg', 'dqn'):
    # We want to exhaust `timesteps_per_actorbatch` samples each iteration
    # roughly.
    learner_iterations_per_call = np.maximum(
        1, timesteps_per_actorbatch / batch_size
    ).astype(int)
    train_steps_per_iteration = learner_iterations_per_call
    shuffle_buffer_size = -1
    initial_collect_steps = num_collect_steps_per_actor
    num_iterations = np.maximum(
        1, total_env_steps / timesteps_per_actorbatch
    ).astype(int)
  else:
    raise ValueError(f'Unsupported algorithm: {algorithm}')
  if train_steps_per_iteration < 1:
    raise ValueError(
        'train_steps_per_iteration must be at least 1, got'
        f' {train_steps_per_iteration}'
    )
  min_table_size_before_sampling = 1
  max_train_steps = train_steps_per_iteration * num_iterations

  policy_checkpoint_interval = np.maximum(
      1,
      np.round(
          policy_checkpoint_interval
          / timesteps_per_actorbatch
          * train_steps_per_iteration
      ).astype(int),
  )
  train_checkpoint_interval = np.maximum(
      1,
      np.round(
          train_checkpoint_interval
          / timesteps_per_actorbatch
          * train_steps_per_iteration
      ).astype(int),
  )
  eval_interval = np.maximum(
      1,
      np.round(
          eval_interval / timesteps_per_actorbatch * train_steps_per_iteration
      ).astype(int),
  )
  log_interval = np.maximum(
      1,
      np.round(
          log_interval / timesteps_per_actorbatch * train_steps_per_iteration
      ).astype(int),
  )

  logging.info('train_steps_per_iteration: %s', train_steps_per_iteration)
  logging.info('num_iterations: %s', num_iterations)
  logging.info('max_train_steps: %s', max_train_steps)
  logging.info(
      'converted policy_checkpoint_interval: %s', policy_checkpoint_interval
  )
  logging.info(
      'converted train_checkpoint_interval: %s', train_checkpoint_interval
  )
  logging.info('converted eval_interval: %s', eval_interval)
  logging.info('converted log_interval: %s', log_interval)
  logging.info('shuffle_buffer_size: %s', shuffle_buffer_size)
  logging.info('random seed: %s', seed)

  all_processes = []
  env_flags = []
  if env_name == 'WebNavigation-v0':
    env_flags.extend([
        f'--env_name={env_name}',
        f'--num_websites={num_websites}',
        f'--difficulty_level={difficulty_level}',
    ])

    if job_type == 'train':
      vocab_manager_command = [
          'python',
          'distributed/vocabulary_manager.py',
          f'--vocabulary_server_port={vocab_port}',
          f'--vocabulary_server_address={vocabulary_server_address}',
          f'--vocabulary_manager_auth_key={vocabulary_manager_auth_key}',
          f'--max_vocab_size={max_vocab_size}',
          f'--verbosity={logging.get_verbosity()}',
      ]
      logging.info('Command for vocab manager: %s', vocab_manager_command)
      vocab_server_process = create_and_manage_process(
          vocab_manager_command, all_processes
      )

      if vocab_server_process.poll() is not None:
        raise ValueError('Vocabulary manager server failed to start.')
      else:
        logging.info('Successfully launched vocab manager server.')
  elif env_name == 'QuadrupedLocomotion-v0':
    env_flags.extend(
        [f'--env_name={env_name}', f'--motion_file_path={motion_file_path}']
    )
  elif env_name == 'CircuitTraining-v0':
    env_flags.extend([
        '--netlist_index=0',
        f'--std_cell_placer_mode={std_cell_placer_mode}',
        f'--netlist_file={netlist_path}',
        f'--init_placement={init_placement_path}',
    ])
  else:
    raise ValueError(f'Unsupported environment: {env_name}')

  if job_type == 'collect':
    # Launch collect jobs with domain-specific configurations
    if env_name == 'CircuitTraining-v0':
      collect_job_commands = [
          [
              'python',
              '-m',
              'distributed.circuit_training.learning.ppo_collect',
              f'--root_dir={root_dir}',
              f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
              f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
              f'--task_id={i}',
              f'--debug={debug}',
              f'--global_seed={seed}',
              f'--max_train_steps={max_train_steps}',
              f'--verbosity={"1" if i == 0 else "-1"}',
              f'--summary_interval={log_interval}',
              f'--initial_collect_steps={initial_collect_steps}',
              f'--sequence_length={num_collect_steps_per_actor}',
              f'--initial_collect_steps={initial_collect_steps}',
          ]
          + env_flags
          for i in range(num_collect_jobs)
      ]
    else:
      collect_job_commands = [
          [
              'python',
              'distributed/collect.py',
              f'--algorithm={algorithm}',
              f'--root_dir={root_dir}',
              f'--sequence_length={num_collect_steps_per_actor}',
              f'--summary_interval={log_interval}',
              f'--env_batch_size={env_batch_size}',
              f'--initial_collect_steps={initial_collect_steps}',
              f'--max_train_steps={max_train_steps}',
              f'--debug={debug}',
              f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
              f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
              f'--task={i}',
              f'--vocabulary_manager_auth_key={vocabulary_manager_auth_key}',
              f'--vocabulary_server_address={vocabulary_server_address}',
              f'--vocabulary_server_port={vocabulary_server_port}',
              f'--verbosity={"1" if i == 0 else "-1"}',
          ]
          + env_flags
          for i in range(num_collect_jobs)
      ]

    # Display one of the commands
    logging.info(' '.join(collect_job_commands[0]))

    collect_jobs = []
    for command in collect_job_commands:
      collect_jobs.append(create_and_manage_process(command, all_processes))
    logging.info('Successfully launched collect jobs.')

    # Recall that root dir is modified for collect jobs to separate
    # system metrics from the train job
    while not os.path.exists(
        os.path.join(root_dir, '../../', 'training_complete')
    ):
      time.sleep(PROCESS_WAIT_INTERVAL)

  elif job_type == 'train':
    reverb_command = [
        'python',
        'distributed/reverb_server.py',
        f'--environment_name={env_name}',
        f'--port={replay_buffer_server_port}',
        '--num_netlists=1',
        f'--root_dir={root_dir}',
        f'--replay_buffer_capacity={replay_buffer_capacity}',
        f'--algorithm={algorithm}',
        f'--min_table_size_before_sampling={min_table_size_before_sampling}',
        f'--verbosity={logging.get_verbosity()}',
    ]
    logging.info(' '.join(reverb_command))
    create_and_manage_process(reverb_command, all_processes)
    logging.info('Successfully launched reverb server.')

    if env_name == 'CircuitTraining-v0':
      train_job_command = [
          'python',
          '-m',
          'distributed.circuit_training.learning.train_ppo',
          f'--entropy_regularization={entropy_regularization}',
          f'--use_gae={use_gae}',
          f'--netlist_file={netlist_path}',
          f'--init_placement={init_placement_path}',
          '--std_cell_placer_mode=dreamplace',
          f'--root_dir={root_dir}',
          f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
          f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
          f'--sequence_length={num_collect_steps_per_actor}',
          f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
          f'--std_cell_placer_mode={std_cell_placer_mode}',
          f'--summary_interval={log_interval}',
          '--use_gpu=True',
          f'--global_seed={seed}',
          f'--num_epochs={num_epochs}',
          f'--batch_size={batch_size}',
          f'--shuffle_buffer_size={shuffle_buffer_size}',
          f'--algorithm={algorithm}',
          f'--debug={debug}',
          f'--epsilon_greedy={epsilon_greedy}',
          f'--train_checkpoint_interval={train_checkpoint_interval}',
          f'--policy_checkpoint_interval={policy_checkpoint_interval}',
          f'--env_batch_size={env_batch_size}',
          f'--learning_rate={learning_rate}',
          f'--summary_interval={log_interval}',
          f'--algorithm={algorithm}',
          f'--debug={debug}',
          f'--max_train_steps={max_train_steps}',
          f'--learner_iterations_per_call={learner_iterations_per_call}',
          # Only use these if you have a pretrained policy to bootstrap from
          # f'--policy_saved_model_dir={root_dir}/policies/policy',
          # f'--policy_checkpoint_dir={root_dir}/policies/checkpoints',
      ]

    else:
      train_job_command = [
          'python',
          '-m',
          'distributed.train',
          f'--entropy_regularization={entropy_regularization}',
          f'--exploration_noise_std={exploration_noise_std}',
          f'--num_epochs={num_epochs}',
          f'--batch_size={batch_size}',
          f'--shuffle_buffer_size={shuffle_buffer_size}',
          f'--algorithm={algorithm}',
          f'--debug={debug}',
          f'--learner_iterations_per_call={learner_iterations_per_call}',
          f'--sequence_length={num_collect_steps_per_actor}',
          f'--policy_checkpoint_interval={policy_checkpoint_interval}',
          f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
          f'--root_dir={root_dir}',
          f'--train_checkpoint_interval={train_checkpoint_interval}',
          f'--max_train_steps={max_train_steps}',
          f'--env_batch_size={env_batch_size}',
          f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
          f'--learning_rate={learning_rate}',
          f'--log_interval={log_interval}',
          f'--seed={seed}',
          f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
          '--use_gpu=True',
          f'--use_gae={use_gae}',
          '--use_tpu=False',
          f'--embedding_dim={embedding_dim}',
          f'--latent_dim={latent_dim}',
          f'--epsilon_greedy={epsilon_greedy}',
          f'--profile_value_dropout={profile_value_dropout}',
          f'--max_vocab_size={max_vocab_size}',
          f'--num_websites={num_websites}',
          f'--difficulty_level={difficulty_level}',
          f'--motion_file_path={motion_file_path}',
          f'--verbosity={logging.get_verbosity()}',
      ] + env_flags

    # Display the command
    logging.info(' '.join(train_job_command))
    create_and_manage_process(train_job_command, all_processes)
    logging.info('Successfully launched train job.')

    while not os.path.exists(os.path.join(root_dir, 'training_complete')):
      time.sleep(PROCESS_WAIT_INTERVAL)

  logging.info('Training complete.')
  for process in all_processes:
    process.kill()
  logging.info('All processes killed.')


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
