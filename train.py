import os
import subprocess
import threading
import time

from absl import app
from absl import logging
import numpy as np


PROCESS_WAIT_INTERVAL = 120  # 2 minutes


def print_subprocess_output(process):
  for line in iter(process.stdout.readline, b''):
    print(line.decode(), end='')


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
  env_name = os.environ.get('ENV_NAME', None)
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
  adjusted_timesteps_per_actorbatch = np.maximum(
      1, timesteps_per_actorbatch // env_batch_size
  )

  # Networking params
  num_collect_machines = int(os.environ.get('NUM_COLLECT_MACHINES', 1))
  auth_key = os.environ.get('AUTH_KEY', 'secretkey')
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

  print(f'replay_buffer_server_address: {replay_buffer_server_address}')
  print(f'replay_buffer_server_port: {replay_buffer_server_port}')
  print(
      f'variable_container_server_address: {variable_container_server_address}'
  )
  print(f'variable_container_server_port: {variable_container_server_port}')

  print(f'batch_size: {batch_size}')
  print(f'debug: {debug}')
  print(f'entropy_regularization: {entropy_regularization}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'env_name: {env_name}')
  print(f'eval_interval: {eval_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'log_interval: {log_interval}')
  print(f'num_epochs: {num_epochs}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'root_dir: {root_dir}')
  print(f'seed: {seed}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')

  if env_name == 'QuadrupedLocomotion-v0':
    print(f'motion_file_path: {motion_file_path}')
  elif env_name == 'WebNavigation-v0':
    print(f'vocabulary_server_address: {vocabulary_server_address}')
    print(f'vocabulary_server_port: {vocabulary_server_port}')
    print(f'difficulty_level: {difficulty_level}')
    print(f'num_websites: {num_websites}')
    print(f'embedding_dim: {embedding_dim}')
    print(f'latent_dim: {latent_dim}')
    print(f'epsilon_greedy: {epsilon_greedy}')
    print(f'profile_value_dropout: {profile_value_dropout}')
    print(f'max_vocab_size: {max_vocab_size}')

  # Check if the selected algorithm is Proximal Policy Optimization (PPO)
  if algorithm == 'ppo':
    # One step per minibatch. There are `timesteps_per_actorbatch` timesteps
    # per iteration, then multiplied by the number of epochs.
    train_steps_per_iteration = (
        timesteps_per_actorbatch // batch_size * num_epochs
    )

    # Shuffle buffer just needs to be enough to uncorrelate samples within a
    # single sequence.
    shuffle_buffer_size = adjusted_timesteps_per_actorbatch

    # Only a single iteration is performed per call to the learner. We set the
    # `num_samples` argument to `env_batch_size` to ensure that the learner
    # processes all the data collected by the actors in a single call.
    learner_iterations_per_call = 1

    # No need to collect data initially in PPO.
    initial_collect_steps = 0

    # We want to exhaust `timesteps_per_actorbatch` samples each iteration roughly.
    num_iterations = np.maximum(1, total_env_steps // timesteps_per_actorbatch)
  else:
    # We want to exhaust `timesteps_per_actorbatch` samples each iteration
    # roughly.
    learner_iterations_per_call = np.maximum(
        1, timesteps_per_actorbatch // batch_size
    )
    train_steps_per_iteration = learner_iterations_per_call
    shuffle_buffer_size = -1
    initial_collect_steps = adjusted_timesteps_per_actorbatch

    num_iterations = np.maximum(1, total_env_steps // timesteps_per_actorbatch)

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

  print(f'train_steps_per_iteration: {train_steps_per_iteration}')
  print(f'num_iterations: {num_iterations}')
  print(f'max_train_steps: {max_train_steps}')
  print(f'converted policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'converted train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'converted eval_interval: {eval_interval}')
  print(f'converted log_interval: {log_interval}')
  print(f'shuffle_buffer_size: {shuffle_buffer_size}')
  print(f'random seed: {seed}')

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
          f'--port={vocab_port}',
          f'--auth_key={auth_key}',
          f'--max_vocab_size={max_vocab_size}',
          f'--verbosity={logging.get_verbosity()}',
      ]
      vocab_manager_process = subprocess.Popen(
          vocab_manager_command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=os.environ.copy(),
      )
      all_processes.append(vocab_manager_process)
      threading.Thread(
          target=print_subprocess_output, args=(vocab_manager_process,)
      ).start()
      logging.info('Successfully launched vocab manager server.')
  elif env_name == 'QuadrupedLocomotion-v0':
    env_flags.extend(
        [f'--env_name={env_name}', f'--motion_file_path={motion_file_path}']
    )

  if job_type == 'collect':
    # Launch collect jobs with domain-specific configurations
    # Note: each collect script runs a single environment, so we adjust
    # the number of jobs started based on the number of machines
    # take the ceiling to ensure we have enough jobs to cover the
    # requested number of environments
    num_collect_jobs = np.ceil(env_batch_size / num_collect_machines).astype(
        int
    )

    collect_job_commands = [
        [
            'python',
            'distributed/collect.py',
            f'--algorithm={algorithm}',
            f'--root_dir={root_dir}',
            f'--sequence_length={adjusted_timesteps_per_actorbatch}',
            f'--summary_interval={log_interval}',
            f'--env_batch_size={env_batch_size}',
            f'--initial_collect_steps={initial_collect_steps}',
            f'--max_train_steps={max_train_steps}',
            f'--debug={debug}',
            f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
            f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
            f'--task={i}',
            f'--auth_key={auth_key}',
            f'--vocabulary_server_hostname={vocabulary_server_address}',
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
      process = subprocess.Popen(
          command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=os.environ.copy(),
      )
      all_processes.append(process)
      collect_jobs.append(process)
    threading.Thread(target=print_subprocess_output, args=(collect_jobs[0],)).start(
    )
    logging.info('Successfully launched collect jobs.')

    while True:
      try:
        for process in collect_jobs:
          process.wait(timeout=PROCESS_WAIT_INTERVAL)
        break
      except subprocess.TimeoutExpired:
        logging.info('Collect jobs still running.')
        continue
    logging.info('Collect jobs finished.')

  elif job_type == 'train':
    reverb_command = [
        'python',
        'distributed/reverb_server.py',
        f'--port={replay_buffer_server_port}',
        f'--root_dir={root_dir}',
        f'--replay_buffer_capacity={replay_buffer_capacity}',
        f'--algorithm={algorithm}',
        f'--min_table_size_before_sampling={min_table_size_before_sampling}',
        f'--verbosity={logging.get_verbosity()}',
    ]
    logging.info(' '.join(reverb_command))

    reverb_process = subprocess.Popen(
        reverb_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    all_processes.append(reverb_process)
    threading.Thread(
        target=print_subprocess_output, args=(reverb_process,)
    ).start()
    logging.info('Successfully launched reverb server.')

    train_job_command = [
        'python',
        'distributed/train.py',
        f'--entropy_regularization={entropy_regularization}',
        f'--exploration_noise_std={exploration_noise_std}',
        f'--num_epochs={num_epochs}',
        f'--batch_size={batch_size}',
        f'--shuffle_buffer_size={shuffle_buffer_size}',
        f'--algorithm={algorithm}',
        f'--debug={debug}',
        f'--learner_iterations_per_call={learner_iterations_per_call}',
        f'--sequence_length={adjusted_timesteps_per_actorbatch}',
        f'--policy_checkpoint_interval={policy_checkpoint_interval}',
        f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
        f'--root_dir={root_dir}',
        f'--train_checkpoint_interval={train_checkpoint_interval}',
        f'--max_train_steps={max_train_steps}',
        f'--env_batch_size={env_batch_size}',
        f'--learning_rate={learning_rate}',
        f'--log_interval={log_interval}',
        f'--seed={seed}',
        f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
        f'--use_gpu=True',
        f'--use_gae={use_gae}',
        f'--use_tpu=False',
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

    train_job = subprocess.Popen(
        train_job_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    all_processes.append(train_job)
    threading.Thread(target=print_subprocess_output, args=(train_job,)).start()
    logging.info('Successfully launched train job.')

    while True:
      try:
        train_job.wait(timeout=PROCESS_WAIT_INTERVAL)
        break
      except subprocess.TimeoutExpired:
        logging.info('Train job still running.')
        continue
    logging.info('Train job finished.')

    # Wait before killing the reverb server so the collect jobs
    # can read the final train step.
    time.sleep(PROCESS_WAIT_INTERVAL)

  for process in all_processes:
    process.kill()
    try:
      process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      logging.info('Process killed unsuccessfully.')
      return


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
