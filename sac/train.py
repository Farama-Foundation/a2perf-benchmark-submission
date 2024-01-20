import os
import subprocess
import threading

from absl import app
from absl import logging
import numpy as np
import tensorflow as tf


def print_subprocess_output(process):
  for line in iter(process.stdout.readline, b''):
    print(line.decode(), end='')


def train():
  mode = os.environ.get('MODE', None)
  job_type = os.environ.get('JOB_TYPE', None)
  if job_type is None:
    raise ValueError('Job type must be set.')
  seed = int(os.environ.get('SEED', -1))
  root_dir = os.environ.get('ROOT_DIR', None)
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', -1))
  batch_size = int(os.environ.get('BATCH_SIZE', -1))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', -1))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', -1))
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
  rb_capacity = int(os.environ.get('RB_CAPACITY', -1))
  embedding_dim = int(os.environ.get('EMBEDDING_DIM', -1))
  latent_dim = int(os.environ.get('LATENT_DIM', -1))
  epsilon_greedy = float(os.environ.get('EPSILON_GREEDY', -1))
  profile_value_dropout = float(os.environ.get('PROFILE_VALUE_DROPOUT', -1))

  # Networking params
  replay_buffer_server_address = os.environ.get(
      'REPLAY_BUFFER_SERVER_ADDRESS', None
  )
  variable_container_server_address = os.environ.get(
      'VARIABLE_CONTAINER_SERVER_ADDRESS', None
  )
  replay_buffer_server_port = int(
      os.environ.get('REPLAY_BUFFER_SERVER_PORT', -1))
  variable_container_server_port = int(
      os.environ.get('VARIABLE_CONTAINER_SERVER_PORT', -1)
  )
  vocabulary_server_address = os.environ.get(
      'VOCABULARY_SERVER_ADDRESS', None
  )
  vocabulary_server_port = int(os.environ.get('VOCABULARY_SERVER_PORT', -1))

  print(f'replay_buffer_server_address: {replay_buffer_server_address}')
  print(f'replay_buffer_server_port: {replay_buffer_server_port}')
  print(
      f'variable_container_server_address: {variable_container_server_address}')
  print(f'variable_container_server_port: {variable_container_server_port}')

  print(f'batch_size: {batch_size}')
  print(f'debug: {debug}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'env_name: {env_name}')
  print(f'eval_interval: {eval_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'log_interval: {log_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'rb_capacity: {rb_capacity}')
  print(f'root_dir: {root_dir}')
  print(f'seed: {seed}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')

  if env_name == 'QuadrupedLocomotion-v0':
    print(f'motion_file_path: {motion_file_path}')
    print(f'max_vocab_size: {max_vocab_size}')
  elif env_name == 'WebNavigation-v0':
    print(f'vocabulary_server_address: {vocabulary_server_address}')
    print(f'vocabulary_server_port: {vocabulary_server_port}')
    print(f'difficulty_level: {difficulty_level}')
    print(f'num_websites: {num_websites}')
    print(f'embedding_dim: {embedding_dim}')
    print(f'latent_dim: {latent_dim}')
    print(f'epsilon_greedy: {epsilon_greedy}')
    print(f'profile_value_dropout: {profile_value_dropout}')

  gpus = tf.config.list_physical_devices('GPU')
  num_replicas = len(gpus) if gpus else 1

  # Parameters for training
  num_minibatches = timesteps_per_actorbatch // batch_size
  train_steps_per_iteration = num_minibatches // num_replicas
  num_iterations = np.maximum(1, total_env_steps // timesteps_per_actorbatch)
  max_train_steps = train_steps_per_iteration * num_iterations
  adjusted_timesteps_per_actorbatch = np.maximum(
      1, timesteps_per_actorbatch // env_batch_size
  )
  learner_iterations_per_call = 1
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
      ).astype(int))

  logging.info(f'train_steps_per_iteration: {train_steps_per_iteration}')
  logging.info(f'num_iterations: {num_iterations}')
  logging.info(f'max_train_steps: {max_train_steps}')
  logging.info(
      f'converted policy_checkpoint_interval: {policy_checkpoint_interval}'
  )
  logging.info(
      f'converted train_checkpoint_interval: {train_checkpoint_interval}'
  )
  logging.info(f'converted eval_interval: {eval_interval}')
  logging.info(f'converted log_interval: {log_interval}')
  logging.info(f'random seed: {seed}')

  no_gpu_env = os.environ.copy()
  no_gpu_env['CUDA_VISIBLE_DEVICES'] = '-1'

  env_flags = []
  all_processes = []
  if env_name == 'WebNavigation-v0':
    env_flags.extend([
        f'--env_name={env_name}',
        f'--num_websites={num_websites}',
        f'--difficulty_level={difficulty_level}',
    ])

    if job_type == 'collect':
      auth_key = 'secretkey'
      manager_command = [
          'python',
          'distributed/vocabulary_manager.py',
          f'--port={vocab_port}',
          f'--auth_key={auth_key}',
          f'--max_vocab_size={max_vocab_size}',
          '--verbosity=2',
      ]
      manager_process = subprocess.Popen(
          manager_command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=no_gpu_env
      )
      all_processes.append(manager_process)
      threading.Thread(
          target=print_subprocess_output, args=(manager_process,)
      ).start()
      logging.info('Successfully launched vocab manager server.')

  elif env_name == 'QuadrupedLocomotion-v0':
    env_flags.extend(
        [f'--env_name={env_name}', f'--motion_file_path={motion_file_path}']
    )

  if job_type == 'train':
    reverb_command = [
        'python',
        'distributed/sac_reverb_server.py',
        f'--port={replay_buffer_server_port}',
        f'--root_dir={root_dir}',
        f'--replay_buffer_capacity={rb_capacity}',
        f'--min_table_size_before_sampling={batch_size}',
        '--verbosity=2',
    ]

    reverb_process = subprocess.Popen(
        reverb_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=no_gpu_env,
    )
    threading.Thread(
        target=print_subprocess_output, args=(reverb_process,)
    ).start()
    logging.info('Successfully launched reverb server.')

  if job_type == 'collect':
    collect_job_commands = [
        [
            'python',
            'distributed/sac_collect.py',
            # Note: Use SAC-specific collect script
            f'--root_dir={root_dir}',
            f'--sequence_length={adjusted_timesteps_per_actorbatch}',
            f'--summary_interval={log_interval}',
            f'--env_batch_size={env_batch_size}',
            f'--initial_collect_steps={adjusted_timesteps_per_actorbatch}',
            f'--max_train_steps={max_train_steps}',
            f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
            f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
            f'--task={i}',
            '--verbosity=2' if i == 0 else '--verbosity=-2',
        ]
        + env_flags
        for i in range(env_batch_size)
    ]

    collect_jobs = []
    for command in collect_job_commands:
      process = subprocess.Popen(
          command,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          env=no_gpu_env,
      )
      collect_jobs.append(process)
      threading.Thread(target=print_subprocess_output, args=(process,)).start()
    all_processes.extend(collect_jobs)
    logging.info('Successfully launched collect jobs.')

  if job_type == 'train':
    train_job_command = [
                            'python',
                            'distributed/sac_train.py',
                            f'--batch_size={batch_size}',
                            f'--debug={debug}',
                            f'--env_batch_size={env_batch_size}',
                            f'--learner_iterations_per_call={learner_iterations_per_call}',
                            f'--learning_rate={learning_rate}',
                            f'--log_interval={log_interval}',
                            f'--max_train_steps={max_train_steps}',
                            f'--policy_checkpoint_interval={policy_checkpoint_interval}',
                            f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
                            f'--root_dir={root_dir}',
                            f'--seed={seed}',
                            f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
                            f'--train_checkpoint_interval={train_checkpoint_interval}',
                            f'--use_gpu',
                            f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
                            f'--verbosity=2',
                        ] + env_flags

    # Display the command
    logging.info(' '.join(train_job_command))

    train_job = subprocess.Popen(
        train_job_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )

    threading.Thread(target=print_subprocess_output, args=(train_job,)).start()
    logging.info('Successfully launched train job.')

  if job_type == 'train':
    while True:
      try:
        train_job.wait(timeout=30)
        break
      except subprocess.TimeoutExpired:
        logging.info('Train job still running.')
        continue
    logging.info('Train job finished.')
    reverb_process.terminate()
    logging.info('Successfully terminated reverb server.')

  # Wait for all processes to finish
  for process in all_processes:
    process.terminate()
    logging.info('Successfully terminated process.')


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
