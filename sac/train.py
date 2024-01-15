import os
import subprocess
import threading

import numpy as np
from absl import app
from absl import logging


def print_subprocess_output(process):
  for line in iter(process.stdout.readline, b''):
    print(line.decode(), end='')


def train():
  # Environment variables and parameters
  env_name = os.environ.get('ENV_NAME')
  domain = 'quadruped_locomotion' if env_name == 'QuadrupedLocomotion-v0' else 'web_navigation'

  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  batch_size = int(os.environ.get('BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  timesteps_per_actorbatch = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))

  motion_file_path = os.environ.get('MOTION_FILE_PATH',
                                    None) if domain == 'quadruped_locomotion' else None
  vocab_port = int(os.environ.get('VOCAB_PORT',
                                  '50000')) if domain == 'web_navigation' else None
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL',
                                        None)) if domain == 'web_navigation' else None
  num_websites = int(os.environ.get('NUM_WEBSITES',
                                    None)) if domain == 'web_navigation' else None
  max_vocab_size = int()
  port = int(os.environ.get('PORT', '8008'))
  host = os.environ.get('HOST', 'localhost')
  replay_buffer_server_address = f'{host}:{port}'
  variable_container_server_address = f'{host}:{port}'
  debug = bool(os.environ.get('DEBUG', False))
  rb_capacity = int(os.environ.get('RB_CAPACITY', None))

  # Print extracted and computed values
  print(f'Domain: {domain}')
  print(f'seed: {seed}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'motion_file_path: {motion_file_path}')
  print

  # Parameters for training
  train_steps_per_iteration = timesteps_per_actorbatch
  num_iterations = total_env_steps // timesteps_per_actorbatch
  max_train_steps = train_steps_per_iteration * num_iterations
  adjusted_timesteps_per_actorbatch = timesteps_per_actorbatch // env_batch_size
  learner_iterations_per_call = timesteps_per_actorbatch
  policy_checkpoint_interval = np.maximum(1, np.round(
      policy_checkpoint_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  train_checkpoint_interval = np.maximum(1, np.round(
      train_checkpoint_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  eval_interval = np.maximum(1, np.round(
      eval_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  log_interval = np.maximum(1, np.round(
      log_interval / env_batch_size / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))

  logging.info(f'train_steps_per_iteration: {train_steps_per_iteration}')
  logging.info(f'num_iterations: {num_iterations}')
  logging.info(f'max_train_steps: {max_train_steps}')
  logging.info(
      f'converted policy_checkpoint_interval: {policy_checkpoint_interval}')
  logging.info(
      f'converted train_checkpoint_interval: {train_checkpoint_interval}')
  logging.info(f'converted eval_interval: {eval_interval}')
  logging.info(f'converted log_interval: {log_interval}')
  logging.info(f'random seed: {seed}')

  if domain == 'web_navigation':
    auth_key = 'secretkey'
    manager_command = [
        'python', 'distributed/vocabulary_manager.py',
        f'--port={vocab_port}',
        f'--auth_key={auth_key}',
        f'--max_vocab_size=500',
        '--verbosity=2'
    ]
    manager_process = subprocess.Popen(manager_command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       env=os.environ.copy())
    threading.Thread(target=print_subprocess_output,
                     args=(manager_process,)).start()
    logging.info('Successfully launched vocab manager server.')

  # Launch reverb server
  reverb_command = [
      'python',
      'distributed/sac_reverb_server.py',
      f'--port={port}',
      f'--root_dir={root_dir}',
      f'--replay_buffer_capacity={rb_capacity}',
      f'--min_table_size_before_sampling={timesteps_per_actorbatch}',
      '--verbosity=2',
  ]

  reverb_process = subprocess.Popen(reverb_command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    env=os.environ.copy())
  threading.Thread(target=print_subprocess_output,
                   args=(reverb_process,)).start()
  logging.info('Successfully launched reverb server.')

  # Launch collect jobs with domain-specific configurations
  collect_job_commands = [
      ['python',
       'distributed/sac_collect.py',  # Note: Use SAC-specific collect script
       f'--root_dir={root_dir}',
       f'--sequence_length={adjusted_timesteps_per_actorbatch}',
       f'--summary_interval={log_interval}',
       f'--env_name={domain}-v0',  # Use domain-specific environment
       f'--env_batch_size={env_batch_size}',
       f'--motion_file_path={motion_file_path}' if domain == 'quadruped_locomotion' else '',
       f'--replay_buffer_server_address={replay_buffer_server_address}',
       f'--variable_container_server_address={variable_container_server_address}',
       f'--task={i}',
       f'--seed={seed}',
       '--verbosity=2' if i == 0 else '--verbosity=-2',
       ] for i in range(env_batch_size)
  ]

  collect_jobs = []
  for command in collect_job_commands:
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=os.environ.copy())
    collect_jobs.append(process)
    threading.Thread(target=print_subprocess_output, args=(process,)).start()
  logging.info('Successfully launched collect jobs.')

  # Launch train job with domain-specific configurations
  train_job_command = [
      'python',
      'distributed/sac_train.py',  # Note: Use SAC-specific train script
      f'--env_name={domain}-v0',
      f'--batch_size={batch_size}',
      f'--debug={debug}',
      f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
      f'--policy_checkpoint_interval={policy_checkpoint_interval}',
      f'--replay_buffer_server_address={replay_buffer_server_address}',
      f'--max_train_steps={max_train_steps}',
      f'--root_dir={root_dir}',
      f'--train_checkpoint_interval={train_checkpoint_interval}',
      f'--env_batch_size={env_batch_size}',
      f'--learning_rate={learning_rate}',
      f'--motion_file_path={motion_file_path}' if domain == 'quadruped_locomotion' else '',
      f'--log_interval={log_interval}',
      f'--seed={seed}',
      f'--variable_container_server_address={variable_container_server_address}',
      f'--learner_iterations_per_call={learner_iterations_per_call}',
      f'--use_gpu',
      f'--verbosity=2'
  ]
  train_job = subprocess.Popen(train_job_command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=os.environ.copy())

  threading.Thread(target=print_subprocess_output, args=(train_job,)).start()
  logging.info('Successfully launched train job.')

  # Monitor the training job and handle its completion
  while True:
    try:
      train_job.wait(timeout=30)
      break
    except subprocess.TimeoutExpired:
      logging.info('Train job still running.')
      continue
  logging.info('Train job finished.')

  # Terminate the reverb server
  reverb_process.terminate()
  logging.info('Successfully terminated reverb server.')

  # Terminate the collect jobs
  for process in collect_jobs:
    process.terminate()
  logging.info('Successfully terminated collect jobs.')


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
