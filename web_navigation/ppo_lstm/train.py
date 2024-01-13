import os
import subprocess
import threading

import numpy as np
from absl import app
from absl import logging
import tensorflow as tf


def print_subprocess_output(process):
  for line in iter(process.stdout.readline, b''):
    print(line.decode(), end='')


def train():
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  num_epochs = int(os.environ.get('NUM_EPOCHS', None))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  batch_size = int(os.environ.get('BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  entropy_regularization = float(os.environ.get('ENTROPY_REGULARIZATION', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  timesteps_per_actorbatch = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))
  reverb_port = int(os.environ.get('REVERB_PORT', '8008'))
  vocab_port = int(os.environ.get('VOCAB_PORT', '50000'))
  host = os.environ.get('HOST', 'localhost')
  replay_buffer_server_address = f'{host}:{reverb_port}'
  variable_container_server_address = f'{host}:{reverb_port}'
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  num_websites = int(os.environ.get('NUM_WEBSITES', None))

  # Print extracted and computed values
  print(f'seed: {seed}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'num_epochs: {num_epochs}')
  print(f'batch_size: {batch_size}')
  print(f'entropy_regularization: {entropy_regularization}')
  print(f'reverb_port: {reverb_port}')
  print(f'vocab_port: {vocab_port}')
  print(f'host: {host}')
  print(f'replay_buffer_server_address: {replay_buffer_server_address}')
  print(
      f'variable_container_server_address: {variable_container_server_address}')
  print(f'num_websites: {num_websites}')

  # Get the number of GPUs
  gpus = tf.config.list_physical_devices('GPU')
  num_replicas = len(gpus) if gpus else 1

  # Parameters for training
  num_minibatches = timesteps_per_actorbatch // batch_size
  train_steps_per_iteration = num_minibatches * num_epochs // num_replicas
  num_iterations = total_env_steps // timesteps_per_actorbatch
  max_train_steps = train_steps_per_iteration * num_iterations
  adjusted_timesteps_per_actorbatch = timesteps_per_actorbatch // env_batch_size

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

  # Launch multiprocessing manager server
  auth_key = 'secretkey'
  manager_command = [
      'python', 'distributed/vocabulary_manager.py',
      f'--port={vocab_port}',
      f'--auth_key={auth_key}',
      f'--max_vocab_size=500',
      '--verbosity=2',
  ]

  # Launch the subprocess with the same environment and output redirection
  manager_process = subprocess.Popen(manager_command, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     env=os.environ.copy())
  threading.Thread(target=print_subprocess_output,
                   args=(manager_process,)).start()
  logging.info('Successfully launched vocab manager server.')

  # Launch reverb server
  reverb_command = [
      'python',
      'distributed/ppo_lstm_reverb_server.py',
      f'--port={reverb_port}',
      f'--root_dir={root_dir}',
      '--verbosity=2',
  ]

  # Launch the subprocess with the same environment and output redirection
  reverb_process = subprocess.Popen(reverb_command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    env=os.environ.copy())
  threading.Thread(target=print_subprocess_output,
                   args=(reverb_process,)).start()
  logging.info('Successfully launched reverb server.')

  # Launch collect jobs
  collect_job_commands = [
      ['python',
       'distributed/ppo_lstm_collect.py',
       f'--root_dir={root_dir}',
       f'--vocab_port={vocab_port}',
       f'--auth_key={auth_key}',
       f'--sequence_length={adjusted_timesteps_per_actorbatch}',
       f'--summary_interval={log_interval}',
       f'--env_name=WebNavigation-v0',
       f'--env_batch_size={env_batch_size}',
       f'--replay_buffer_server_address={replay_buffer_server_address}',
       f'--variable_container_server_address={variable_container_server_address}',
       f'--num_websites={num_websites}',
       f'--difficulty_level={difficulty_level}',
       f'--task={i}',
       '--verbosity=-2',
       ] for i in range(env_batch_size)
  ]

  collect_jobs = []
  for command in collect_job_commands:
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=os.environ.copy())
  collect_jobs.append(process)
  threading.Thread(target=print_subprocess_output, args=(process,)).start()
  logging.info('Successfully launched collect jobs.')

  # Launch train job
  train_job_command = [
      'python',
      'distributed/ppo_lstm_train.py',
      f'--entropy_regularization={entropy_regularization}',
      f'--env_name=WebNavigation-v0',
      f'--num_epochs={num_epochs}',
      f'--batch_size={batch_size}',
      f'--debug=False',
      f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
      f'--sequence_length={adjusted_timesteps_per_actorbatch}',
      f'--policy_checkpoint_interval={policy_checkpoint_interval}',
      f'--replay_buffer_server_address={replay_buffer_server_address}',
      f'--root_dir={root_dir}',
      f'--train_checkpoint_interval={train_checkpoint_interval}',
      f'--num_websites={num_websites}',
      f'--difficulty_level={difficulty_level}',
      f'--max_train_steps={max_train_steps}',
      f'--env_batch_size={env_batch_size}',
      f'--learning_rate={learning_rate}',
      f'--embedding_dim=100',
      f'--latent_dim=50',
      f'--profile_value_dropout=0.0',
      f'--max_vocab_size=500',
      f'--log_interval={log_interval}',
      f'--seed={seed}',
      f'--variable_container_server_address={variable_container_server_address}',
      f'--use_gpu'
  ]
  train_job = subprocess.Popen(train_job_command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=os.environ.copy())
  threading.Thread(target=print_subprocess_output, args=(train_job,)).start()
  logging.info('Successfully launched train job.')

  # We need to wait for the train job to finish before we can terminate the
  # reverb server. Otherwise, the reverb server will terminate before the train
  # job is finished and the train job will fail.
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
