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
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  batch_size = int(os.environ.get('BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  rb_capacity = int(os.environ.get('RB_CAPACITY', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  timesteps_per_actorbatch = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))
  port = int(os.environ.get('PORT', '8008'))
  host = os.environ.get('HOST', 'localhost')
  replay_buffer_server_address = f'{host}:{port}'
  variable_container_server_address = f'{host}:{port}'
  debug = bool(os.environ.get('DEBUG', False))
  epsilon_greedy = float(os.environ.get('EPSILON_GREEDY', None))
  reverb_port = int(os.environ.get('REVERB_PORT', '8008'))
  vocab_port = int(os.environ.get('VOCAB_PORT', '50000'))
  num_websites = int(os.environ.get('NUM_WEBSITES', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  max_vocab_size = int(os.environ.get('MAX_VOCAB_SIZE', None))
  latent_dim = int(os.environ.get('LATENT_DIM', None))
  embedding_dim = int(os.environ.get('EMBEDDING_DIM', None))
  profile_value_dropout = float(os.environ.get('PROFILE_VALUE_DROPOUT', None))

  # Print extracted and computed values
  print(f'batch_size: {batch_size}')
  print(f'debug: {debug}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'epsilon_greedy: {epsilon_greedy}')
  print(f'eval_interval: {eval_interval}')
  print(f'learning_rate: {learning_rate}')
  print(f'log_interval: {log_interval}')
  print(f'num_websites: {num_websites}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'port: {port}')
  print(f'rb_capacity: {rb_capacity}')
  print(f'replay_buffer_server_address: {replay_buffer_server_address}')
  print(f'reverb_port: {reverb_port}')
  print(f'root_dir: {root_dir}')
  print(f'seed: {seed}')
  print(f'timesteps_per_actorbatch: {timesteps_per_actorbatch}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(
      f'variable_container_server_address: {variable_container_server_address}')
  print(f'vocab_port: {vocab_port}')
  print(f'max_vocab_size: {max_vocab_size}')
  print(f'latent_dim: {latent_dim}')
  print(f'profile_value_dropout: {profile_value_dropout}')

  # Parameters for training
  train_steps_per_iteration = timesteps_per_actorbatch
  num_iterations = total_env_steps // timesteps_per_actorbatch
  max_train_steps = train_steps_per_iteration * num_iterations
  adjusted_timesteps_per_actorbatch = timesteps_per_actorbatch // env_batch_size

  eval_interval = np.maximum(1, np.round(
      eval_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  log_interval = np.maximum(1, np.round(
      log_interval / env_batch_size / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  policy_checkpoint_interval = np.maximum(1, np.round(
      policy_checkpoint_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  train_checkpoint_interval = np.maximum(1, np.round(
      train_checkpoint_interval / timesteps_per_actorbatch * train_steps_per_iteration).astype(
      int))
  learner_iterations_per_call = np.maximum(1, np.round(
      timesteps_per_actorbatch / batch_size).astype(int))
  logging.info(f'converted eval_interval: {eval_interval}')
  logging.info(f'converted log_interval: {log_interval}')
  logging.info(
      f'converted policy_checkpoint_interval: {policy_checkpoint_interval}')
  logging.info(
      f'converted train_checkpoint_interval: {train_checkpoint_interval}')
  logging.info(f'max_train_steps: {max_train_steps}')
  logging.info(f'num_iterations: {num_iterations}')
  logging.info(f'random seed: {seed}')
  logging.info(f'train_steps_per_iteration: {train_steps_per_iteration}')
  logging.info(
      f'learner_iterations_per_call: {learner_iterations_per_call}')

  # Launch multiprocessing manager server
  auth_key = 'secretkey'
  manager_command = [
      'python', '-u', 'distributed/vocabulary_manager.py',
      f'--port={vocab_port}',
      f'--auth_key={auth_key}',
      f'--max_vocab_size=500',
      '--verbosity=2',
  ]

  # Launch the subprocess with the same environment and output redirection
  vocab_manager = subprocess.Popen(manager_command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   env=os.environ.copy())
  threading.Thread(target=print_subprocess_output,
                   args=(vocab_manager,)).start()
  logging.info('Successfully launched vocab manager server.')

  # Launch reverb server
  reverb_command = [
      'python', '-u',
      'distributed/ddqn_reverb_server.py',
      '--verbosity=2',
      f'--min_table_size_before_sampling={timesteps_per_actorbatch}',
      f'--port={port}',
      f'--replay_buffer_capacity={rb_capacity}',
      f'--root_dir={root_dir}',
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
      ['xvfb-run',
       'python', '-u',
       'distributed/ddqn_collect.py',
       '--verbosity=2' if debug else '--verbosity=-2',
       f'--env_batch_size={env_batch_size}',
       f'--env_name=WebNavigation-v0',
       f'--initial_collect_steps={adjusted_timesteps_per_actorbatch}',
       f'--max_train_steps={max_train_steps}',
       f'--num_websites={num_websites}',
       f'--difficulty_level={difficulty_level}',
       f'--replay_buffer_server_address={replay_buffer_server_address}',
       f'--root_dir={root_dir}',
       f'--vocab_port={vocab_port}',
       f'--auth_key={auth_key}',

       f'--summary_interval={log_interval}',
       f'--task={i}',
       f'--variable_container_server_address={variable_container_server_address}',
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
      'python', '-u',
      'distributed/ddqn_train.py',
      f'--batch_size={batch_size}',
      f'--debug={debug}',
      f'--difficulty_level={difficulty_level}',
      f'--embedding_dim={embedding_dim}',
      f'--env_batch_size={env_batch_size}',
      f'--env_name=WebNavigation-v0',
      f'--epsilon_greedy={epsilon_greedy}',
      f'--latent_dim={latent_dim}',
      f'--learner_iterations_per_call={learner_iterations_per_call}',
      f'--learning_rate={learning_rate}',
      f'--log_interval={log_interval}',
      f'--max_train_steps={max_train_steps}',
      f'--max_vocab_size={max_vocab_size}',
      f'--num_websites={num_websites}',
      f'--policy_checkpoint_interval={policy_checkpoint_interval}',
      f'--profile_value_dropout={profile_value_dropout}',
      f'--replay_buffer_server_address={replay_buffer_server_address}',
      f'--root_dir={root_dir}',
      f'--seed={seed}',
      f'--timesteps_per_actorbatch={timesteps_per_actorbatch}',
      f'--train_checkpoint_interval={train_checkpoint_interval}',
      f'--use_gpu',
      f'--variable_container_server_address={variable_container_server_address}',
      f'--verbosity=2'
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

  # Terminate the vocab manager server
  vocab_manager.terminate()
  logging.info('Successfully terminated vocab manager server.')


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
