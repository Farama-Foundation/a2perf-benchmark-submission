import os
import subprocess

from absl import app
from absl import logging
from a2perf.domains import quadruped_locomotion


def main(_):
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  num_epochs = int(os.environ.get('NUM_EPOCHS', None))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  entropy_regularization = float(os.environ.get('ENTROPY_REGULARIZATION', None))
  train_checkpoint_interval = int(
      os.environ.get('INT_SAVE_FREQ', None))
  policy_checkpoint_interval = int(
      os.environ.get('INT_SAVE_FREQ', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  timesteps_per_actorbatch = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))
  # motion_file_path = os.environ.get('MOTION_FILE_PATH', None)
  # the motion file dog_pace is in the a2perf package
  motion_file_path = os.path.join(
      os.path.dirname(quadruped_locomotion.__file__), 'motion_imitation',
      'data', 'motions', 'dog_pace.txt')

  port = int(os.environ.get('PORT', '8008'))
  host = os.environ.get('HOST', 'localhost')
  replay_buffer_server_address = f'{host}:{port}'
  variable_container_server_address = f'{host}:{port}'

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
  print(f'motion_file_path: {motion_file_path}')
  print(f'num_epochs: {num_epochs}')
  num_iterations = int(total_env_steps / timesteps_per_actorbatch)

  # Launch reverb server
  reverb_command = [
      'python',
      'ppo_reverb_server.py',
      f'--min_table_size_before_sampling={timesteps_per_actorbatch}',
      f'--replay_buffer_capacity={timesteps_per_actorbatch}',
      f'--port={port}',
      f'--root_dir={root_dir}',
      '--alsologtostderr'
  ]

  # Launch the subprocess with output redirection
  reverb_process = subprocess.Popen(reverb_command, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
  # Used for breakpoint debugging
  # reverb_output = reverb_process.communicate()[0]
  logging.info('Successfully launched reverb server.')

  # Launch collect jobs
  adjusted_timesteps_per_actorbatch = timesteps_per_actorbatch // env_batch_size
  collect_job_commands = [
      ['python',
       'ppo_collect.py',
       f'--root_dir={root_dir}',
       f'--env_name=QuadrupedLocomotion-v0',
       f'--motion_file_path={motion_file_path}',
       f'--summary_interval={log_interval}',
       f'--replay_buffer_server_address={replay_buffer_server_address}',
       f'--variable_container_server_address={variable_container_server_address}',
       f'--task={i}',
       '--alsologtostderr',
       ] for i in range(env_batch_size)
  ]

  collect_jobs = [subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT) for command in
                  collect_job_commands]
  # Used for breakpoint debugging
  # collect_output = collect_jobs[0].communicate()[0]
  logging.info('Successfully launched collect jobs.')

  # Launch train job
  train_job_command = [
      'python',
      'ppo_train.py',
      f'--entropy_regularization={entropy_regularization}',
      f'--env_name=QuadrupedLocomotion-v0',
      f'--num_epochs={num_epochs}',
      f'--batch_size={timesteps_per_actorbatch}',
      # sample entire replay buffer since on-policy
      f'--policy_checkpoint_interval={policy_checkpoint_interval}',
      f'--replay_buffer_server_address={replay_buffer_server_address}',
      f'--root_dir={root_dir}',
      f'--train_checkpoint_interval={train_checkpoint_interval}',
      f'--use_gpu=True',
      f'--use_tpu=False',
      f'--env_batch_size={env_batch_size}',
      f'--learning_rate={learning_rate}',
      f'--motion_file_path={motion_file_path}',
      f'--log_interval={log_interval}',
      f'--variable_container_server_address={variable_container_server_address}',
  ]
  train_job = subprocess.Popen(train_job_command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
  # Used for breakpoint debugging
  # train_output = train_job.communicate()[0]
  logging.info('Successfully launched train job.')

  # We need to wait for the train job to finish before we can terminate the
  # reverb server. Otherwise, the reverb server will terminate before the train
  # job is finished and the train job will fail.

  # Print out some logging with the wait funciton tho

  # Wait for the train job to finish
  while True:
    try:
      train_job.wait(timeout=10)
      break
    except subprocess.TimeoutExpired:
      logging.info('Train job still running.')
      continue
  logging.info('Train job finished.')


if __name__ == '__main__':
  app.run(main)
