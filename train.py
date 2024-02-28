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


def train_command(num_iterations, entropy_regularization, use_gae, root_dir,
    variable_container_server_address, variable_container_server_port,
    replay_buffer_server_address, replay_buffer_server_port, env_name,
    max_sequence_length, num_episodes_per_iteration, log_interval, use_gpu,
    seed, task_name,
    num_epochs, batch_size, shuffle_buffer_size, num_replicas, algorithm, debug,
    epsilon_greedy, train_checkpoint_interval, policy_checkpoint_interval,
    env_batch_size, learning_rate, exploration_noise_std, max_train_steps,
    learner_iterations_per_call):
  return [
      'python',
      '-m',
      'train_lib.train',
      f'--algorithm={algorithm}',
      f'--batch_size={batch_size}',
      f'--debug={debug}',
      f'--entropy_regularization={entropy_regularization}',
      f'--env_name={env_name}',
      f'--epsilon_greedy={epsilon_greedy}',
      f'--exploration_noise_std={exploration_noise_std}',
      f'--learner_iterations_per_call={learner_iterations_per_call}',
      f'--learning_rate={learning_rate}',
      f'--log_interval={log_interval}',
      f'--sequence_length={max_sequence_length}',
      f'--max_train_steps={max_train_steps}',
      f'--num_epochs={num_epochs}',
      f'--num_iterations={num_iterations}',
      f'--policy_checkpoint_interval={policy_checkpoint_interval}',
      f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
      f'--root_dir={root_dir}',
      f'--seed={seed}',
      f'--shuffle_buffer_size={shuffle_buffer_size}',
      f'--num_episodes_per_iteration={num_episodes_per_iteration}',
      f'--train_checkpoint_interval={train_checkpoint_interval}',
      f'--use_gae={use_gae}',
      f'--use_gpu={use_gpu}',
      f'--task_name={task_name}',
      f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
      # Only use these if you have a pretrained policy to bootstrap from
      # f'--policy_saved_model_dir={root_dir}/policies/policy',
      # f'--policy_checkpoint_dir={root_dir}/policies/checkpoints',
  ]


def collect_command(algorithm, debug, env_batch_size, env_name,
    initial_collect_steps,
    max_sequence_length, max_train_steps, num_replicas,
    replay_buffer_server_address, replay_buffer_server_port, root_dir, seed,
    log_interval, variable_container_server_address,
    variable_container_server_port, vocabulary_manager_auth_key,
    vocabulary_server_address, vocabulary_server_port, task):
  return ['python',
          '-m',
          'train_lib.collect',
          f'--algorithm={algorithm}',
          f'--debug={debug}',
          f'--env_batch_size={env_batch_size}',
          f'--env_name={env_name}',
          f'--initial_collect_steps={initial_collect_steps}',
          f'--max_sequence_length={max_sequence_length}',
          f'--max_train_steps={max_train_steps}',
          f'--num_replicas={num_replicas}',
          f'--replay_buffer_server_address={replay_buffer_server_address}:{replay_buffer_server_port}',
          f'--root_dir={root_dir}',
          f'--seed={seed}',
          f'--summary_interval={log_interval}',
          f'--task={task}',
          f'--variable_container_server_address={variable_container_server_address}:{variable_container_server_port}',
          f'--verbosity={"1" if task == 0 else "-1"}',
          f'--vocabulary_manager_auth_key={vocabulary_manager_auth_key}',
          f'--vocabulary_server_address={vocabulary_server_address}',
          f'--vocabulary_server_port={vocabulary_server_port}']


def reverb_command(task_name, replay_buffer_server_port, root_dir,
    replay_buffer_capacity, algorithm, min_table_size_before_sampling,
):
  return [
      'python',
      '-m',
      'train_lib.reverb_server',
      f'--port={replay_buffer_server_port}',
      f'--task_name={task_name}',
      f'--root_dir={root_dir}',
      f'--replay_buffer_capacity={replay_buffer_capacity}',
      f'--algorithm={algorithm}',
      f'--min_table_size_before_sampling={min_table_size_before_sampling}',
      f'--verbosity={logging.get_verbosity()}',
  ]


def train():
  job_type = os.environ.get('JOB_TYPE', None)
  if job_type is None:
    raise ValueError('Job type must be set.')

  mode = os.environ.get('MODE', None)
  if mode is None:
    raise ValueError('Mode must be set to either train or inference.')
  task_name = os.environ.get('TASK_NAME', None)
  algorithm = os.environ.get('ALGORITHM', None)
  seed = int(os.environ.get('SEED', -1))
  use_gae = bool(os.environ.get('USE_GAE', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  num_epochs = int(os.environ.get('NUM_EPOCHS', -1))
  replay_buffer_capacity = int(os.environ.get('RB_CAPACITY', -1))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', -1))
  batch_size = int(os.environ.get('BATCH_SIZE', -1))
  num_iterations = int(os.environ.get('NUM_ITERATIONS', -1))
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
  num_episodes_per_iteration = int(
      os.environ.get('NUM_EPISODES_PER_ITERATION', -1)
  )
  max_sequence_length = int(os.environ.get('MAX_SEQUENCE_LENGTH', -1))
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
  num_replicas = int(os.environ.get('NUM_REPLICAS', -1))

  # Networking params
  num_collect_jobs = int(os.environ.get('NUM_COLLECT_JOBS_PER_MACHINE', -1))
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

  print('batch_size:', batch_size)
  print('debug:', debug)
  print('difficulty_level:', difficulty_level)
  print('embedding_dim:', embedding_dim)
  print('entropy_regularization:', entropy_regularization)
  print('env_batch_size:', env_batch_size)
  print('env_name:', env_name)
  print('epsilon_greedy:', epsilon_greedy)
  print('eval_interval:', eval_interval)
  print('init_placement_path:', init_placement_path)
  print('latent_dim:', latent_dim)
  print('learning_rate:', learning_rate)
  print('log_interval:', log_interval)
  print('max_vocab_size:', max_vocab_size)
  print('motion_file_path:', motion_file_path)
  print('netlist_path:', netlist_path)
  print('num_epochs:', num_epochs)
  print('num_websites:', num_websites)
  print('policy_checkpoint_interval:', policy_checkpoint_interval)
  print('profile_value_dropout:', profile_value_dropout)
  print('replay_buffer_server_address:', replay_buffer_server_address)
  print('replay_buffer_server_port:', replay_buffer_server_port)
  print('root_dir:', root_dir)
  print('seed:', seed)
  print('std_cell_placer_mode:', std_cell_placer_mode)
  print('num_episodes_per_iteration:', num_episodes_per_iteration)
  print('train_checkpoint_interval:', train_checkpoint_interval)
  print('variable_container_server_address:', variable_container_server_address)
  print('variable_container_server_port:', variable_container_server_port)
  print('vocabulary_server_address:', vocabulary_server_address)
  print('vocabulary_server_port:', vocabulary_server_port)

  # Check if the selected algorithm is Proximal Policy Optimization (PPO)
  if algorithm in ('ppo',):
    train_steps_per_iteration = max(1, int(
        num_episodes_per_iteration * max_sequence_length / batch_size * num_epochs / num_replicas
    ))

    # Shuffle the data coming from three episodes
    shuffle_buffer_size = 3 * max_sequence_length

    # Only a single iteration is performed per call to the learner. We set the
    # `num_samples` argument to `env_batch_size` to ensure that the learner
    # processes all the data collected by the actors in a single call.
    learner_iterations_per_call = 1

    # No need to collect data initially in PPO.
    initial_collect_steps = 0

    min_table_size_before_sampling = 1

  elif algorithm in ('sac', 'ddqn', 'td3', 'ddpg', 'dqn'):
    # We want to exhaust `num_episodes_per_iteration` worth of samples each iteration
    learner_iterations_per_call = np.maximum(
        1, (num_episodes_per_iteration * max_sequence_length) / batch_size
    ).astype(int)
    train_steps_per_iteration = learner_iterations_per_call
    shuffle_buffer_size = -1
    initial_collect_steps = max_sequence_length
    min_table_size_before_sampling = 1
  else:
    raise ValueError(f'Unsupported algorithm: {algorithm}')
  if train_steps_per_iteration < 1:
    raise ValueError(
        'train_steps_per_iteration must be at least 1, got'
        f' {train_steps_per_iteration}'
    )

  max_train_steps = train_steps_per_iteration * num_iterations
  print('shuffle_buffer_size:', shuffle_buffer_size)
  print('train_steps_per_iteration:', train_steps_per_iteration)
  print('learner_iterations_per_call:', learner_iterations_per_call)
  print('initial_collect_steps:', initial_collect_steps)
  print('num_iterations:', num_iterations)
  print('min_table_size_before_sampling:', min_table_size_before_sampling)
  print('max_train_steps:', max_train_steps)

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
      print('Command for vocab manager:', vocab_manager_command)
      vocab_server_process = create_and_manage_process(
          vocab_manager_command, all_processes
      )

      if vocab_server_process.poll() is not None:
        raise ValueError('Vocabulary manager server failed to start.')
      else:
        print('Successfully launched vocab manager server.')
  elif env_name == 'QuadrupedLocomotion-v0':
    env_flags.extend(
        [f'--env_name={env_name}', f'--motion_file_path={motion_file_path}']
    )
  elif env_name == 'CircuitTraining-v0':
    env_flags.extend([
        f'--std_cell_placer_mode={std_cell_placer_mode}',
        f'--netlist_file={netlist_path}',
        f'--init_placement={init_placement_path}',
    ])
  else:
    raise ValueError(f'Unsupported environment: {env_name}')

  if job_type == 'collect':
    collect_job_commands = [
        collect_command(
            algorithm=algorithm, env_name=env_name, debug=debug,
            env_batch_size=env_batch_size,
            initial_collect_steps=initial_collect_steps,
            max_sequence_length=max_sequence_length,
            max_train_steps=max_train_steps, num_replicas=num_replicas,
            replay_buffer_server_address=replay_buffer_server_address,
            replay_buffer_server_port=replay_buffer_server_port,
            root_dir=root_dir, seed=seed, log_interval=log_interval,
            variable_container_server_address=variable_container_server_address,
            variable_container_server_port=variable_container_server_port,
            vocabulary_manager_auth_key=vocabulary_manager_auth_key,
            vocabulary_server_address=vocabulary_server_address,
            vocabulary_server_port=vocabulary_server_port,
            task=i
        ) + env_flags
        for i in range(num_collect_jobs)
    ]

    # Display one of the commands
    print(' '.join(collect_job_commands[0]))

    collect_jobs = []
    for command in collect_job_commands:
      collect_jobs.append(create_and_manage_process(command, all_processes,
                                                    ))
    print('Successfully launched collect jobs.')

    # Recall that root dir is modified for collect jobs to separate
    # system metrics from the train job
    while not os.path.exists(
        os.path.join(root_dir, '../../', 'training_complete')
    ):
      time.sleep(PROCESS_WAIT_INTERVAL)

  elif job_type == 'train':
    reverb_job_command = reverb_command(
        task_name=task_name,
        replay_buffer_server_port=replay_buffer_server_port,
        root_dir=root_dir, replay_buffer_capacity=replay_buffer_capacity,
        algorithm=algorithm,
        min_table_size_before_sampling=min_table_size_before_sampling
    )
    print(' '.join(reverb_job_command))
    create_and_manage_process(
        reverb_job_command,
        all_processes,
        env_vars=dict(**os.environ, CUDA_VISIBLE_DEVICES='-1'),
    )
    print('Successfully launched reverb server.')

    train_job_command = train_command(
        num_iterations=num_iterations,
        entropy_regularization=entropy_regularization, use_gae=use_gae,
        root_dir=root_dir,
        env_name=env_name,
        variable_container_server_address=variable_container_server_address,
        variable_container_server_port=variable_container_server_port,
        replay_buffer_server_address=replay_buffer_server_address,
        replay_buffer_server_port=replay_buffer_server_port,
        max_sequence_length=max_sequence_length,
        num_episodes_per_iteration=num_episodes_per_iteration,
        log_interval=log_interval,
        task_name=task_name,
        use_gpu=True, seed=seed, num_epochs=num_epochs, batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size, num_replicas=num_replicas,
        algorithm=algorithm, debug=debug, epsilon_greedy=epsilon_greedy,
        train_checkpoint_interval=train_checkpoint_interval,
        policy_checkpoint_interval=policy_checkpoint_interval,
        env_batch_size=env_batch_size, learning_rate=learning_rate,
        exploration_noise_std=exploration_noise_std,
        max_train_steps=max_train_steps,
        learner_iterations_per_call=learner_iterations_per_call
    ) + env_flags

    # Display the command
    print(' '.join(train_job_command))
    create_and_manage_process(train_job_command, all_processes)
    print('Successfully launched train job.')

    while not os.path.exists(os.path.join(root_dir, 'training_complete')):
      time.sleep(PROCESS_WAIT_INTERVAL)

  print('Training complete.')


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
