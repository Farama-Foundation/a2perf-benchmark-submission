import json
import multiprocessing as mp
import os
import random
import time

import gin
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tf_agents
from absl import logging
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from a2perf.domains.web_navigation.gwob.CoDE import networks
from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node


def create_env(
    env_name='CartPole-v0',
    env_args=None,
):
  return suite_gym.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs=env_args,
  )


def filter_invalid_transition(trajectories, _):
  # Prepare replay buffer as dataset with invalid transitions filtered.
  return ~trajectories.is_boundary()[0]


@gin.configurable
def train_eval(
    root_dir,
    env_name='WebNavigation-v0',
    num_iterations=100000,
    # Params for collect
    initial_collect_steps=5,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.05,
    max_vocab_size=500,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5000,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=32,
    environment_batch_size=1,
    learning_rate=1e-4,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    # Params for summaries and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    env_args=None,
    seed=0,
):
  """A simple train and eval for DQN."""

  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  summary_dir = os.path.join(root_dir, 'summaries')

  train_summary_writer = tf.summary.create_file_writer(
      logdir=os.path.join(summary_dir, 'train'),
      name='train_summary_writer',
      flush_millis=summaries_flush_secs * 1000,
      max_queue=10,
      experimental_trackable=True,

  )
  eval_summary_writer = tf.summary.create_file_writer(
      logdir=os.path.join(summary_dir, 'eval'),
      name='eval_summary_writer',
      flush_millis=summaries_flush_secs * 1000,
      max_queue=tf.data.experimental.AUTOTUNE,
      experimental_trackable=True,
  )
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]
  global_step = tf.compat.v1.train.get_or_create_global_step()
  manager = mp.Manager()
  global_vocab = vocabulary_node.LockedMultiprocessingVocabulary(
      max_vocabulary_size=max_vocab_size,
      multiprocessing_manager=manager,
  )
  env_args.update({'global_vocabulary': global_vocab})

  # Parallel environment creation
  envs = [
      lambda: create_env(
          env_name=env_name,
          env_args=env_args,
      )
      for i in range(environment_batch_size)
  ]
  eval_env_args = env_args.copy()
  eval_env_args.update(
      dict(
          seed=seed + environment_batch_size,
          cyclic_action_penalty=0.0,
          timestep_penalty=0.0,
      )
  )
  eval_env = tf_agents.environments.ParallelPyEnvironment(
      [
          lambda: create_env(
              env_name=env_name,
              env_args=eval_env_args,
          )
      ]
  )
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
  parallel_py_env = tf_agents.environments.ParallelPyEnvironment(
      envs, blocking=False, start_serially=True, flatten=False
  )
  tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)
  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()
  logging.info('Successfully created environments')

  with tf.name_scope('QNetwork'):
    q_net = networks.WebLSTMQNetwork(
        vocab_size=max_vocab_size
        if max_vocab_size is not None
        else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
        profile_value_dropout=0.0,
        q_min=None,
        q_max=None,
        embedding_dim=100,
        name='q_network',
        latent_dim=50,
        return_state_value=False
    )
  with tf.name_scope('DQNAgent'):
    tf_agent = dqn_agent.DqnAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate
        ),
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
        name='dqn_agent',
    )
    tf_agent.initialize()

  environment_steps_metric = tf_metrics.EnvironmentSteps()
  step_metrics = [
      tf_metrics.NumberOfEpisodes(),
      environment_steps_metric,
  ]
  train_metrics = step_metrics + [
      tf_metrics.AverageReturnMetric(batch_size=environment_batch_size,
                                     buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(
          batch_size=environment_batch_size, buffer_size=num_eval_episodes
      ),
  ]

  eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
  initial_collect_policy = random_tf_policy.RandomTFPolicy(
      tf_env.time_step_spec(), tf_env.action_spec()
  )
  collect_policy = tf_agent.collect_policy

  # Make the replay buffer.
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      data_spec=tf_agent.collect_data_spec,
      batch_size=tf_env.batch_size,
      max_length=replay_buffer_capacity,
      device='/cpu:*',
  )
  replay_observer = [replay_buffer.add_batch]

  train_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'train'),
      agent=tf_agent,
      global_step=global_step,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
  )

  saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
  saved_model_dir = os.path.join(root_dir, 'policies')
  train_checkpointer.initialize_or_restore()

  # Restore the vocab from the corresponding global step
  vocab_path = os.path.join(
      train_dir, f'vocab_{global_step.value().numpy()}.npy'
  )

  if os.path.exists(vocab_path):
    state = json.load(open(vocab_path, 'r'))
    global_vocab.restore(state)

  initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      initial_collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=initial_collect_steps,
  )

  collect_driver = dynamic_step_driver.DynamicStepDriver(
      tf_env,
      collect_policy,
      observers=replay_observer + train_metrics,
      num_steps=collect_steps_per_iteration,
  )

  dataset = (
      replay_buffer.as_dataset(sample_batch_size=batch_size,
                               num_steps=2,
                               num_parallel_calls=tf.data.AUTOTUNE
                               )
      .unbatch()
      .filter(filter_invalid_transition)
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
  )

  # Dataset generates trajectories with shape [Bx2x...]
  iterator = iter(dataset)

  def train_step():
    experience, _ = next(iterator)
    return tf_agent.train(experience)

  if use_tf_functions:
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

  if replay_buffer.num_frames() == 0:
    # Collect initial replay data.
    logging.info(
        'Initializing replay buffer by collecting experience for %d steps '
        'with a random policy.',
        initial_collect_steps,
    )
    initial_collect_driver.run()

  train_step = common.function(train_step)
  time_acc = 0
  collect_time = 0
  train_time = 0
  timed_at_step = global_step.value()
  iters_so_far = 0
  start_time = time.time()

  # Eval once before training.
  metric_utils.eager_compute(
      environment=eval_tf_env,
      policy=eval_policy,
      num_episodes=num_eval_episodes,
      train_step=iters_so_far,
      summary_writer=eval_summary_writer,
      use_function=True,
      summary_prefix='Metrics',
      metrics=eval_metrics,
  )

  # Compute train metrics once at the beginning of training
  with train_summary_writer.as_default():
    for train_metric in train_metrics:
      metric_value = train_metric.result()
      metric_name = f"Metrics/{train_metric.name}"
      tf.summary.scalar(metric_name, metric_value, step=global_step)
    tf.summary.scalar('info/iters_so_far', iters_so_far, step=iters_so_far)

  logging.info('Beginning training at step: %d', global_step.value().numpy())
  with train_summary_writer.as_default():
    while iters_so_far < num_iterations:
      start = time.time()
      collect_driver.run()
      collect_time += time.time() - start

      start = time.time()
      train_loss = sum(
          [train_step().loss for _ in range(train_steps_per_iteration)])
      train_time += time.time() - start

      iters_so_far += 1
      tf.summary.scalar('info/iters_so_far', iters_so_far, step=iters_so_far)
      global_step_val = global_step.value()

      if iters_so_far % log_interval == 0:
        metric_utils.log_metrics(train_metrics)
        time_acc += time.time() - start_time
        logging.info('step = %d, loss = %f', global_step_val.numpy(),
                     train_loss)
        print('step = %d, loss = %f', global_step_val.numpy(), train_loss)
        steps_per_sec = (
                            global_step_val.numpy() - timed_at_step.numpy()) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        print('%.3f steps/sec', steps_per_sec)

      if iters_so_far % summary_interval == 0:
        for train_metric in train_metrics:
          metric_value = train_metric.result()
          metric_name = f'Metrics/{train_metric.name}'
          tf.summary.scalar(metric_name, metric_value, step=iters_so_far)
        train_summary_writer.flush()

      if iters_so_far % eval_interval == 0:
        eval_start_time = time.time()
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=iters_so_far,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
            use_function=True,
        )
        eval_time = time.time() - eval_start_time
        print(f'eval_time: {eval_time}')
        metric_utils.log_metrics(eval_metrics)

      if iters_so_far % train_checkpoint_interval == 0:
        logging.info('Saving train checkpoint at step %d  (iteration %d)',
                     global_step_val.numpy(), iters_so_far)
        train_checkpointer.save(global_step=global_step_val)
        train_vocab_save_path = os.path.join(
            train_dir, f'vocab_{global_step_val.numpy()}.npy'
        )
        json.dump(
            dict(global_vocab._local_vocab), open(train_vocab_save_path, 'w')
        )
      if iters_so_far % policy_checkpoint_interval == 0:
        logging.info('Saving policy checkpoint at step %d  (iteration %d)',
                     global_step_val.numpy(), iters_so_far)
        save_location = os.path.join(
            saved_model_dir,
            'policy_' + str(environment_steps_metric.result().numpy()),
        )
        saved_model.save(save_location)
        policy_vocab_save_path = os.path.join(
            saved_model_dir,
            f'vocab_{environment_steps_metric.result().numpy()}.npy',
        )
        json.dump(
            dict(global_vocab._local_vocab), open(policy_vocab_save_path, 'w')
        )
      train_summary_writer.flush()

  # Save the final policy and vocabulary
  save_location = os.path.join(
      saved_model_dir,
      'policy_' + str(environment_steps_metric.result().numpy()),
  )
  saved_model.save(save_location)
  policy_vocab_save_path = os.path.join(
      saved_model_dir, f'vocab_{environment_steps_metric.result().numpy()}.npy'
  )
  json.dump(dict(global_vocab._local_vocab), open(policy_vocab_save_path, 'w'))
  tf_env.close()
  eval_tf_env.close()
  manager.shutdown()


def train_mp(_):
  # Extract environment variables
  batch_size = int(os.environ.get('BATCH_SIZE', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None))
  rb_capacity = int(os.environ.get('RB_CAPACITY', None))
  rb_checkpoint_interval = int(os.environ.get('RB_CHECKPOINT_INTERVAL', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  seed = int(os.environ.get('SEED', None))
  summary_interval = int(os.environ.get('SUMMARY_INTERVAL', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None))
  epsilon_greedy = float(os.environ.get('EPSILON_GREEDY', None))
  timesteps_per_actorbatch_param = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None))
  batched_total_env_steps = total_env_steps // env_batch_size
  timesteps_per_actorbatch = max(1,
                                 timesteps_per_actorbatch_param // env_batch_size)
  num_iterations = max(1, batched_total_env_steps // timesteps_per_actorbatch)

  # Print extracted and computed values
  print(f'difficulty_level: {difficulty_level}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'eval_interval: {eval_interval}')
  print(f'epsilon_greedy: {epsilon_greedy}')
  print(f'learning_rate: {learning_rate}')
  print(f'log_interval: {log_interval}')
  print(f'num_iterations: {num_iterations}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'rb_checkpoint_interval: {rb_checkpoint_interval}')
  print(f'root_dir: {root_dir}')
  print(f'seed: {seed}')
  print(f'summary_interval: {summary_interval}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'train_steps_per_iteration: {timesteps_per_actorbatch_param}')

  # Convert all of the intervals to be in terms of iterations instead of environment steps
  eval_interval = max(1, eval_interval // timesteps_per_actorbatch_param)
  train_checkpoint_interval = max(
      1, train_checkpoint_interval // timesteps_per_actorbatch_param
  )
  policy_checkpoint_interval = max(
      1, policy_checkpoint_interval // timesteps_per_actorbatch_param
  )
  rb_checkpoint_interval = max(
      1, rb_checkpoint_interval // timesteps_per_actorbatch_param
  )
  log_interval = max(1, log_interval // timesteps_per_actorbatch_param)
  summary_interval = max(1, summary_interval // timesteps_per_actorbatch_param)
  rb_capacity = max(1, rb_capacity // env_batch_size)

  print(f'eval_interval: {eval_interval}')
  print(f'log_interval: {log_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'rb_checkpoint_interval: {rb_checkpoint_interval}')
  print(f'summary_interval: {summary_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')

  train_eval(
      batch_size=batch_size,
      collect_steps_per_iteration=timesteps_per_actorbatch,
      debug_summaries=False,
      difficulty=difficulty_level,
      environment_batch_size=env_batch_size,
      eval_interval=eval_interval,
      initial_collect_steps=timesteps_per_actorbatch,
      learning_rate=learning_rate,
      log_interval=log_interval,
      num_iterations=num_iterations,
      policy_checkpoint_interval=policy_checkpoint_interval,
      rb_checkpoint_interval=rb_checkpoint_interval,
      replay_buffer_capacity=rb_capacity,
      root_dir=root_dir,
      seed=seed,
      epsilon_greedy=epsilon_greedy,
      summarize_grads_and_vars=False,
      summary_interval=summary_interval,
      train_checkpoint_interval=train_checkpoint_interval,
      train_steps_per_iteration=timesteps_per_actorbatch,
      use_tf_functions=False,
      env_args=dict(
          difficulty=difficulty_level,
          browser_args=dict(
              threading=False,
              chrome_options={
                  '--headless',
                  '--disable-gpu',
                  '--disable-dev-shm-usage',
                  '--no-sandbox',
              }
          )
      )
  )


def train():
  tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(train_mp)
