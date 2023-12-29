from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import multiprocessing as mp
import os
import random
import time

import gin
import gymnasium as gym
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import tf_agents
from absl import app
from absl import logging
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256


def create_env(
    env_name='CartPole-v0',
    env_args=None,
):
  return suite_gym.load(
      environment_name=env_name,
      spec_dtype_map={gym.spaces.Discrete: np.int32},
      gym_kwargs=env_args,
  )


@gin.configurable
def train_eval(
    root_dir,
    env_name='QuadrupedLocomotion-v0',
    # Params for collect
    collect_steps_per_iteration=1,
    # Params for train
    optimizer=None,
    entropy_regularization=0.0,
    eval_interval=1,
    total_env_steps=100000,
    replay_buffer_capacity=1001,
    environment_batch_size=1,
    gradient_clipping=None,
    use_tf_functions=False,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    # Params for summaries and logging
    num_eval_episodes=1,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    env_args=None,
    seed=0,
):
  """A simple train and eval for PPO."""
  tf.random.set_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if env_args:
    env_args.update({'seed': seed})

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
      max_queue=10,
      experimental_trackable=True,
  )
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]
  global_step = tf.compat.v1.train.get_or_create_global_step()
  manager = mp.Manager()

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

  with tf.name_scope('Actor'):

    # Should be simple [512, 256] network
    actor_net = tf_agents.networks.actor_distribution_network.ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        fc_layer_params=[512, 256],
        activation_fn=tf.nn.relu,
        continuous_projection_net=None,
        name='ActorNetwork',
    )
  logging.info('Successfully created actor network')
  # state spec must be defined so that it can be used to reshape our value predictions to [B, T, 1]
  with tf.name_scope('Value'):
    critic_net = tf_agents.networks.value_network.ValueNetwork(
        input_tensor_spec=observation_spec,
        fc_layer_params=[512, 256],
        activation_fn=tf.nn.relu,
        name='ValueNetwork',
    )
  logging.info('Successfully created value network')

  with tf.name_scope('PPOAgent'):
    tf_agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=critic_net,
        gradient_clipping=gradient_clipping,

        entropy_regularization=entropy_regularization,
        greedy_eval=False,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
    )

  logging.info('Successfully created PPO agent')
  tf_agent.initialize()
  logging.info('Successfully initialized PPO agent')

  environment_steps_metric = tf_metrics.EnvironmentSteps()
  step_metrics = [
      tf_metrics.NumberOfEpisodes(),
      environment_steps_metric,
  ]
  train_metrics = step_metrics + [
      tf_metrics.AverageReturnMetric(
          batch_size=environment_batch_size, buffer_size=num_eval_episodes
      ),
      tf_metrics.AverageEpisodeLengthMetric(
          batch_size=environment_batch_size, buffer_size=num_eval_episodes
      ),
  ]

  eval_policy = tf_agent.policy
  collect_policy = tf_agent.collect_policy

  # Make the replay buffer. Cleared after every iteration
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      tf_agent.collect_data_spec,
      batch_size=environment_batch_size,
      max_length=replay_buffer_capacity,
      device='cpu:*',
  )
  logging.info('Successfully created replay buffer')

  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=tf_agent,
      global_step=global_step,
      max_to_keep=1,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train'),
  )
  saved_model = policy_saver.PolicySaver(eval_policy, train_step=global_step)
  saved_model_dir = os.path.join(root_dir, 'policies')
  train_checkpointer.initialize_or_restore()

  collect_driver = dynamic_step_driver.DynamicStepDriver(
      env=tf_env,
      policy=collect_policy,
      observers=[replay_buffer.add_batch] + train_metrics,
      num_steps=collect_steps_per_iteration,
  )

  def train_step():
    trajectories = replay_buffer.gather_all()
    replay_buffer.clear()  # clear within tf.function
    return tf_agent.train(experience=trajectories)

  if use_tf_functions:
    collect_driver.run = common.function(collect_driver.run, autograph=False)
    tf_agent.train = common.function(tf_agent.train, autograph=False)

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
      metric_name = f'Metrics/{train_metric.name}'
      tf.summary.scalar(metric_name, metric_value, step=iters_so_far)
    tf.summary.scalar('info/iters_so_far', iters_so_far, step=iters_so_far)

  logging.info('Beginning training at step: %d', global_step.value().numpy())
  with train_summary_writer.as_default():
    while environment_steps_metric.result().numpy() < total_env_steps:
      start = time.time()
      collect_driver.run()
      collect_time += time.time() - start

      start = time.time()
      train_loss = train_step()
      train_time += time.time() - start

      iters_so_far += 1
      tf.summary.scalar('info/iters_so_far', iters_so_far, step=iters_so_far)
      global_step_val = global_step.value()

      if iters_so_far % log_interval == 0:
        metric_utils.log_metrics(train_metrics)
        time_acc += time.time() - start_time
        logging.info(
            'step = %d, loss = %f', global_step_val.numpy(), train_loss.loss
        )
        print('step = %d, loss = %f', global_step_val.numpy(), train_loss.loss)
        steps_per_sec = (
                            global_step_val.numpy() - timed_at_step.numpy()
                        ) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        print('%.3f steps/sec', steps_per_sec)

        # Add number of iters per second to the train summary writer
        tf.summary.scalar(
            name='info/global_steps_per_sec',
            data=steps_per_sec,
            step=iters_so_far,
        )
        tf.summary.scalar(
            name='info/collect_time', data=collect_time, step=iters_so_far
        )
        tf.summary.scalar(
            name='info/train_time', data=train_time, step=iters_so_far
        )
        print(f'collect_time: {collect_time}')
        print(f'train_time: {train_time}')

        timed_at_step = global_step_val
        time_acc = 0
        collect_time = 0
        train_time = 0
        start_time = time.time()

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

      if iters_so_far % policy_checkpoint_interval == 0:
        logging.info('Saving policy checkpoint at step %d  (iteration %d)',
                     global_step_val.numpy(), iters_so_far)
        save_location = os.path.join(
            saved_model_dir,
            'policy_' + str(environment_steps_metric.result().numpy()),
        )
        saved_model.save(save_location)

      train_summary_writer.flush()

  # Save the final policy
  save_location = os.path.join(
      saved_model_dir,
      'policy_' + str(environment_steps_metric.result().numpy()),
  )
  saved_model.save(save_location)

  tf_env.close()
  eval_tf_env.close()
  manager.shutdown()


def train():
  root_dir = os.environ['ROOT_DIR']
  seed = int(os.environ['SEED'])
  total_timesteps = int(os.environ['TOTAL_ENV_STEPS'])
  parallel_mode = os.environ['PARALLEL_MODE']
  parallel_cores = int(os.environ['PARALLEL_CORES'])
  mode = os.environ['MODE']
  visualize = bool(os.environ['VISUALIZE'])
  int_save_freq = int(os.environ['INT_SAVE_FREQ'])
  setup_path = os.environ['SETUP_PATH']
  motion_file_path = os.environ['MOTION_FILE_PATH']
  timesteps_per_actorbatch = int(
      np.ceil(float(TIMESTEPS_PER_ACTORBATCH) / parallel_cores))
  optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / parallel_cores))
  output_dir = root_dir
  print("root_dir:", root_dir)
  print("seed:", seed)
  print("total_timesteps:", total_timesteps)
  print("parallel_mode:", parallel_mode)
  print("parallel_cores:", parallel_cores)
  print("mode:", mode)
  print("visualize:", visualize)
  print("int_save_freq:", int_save_freq)
  print("setup_path:", setup_path)
  print("timesteps_per_actorbatch:", timesteps_per_actorbatch)
  print("optim_batchsize:", optim_batchsize)

  train_eval(
      root_dir=root_dir,
      env_name='QuadrupedLocomotion-v0',
      collect_steps_per_iteration=TIMESTEPS_PER_ACTORBATCH // parallel_cores,
      optimizer=None,
      entropy_regularization=0.0,
      eval_interval=1,
      total_env_steps=total_timesteps,
      replay_buffer_capacity=TIMESTEPS_PER_ACTORBATCH // parallel_cores,
      environment_batch_size=parallel_cores,
      gradient_clipping=None,
      use_tf_functions=False,
      train_checkpoint_interval=int_save_freq,
      policy_checkpoint_interval=int_save_freq,
      num_eval_episodes=1,
      env_args=dict(motion_files=[motion_file_path], mode=mode, )
  )


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
