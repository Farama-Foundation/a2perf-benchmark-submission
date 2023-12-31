# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

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
import tensorflow as tf
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
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

from a2perf.domains.web_navigation.gwob.CoDE import networks
from a2perf.domains.web_navigation.gwob.CoDE import vocabulary_node

EMBEDDING_DIM = 100
LATENT_DIM = 50


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
    env_name='WebNavigation-v0',
    # Params for collect
    collect_steps_per_iteration=1,
    max_vocab_size=500,
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
  screenshot_dir = os.path.join(root_dir, 'screenshots')

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
  global_step = train_utils.create_train_step()
  manager = mp.Manager()
  global_vocab = vocabulary_node.LockedMultiprocessingVocabulary(
      max_vocabulary_size=max_vocab_size,
      multiprocessing_manager=manager,
  )
  env_args.update({'global_vocabulary': global_vocab})

  # Parallel environment creation
  # All envs get the same seed so that they see the same websites
  envs = [
      lambda: create_env(
          env_name=env_name,
          env_args=env_args,
      )
      for _ in range(environment_batch_size)
  ]
  eval_env_args = env_args.copy()
  eval_env_args.update(
      dict(
          render_mode='image',
          generate_screenshots=True,
          screenshot_save_dir=screenshot_dir,
          cyclic_action_penalty=0.0,
          timestep_penalty=0.0,
      )
  )

  eval_env = create_env(
      env_name=env_name,
      env_args=eval_env_args,
  )
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_env)
  parallel_py_env = tf_agents.environments.ParallelPyEnvironment(
      envs, blocking=False, start_serially=True, flatten=False
  )
  tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)
  observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(tf_env))

  logging.info('Successfully created environments')

  with tf.name_scope('Actor'):
    actor_net = networks.WebLSTMActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        name='actor',
        lstm_kwargs=dict(
            vocab_size=max_vocab_size
            if max_vocab_size is not None
            else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
            latent_dim=LATENT_DIM,
            profile_value_dropout=0.0,
            embedding_dim=EMBEDDING_DIM
        ),
    )
  logging.info('Successfully created actor network')

  with tf.name_scope('Value'):
    critic_net = networks.WebLSTMValueNetwork(
        input_tensor_spec=observation_spec,
        batch_squash=True,
        state_spec=tf.TensorSpec([1]),
        vocab_size=max_vocab_size
        if max_vocab_size is not None
        else tf_env.pyenv.envs[0].env.local_vocab.max_vocabulary_size,
        profile_value_dropout=0.0,
        embedding_dim=EMBEDDING_DIM,
        name='value',
        latent_dim=LATENT_DIM
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
      device='cpu:0',
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

  # Restore the vocab from the corresponding global step
  vocab_path = os.path.join(
      train_dir, f'vocab_{global_step.value().numpy()}.npy'
  )

  if os.path.exists(vocab_path):
    state = json.load(open(vocab_path, 'r'))
    global_vocab.restore(state)

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
      use_function=False,
      summary_prefix='Metrics',
      metrics=eval_metrics,

  )
  eval_tf_env.pyenv.envs[0].write_screenshots(
      screenshot_save_dir=screenshot_dir)

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
            'step = %d, loss = %f', global_step_val.numpy(), train_loss
        )
        print('step = %d, loss = %f', global_step_val.numpy(), train_loss)
        steps_per_sec = (
                            global_step_val.numpy() - timed_at_step.numpy()
                        ) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        print('%.3f steps/sec', steps_per_sec)

        # Info metrics indexed by the current global step (training step)
        tf.summary.scalar(
            'info/steps_per_sec', steps_per_sec, step=global_step_val
        )
        tf.summary.scalar(
            'info/collect_time', collect_time, step=global_step_val
        )
        tf.summary.scalar('info/train_time', train_time, step=global_step_val)
        tf.summary.scalar(
            'info/total_time', collect_time + train_time, step=global_step_val
        )
        tf.summary.scalar(
            'info/avg_train_return',
            train_metrics[2].result(),
            step=global_step_val,
        )
        tf.summary.scalar(
            'info/avg_train_episode_length',
            train_metrics[3].result(),
            step=global_step_val,
        )

        # Train metrics indexed by iters_so_far
        for train_metric in train_metrics:
          metric_value = train_metric.result()
          metric_name = f'Metrics/{train_metric.name}'
          tf.summary.scalar(metric_name, metric_value, step=iters_so_far)

        train_summary_writer.flush()

        time_acc = 0
        timed_at_step = global_step_val
        start_time = time.time()
        collect_time = 0
        train_time = 0

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
        eval_summary_writer.flush()

        # Eval environment generates screenshots
        eval_tf_env.pyenv.envs[0].write_screenshots(
            screenshot_save_dir=screenshot_dir)
        if iters_so_far % train_checkpoint_interval == 0:
          logging.info(
              'Saving train checkpoint at step %d  (iteration %d)',
              global_step_val.numpy(),
              iters_so_far,
          )
        train_checkpointer.save(global_step=global_step_val)
        train_vocab_save_path = os.path.join(
            train_dir, f'vocab_{global_step_val.numpy()}.npy'
        )
        json.dump(
            dict(global_vocab._local_vocab), open(train_vocab_save_path, 'w')
        )
        if iters_so_far % policy_checkpoint_interval == 0:
          logging.info(
              'Saving policy checkpoint at step %d  (iteration %d)',
              global_step_val.numpy(),
              iters_so_far,
          )
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
            saved_model_dir,
            f'vocab_{environment_steps_metric.result().numpy()}.npy'
        )
        json.dump(dict(global_vocab._local_vocab),
                  open(policy_vocab_save_path, 'w'))

        tf_env.close()
        eval_tf_env.close()
        manager.shutdown()


def train_mp(_):
  seed = int(os.environ.get('SEED', None))
  root_dir = os.environ.get('ROOT_DIR', None)
  env_batch_size = int(os.environ.get('ENV_BATCH_SIZE', None))
  total_env_steps = int(os.environ.get('TOTAL_ENV_STEPS', None))
  difficulty_level = int(os.environ.get('DIFFICULTY_LEVEL', None))
  eval_interval = int(os.environ.get('EVAL_INTERVAL', None))
  entropy_regularization = float(os.environ.get('ENTROPY_REGULARIZATION', None))
  train_checkpoint_interval = int(
      os.environ.get('TRAIN_CHECKPOINT_INTERVAL', None)
  )
  policy_checkpoint_interval = int(
      os.environ.get('POLICY_CHECKPOINT_INTERVAL', None)
  )
  log_interval = int(os.environ.get('LOG_INTERVAL', None))
  learning_rate = float(os.environ.get('LEARNING_RATE', None))
  num_websites = int(os.environ.get('NUM_WEBSITES', None))
  timesteps_per_actorbatch_param = int(
      os.environ.get('TIMESTEPS_PER_ACTORBATCH', None)
  )
  batched_total_env_steps = total_env_steps // env_batch_size
  timesteps_per_actorbatch = max(
      1, timesteps_per_actorbatch_param // env_batch_size
  )

  # Print extracted and computed values
  print(f'seed: {seed}')
  print(f'num_websites: {num_websites}')
  print(f'root_dir: {root_dir}')
  print(f'env_batch_size: {env_batch_size}')
  print(f'total_env_steps: {total_env_steps}')
  print(f'difficulty_level: {difficulty_level}')
  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  print(f'learning_rate: {learning_rate}')

  # Convert all of the intervals to be in terms of iterations instead of environment steps
  eval_interval = max(1, eval_interval // timesteps_per_actorbatch_param)
  train_checkpoint_interval = max(
      1, train_checkpoint_interval // timesteps_per_actorbatch_param
  )
  policy_checkpoint_interval = max(
      1, policy_checkpoint_interval // timesteps_per_actorbatch_param
  )
  log_interval = max(1, log_interval // timesteps_per_actorbatch_param)

  print(f'eval_interval: {eval_interval}')
  print(f'train_checkpoint_interval: {train_checkpoint_interval}')
  print(f'policy_checkpoint_interval: {policy_checkpoint_interval}')
  print(f'log_interval: {log_interval}')
  train_eval(
      collect_steps_per_iteration=timesteps_per_actorbatch,
      debug_summaries=False,
      environment_batch_size=env_batch_size,
      eval_interval=eval_interval,
      log_interval=log_interval,
      num_eval_episodes=10,
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      policy_checkpoint_interval=policy_checkpoint_interval,
      root_dir=root_dir,
      seed=seed,
      summarize_grads_and_vars=False,
      total_env_steps=batched_total_env_steps,
      train_checkpoint_interval=train_checkpoint_interval,
      use_tf_functions=False,
      entropy_regularization=entropy_regularization,
      env_args=dict(
          difficulty=difficulty_level,
          use_legacy_reset=True,
          use_legacy_step=True,
          num_websites=num_websites,
          browser_args=dict(
              threading=False,
              chrome_options=['--headless', '--disable-dev-shm-usage',
                              '--no-sandbox'],
          ),
      ),
  )


def train():
  tf_agents.system.multiprocessing.handle_main(train_mp)


if __name__ == '__main__':
  tf_agents.system.multiprocessing.handle_main(
      functools.partial(app.run, train_mp)
  )
